import torch
import cv2
import numpy as np
import math
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.data import common
from diffusers import DDIMScheduler
from ldm.models.diffusion import options
options.LDM_DISTILLATION_ONLY = True
import time

class Zero123Generator:
    def __init__(self, config_path, ckpt_path, device="cuda", precomputed_scale=0.7):
        self.device = device
        self.precomputed_scale = precomputed_scale
        self.model, self.config = self.load_model(config_path, ckpt_path)
        self.scheduler = DDIMScheduler(
            num_train_timesteps=self.config.model.params.timesteps,
            beta_start=self.config.model.params.linear_start,
            beta_end=self.config.model.params.linear_end,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        self.previous_latents = None
        self.threshold = 0.1

    def load_model(self, config_path, ckpt_path):
        #Taken from zero123_guidance
        config = OmegaConf.load(config_path)
        #Set depth model to None
        config.model.params.conditioning_config.params.depth_model_name = None
        model = instantiate_from_config(config.model)
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        return model.to(self.device).eval(), config

    def encode_image(self, img):
        c_crossattn = self.model.get_learned_conditioning(img)
        c_concat = self.model.encode_first_stage(img).mode()
        return c_crossattn, c_concat

    def get_image(self, image_path):
        rgb = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return self.process_image(rgb)

    def process_image(self, rgb):
        # print(rgb.shape)
        rgb = cv2.resize(rgb, (256, 256)).astype(np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img = rgb_tensor * 2 - 1
        return img

    def generate_c2ws(self, gen_azm=[60, -60], gen_elv=20, default_elv=30.0, default_fovy=60.0, default_camera_distance=1.0):
        #Generates canonical c2w and generate-able c2ws
        #Modified from threestudio/data/image/SingleImageDataBase
        elevation_deg = torch.FloatTensor([default_elv])
        azimuth_deg = torch.FloatTensor([0])
        camera_distance = torch.FloatTensor([default_camera_distance])
        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        def get_defaults(azimuth):
            #Taken from threestudio/data/image/SingleImageDataBase
            camera_position: Float[Tensor, "1 3"] = torch.stack(
                [
                    camera_distance * torch.cos(elevation) * torch.cos(azimuth),
                    camera_distance * torch.cos(elevation) * torch.sin(azimuth),
                    camera_distance * torch.sin(elevation),
                ],
                dim=-1,
            )
            center: Float[Tensor, "1 3"] = torch.zeros_like(camera_position)
            up: Float[Tensor, "1 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
                None
            ]
            lookat: Float[Tensor, "1 3"] = F.normalize(center - camera_position, dim=-1)
            right: Float[Tensor, "1 3"] = F.normalize(torch.cross(lookat, up, dim=-1), dim=-1)
            up = F.normalize(torch.cross(right, lookat, dim=-1), dim=-1)
            c2w_3x4: Float[Tensor, "1 3 4"] = torch.cat(
                [
                    torch.stack([right, up, -lookat], dim=-1),
                    camera_position[:, :, None],
                ],
                dim=-1,
            )
            bottom_row = torch.tensor([0, 0, 0, 1], dtype=torch.float32).view(1, 1, 4)
            c2w = torch.cat([c2w_3x4, bottom_row], dim=1)
            return c2w

        #Canonical cam
        canonical_c2w = get_defaults(azimuth)
        #Now generate different azimuths and custom elevation_deg
        elevation_deg = torch.FloatTensor([gen_elv])
        elevation = elevation_deg * math.pi / 180
        def get_sides(gen_azm):
            gen_azm = torch.tensor(gen_azm, dtype=torch.float32)
            azimuths = gen_azm * math.pi / 180
            distances = torch.full_like(azimuths, default_camera_distance)
            camera_positions = torch.stack([
                distances * torch.cos(elevation) * torch.cos(azimuths),
                distances * torch.cos(elevation) * torch.sin(azimuths),
                distances * torch.sin(elevation).expand_as(azimuths),
            ], dim=-1)

            center = torch.zeros_like(camera_positions)
            up = torch.tensor([0, 0, 1], dtype=torch.float32).expand_as(camera_positions)

            lookat = F.normalize(center - camera_positions, dim=-1)
            right = F.normalize(torch.cross(lookat, up, dim=-1), dim=-1)
            up_corrected = F.normalize(torch.cross(right, lookat, dim=-1), dim=-1)

            c2w_3x4 = torch.cat([right.unsqueeze(-1), up_corrected.unsqueeze(-1), -lookat.unsqueeze(-1), camera_positions.unsqueeze(-1)], dim=-1)
            bottom_row = torch.tensor([0, 0, 0, 1], dtype=torch.float32).view(1, 1, 4).expand(len(azimuths), -1, -1)
            c2w = torch.cat([c2w_3x4, bottom_row], dim=1)
            return c2w

        target_c2ws = get_sides(gen_azm)

        #Common fovy for all cases
        fovy = torch.deg2rad(torch.FloatTensor([default_fovy]))

        #Returns: canonical c2w,camera_position and target c2w,camera_positions
        #And fovy
        return canonical_c2w, target_c2ws, fovy

    def make_condition(self, c_crossattn, c_concat, target_c2w, canonical_c2w, fov_deg):
        #Modified from zero123_guidance
        #Generate batch
        batch_size = target_c2w.shape[0]
        cond_cam2world = canonical_c2w.to(self.device).expand_as(target_c2w)
        fov_deg = fov_deg.to(self.device).repeat(batch_size)
        batch = {"target_cam2world": target_c2w.to(self.device), "cond_cam2world": cond_cam2world, "fov_deg": fov_deg}
        #Generate T conditioning
        T = common.compute_T(self.config.model.params.conditioning_config, None, batch, precomputed_scale=self.precomputed_scale)
        T = T[:, None, :].to(self.device)
        #For unbatched and batched crossattn
        b_repeats = len(T) // len(c_crossattn)
        #Generate embedding
        clip_input = torch.cat([c_crossattn.repeat(b_repeats, 1, 1), T], dim=-1)
        clip_emb = self.model.cc_projection(clip_input)
        #Final conditioning
        cond = {
            "c_crossattn": [torch.cat([torch.zeros_like(clip_emb), clip_emb], dim=0)],
            "c_concat": [torch.cat([
                torch.zeros_like(c_concat).repeat(len(T), 1, 1, 1),
                c_concat.repeat(len(T), 1, 1, 1)
            ], dim=0)]
        }
        return cond

    def diffuse(self, cond, scale=3, ddim_steps=50):
        # start = time.time()
        B = cond["c_crossattn"][0].shape[0] // 2
        latents = torch.randn((B, 4, 32, 32), device=self.device)
        self.scheduler.set_timesteps(ddim_steps)
        for t in self.scheduler.timesteps:
            x_in = torch.cat([latents] * 2)
            t_in = torch.cat([t.reshape(1).repeat(B)] * 2).to(self.device)
            noise_pred = self.model.apply_model(x_in, t_in, cond)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + scale * (noise_pred_cond - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]
        # print(f"Time Taken: {time.time() - start}")
        return latents

    def reset_latent(self):
        self.previous_latents = None

    # def diffuse_with_jump(self, cond, scale=3, ddim_steps=50):
    #     B = cond["c_crossattn"][0].shape[0] // 2
    #     #First check if previous_latent is none
    #     self.scheduler.set_timesteps(ddim_steps)
    #     store_index = ddim_steps - 4
    #     #For the first step perform full step and store a checkpoint
    #     if self.previous_latents is None:
    #         latents = torch.randn((B, 4, 32, 32), device=self.device)
    #         store_time = self.scheduler.timesteps[store_index]
    #         for t in self.scheduler.timesteps:
    #             x_in = torch.cat([latents] * 2)
    #             t_in = torch.cat([t.reshape(1).repeat(B)] * 2).to(self.device)
    #             noise_pred = self.model.apply_model(x_in, t_in, cond)
    #             noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
    #             noise_pred = noise_pred_uncond + scale * (noise_pred_cond - noise_pred_uncond)
    #             latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]
    #             if t == store_time:
    #                 self.previous_latents = latents.detach()
    #     else:
    #         #If previous latent is found, start from the stored latent
    #         timesteps = self.scheduler.timesteps[store_index: ddim_steps] #For example, from 45 to 50
    #         noise = torch.randn_like(self.previous_latents)
    #         latents = 0.9 * self.previous_latents + 0.1 * noise
    #         # latents = self.previous_latents
    #         for t in timesteps:
    #             x_in = torch.cat([latents] * 2)
    #             t_in = torch.cat([t.reshape(1).repeat(B)] * 2).to(self.device)
    #             noise_pred = self.model.apply_model(x_in, t_in, cond)
    #             noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
    #             noise_pred = noise_pred_uncond + scale * (noise_pred_cond - noise_pred_uncond)
    #             latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]
    #         # self.previous_latents = latents.detach()
    #     return latents

    def decode_latents(self, latents):
        input_dtype = latents.dtype
        image = self.model.decode_first_stage(latents)
        image = (image * 0.5 + 0.5).clamp(0, 1).to(input_dtype).detach().cpu().numpy().transpose(0, 2, 3, 1)
        resized_image = np.array([cv2.resize(img, (128, 128)) for img in image])
        resized_image = (resized_image*255).astype(np.uint8)
        return resized_image

    def generate_latents(self, img, gen_azm=[60, -60], gen_elv=20, default_elv=30.0, default_fovy=90.0, default_camera_distance=1.0, scale=7.5, ddim_steps=20):
        c_crossattn, c_concat = self.encode_image(img)
        can_c2w, target_c2ws, fovy = self.generate_c2ws(gen_azm=gen_azm, gen_elv=gen_elv, default_elv=default_elv, default_fovy=default_fovy, default_camera_distance=default_camera_distance)
        cond = self.make_condition(c_crossattn, c_concat, target_c2ws, can_c2w, fovy)
        with torch.no_grad():
            latents = self.diffuse(cond, scale=scale, ddim_steps=ddim_steps)
        self.previous_latents = latents.detach()
        return latents

    # def generate_latents_with_jump(self, img, gen_azm=[60, -60], gen_elv=20, default_elv=30.0, default_fovy=90.0, default_camera_distance=1.0, scale=7.5, ddim_steps=20):
    #     c_crossattn, c_concat = self.encode_image(img)
    #     can_c2w, target_c2ws, fovy = self.generate_c2ws(gen_azm=gen_azm, gen_elv=gen_elv, default_elv=default_elv, default_fovy=default_fovy, default_camera_distance=default_camera_distance)
    #     cond = self.make_condition(c_crossattn, c_concat, target_c2ws, can_c2w, fovy)
    #     with torch.no_grad():
    #         latents = self.diffuse_with_jump(cond, scale=scale, ddim_steps=ddim_steps)
    #     self.previous_latents = latents.detach()
    #     return latents

    def generate_views_from_latents(self, latents):
        images = self.decode_latents(latents)
        outputs = [Image.fromarray(img) for img in images]
        return outputs
bac