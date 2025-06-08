import torch
import cv2
import numpy as np
from omegaconf import OmegaConf
from ldm.data import common
from diffusers import DDIMScheduler
from ldm.util import instantiate_from_config
from ldm.models.diffusion import options
from PIL import Image
import math
import torch.nn.functional as F
options.LDM_DISTILLATION_ONLY = True

def generate_c2ws(default_elv=20.0, default_fovy=50.0, azimuths_deg=[60, -60], default_camera_distance=1.0, device="cuda"):
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

    #Canonical form
    canonical_c2w = get_defaults(azimuth)

    #Now generate different azimuths
    def get_sides(azimuths_deg):
        azimuths_deg = torch.tensor(azimuths_deg, dtype=torch.float32)
        azimuths = azimuths_deg * math.pi / 180
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

    c2ws = get_sides(azimuths_deg)

    #Common fovy for all cases
    fovy = torch.deg2rad(torch.FloatTensor([default_fovy]))

    #Returns: canonical c2w,camera_position and target c2w,camera_positions
    #And fovy
    return canonical_c2w, c2ws, fovy

def load_model(config_path, ckpt_path, device='cuda'):
    #Taken from zero123_guidance
    config = OmegaConf.load(config_path)
    #Set depth model to None
    config.model.params.conditioning_config.params.depth_model_name = None
    model = instantiate_from_config(config.model)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model, config

def encode_condition_image(model, image_path, device):
    rgba = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
    rgba = cv2.resize(rgba, (256, 256)).astype(np.float32) / 255.0
    rgb = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])
    rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    img = rgb_tensor * 2 - 1
    c_crossattn = model.get_learned_conditioning(img)
    c_concat = model.encode_first_stage(img).mode()
    return c_crossattn, c_concat

def make_condition(config, model, c_crossattn, c_concat, target_c2w, canonical_c2w, fov_deg, precomputed_scale, device):
    #Modified from zero123_guidance

    #Generate batch
    target_cam2world = target_c2w.to(device)
    batch_size = target_cam2world.shape[0]
    cond_cam2world = canonical_c2w.to(device).broadcast_to(target_cam2world.shape)
    fov_deg = fov_deg.to(device).repeat(batch_size)
    batch = {
        "target_cam2world": target_cam2world,
        "cond_cam2world": cond_cam2world,
        "fov_deg": fov_deg,
    }

    #Generate T conditioning
    T = common.compute_T(config.model.params.conditioning_config, None, batch, precomputed_scale=precomputed_scale)
    T = T[:, None, :].to(device)

    #For unbatched and batched crossattn
    b_repeats = len(T) // len(c_crossattn)
    assert len(c_crossattn) == len(c_concat)

    #Generate embedding
    clip_input = torch.cat([c_crossattn.repeat(b_repeats, 1, 1), T], dim=-1)
    clip_emb = model.cc_projection(clip_input)

    #Final conditioning
    cond = {
        "c_crossattn": [torch.cat([torch.zeros_like(clip_emb), clip_emb], dim=0)],
        "c_concat": [torch.cat([
            torch.zeros_like(c_concat).repeat(b_repeats, 1, 1, 1),
            c_concat.repeat(b_repeats, 1, 1, 1)
        ], dim=0)]
    }
    return cond

def decode_latents(model, latents):
    input_dtype = latents.dtype
    image = model.decode_first_stage(latents)
    image = (image * 0.5 + 0.5).clamp(0, 1)
    return image.to(input_dtype)

def diffuse(model, cond, scheduler, device, scale=3, ddim_steps=50):
    B = cond["c_crossattn"][0].shape[0] // 2
    latents = torch.randn((B, 4, 32, 32), device=device)
    scheduler.set_timesteps(ddim_steps)
    for t in scheduler.timesteps:
        x_in = torch.cat([latents] * 2)
        t_in = torch.cat([t.reshape(1).repeat(B)] * 2).to(device)
        noise_pred = model.apply_model(x_in, t_in, cond)
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + scale * (noise_pred_cond - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents

if __name__ == "__main__":
    device = "cuda"
    config_path = "zeronvs_config.yaml"
    ckpt_path = "zeronvs.ckpt"
    image_path = "smallmoto.png"
    precomputed_scale = 0.7
    #Setup model
    model, config = load_model(config_path, ckpt_path, device)
    #Setup scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=config.model.params.timesteps,
        beta_start=config.model.params.linear_start,
        beta_end=config.model.params.linear_end,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    #Generate image conditioning
    c_crossattn, c_concat = encode_condition_image(model, image_path, device)
    #Generate camera pose conditioning
    can_c2w, target_c2ws, fovy = generate_c2ws(azimuths_deg=[60, -60], device=device)
    #Generate full conditioning
    conds = make_condition(config, model, c_crossattn, c_concat, target_c2ws, can_c2w, fovy, precomputed_scale, device)

    #Generate latents
    latents = diffuse(model, conds, scheduler, device, scale=7.5, ddim_steps=20)

    #Decode for images
    imgs = decode_latents(model, latents)
    imgs = imgs.detach().cpu().numpy().transpose(0, 2, 3, 1)
    for i, img in enumerate(imgs):
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        img_pil.save(f"output_view_{i}.png")
