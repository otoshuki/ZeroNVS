import torch
import cv2
import numpy as np
from omegaconf import OmegaConf
from ldm.data import common
from diffusers import DDIMScheduler
from ldm.util import instantiate_from_config
from PIL import Image

def load_model(config_path, ckpt_path, device='cuda'):
    config = OmegaConf.load(config_path)
    config.model.params.conditioning_config.params.depth_model_name = None
    model = instantiate_from_config(config.model)
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
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

def make_condition(model, c_crossattn, c_concat, config, c2w, fov_deg, device):
    batch = {
        "target_cam2world": torch.tensor(c2w, dtype=torch.float32).unsqueeze(0).to(device),
        "cond_cam2world": torch.tensor(c2w, dtype=torch.float32).unsqueeze(0).to(device),
        "fov_deg": torch.tensor([fov_deg], dtype=torch.float32).to(device),
    }
    T = common.compute_T(config.model.params.conditioning_config, None, batch, precomputed_scale=None)
    clip_emb = model.cc_projection(torch.cat([c_crossattn, T[:, None, :].to(device)], dim=-1))
    return {
        "c_crossattn": [torch.cat([torch.zeros_like(clip_emb), clip_emb], dim=0)],
        "c_concat": [torch.cat([torch.zeros_like(c_concat), c_concat], dim=0)]
    }

def generate_image(model, cond, scheduler, device, scale=3, ddim_steps=50):
    latents = torch.randn((1, 4, 32, 32), device=device)
    scheduler.set_timesteps(ddim_steps)
    for t in scheduler.timesteps:
        x_in = torch.cat([latents] * 2)
        t_in = torch.cat([t.reshape(1).repeat(2)] * 1).to(device)
        noise_pred = model.apply_model(x_in, t_in, cond)
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + scale * (noise_pred_cond - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents)["prev_sample"]
    img = model.decode_first_stage(latents).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)

# --- Main usage ---
if __name__ == "__main__":
    device = "cuda"
    config_path = "zeronvs_config.yaml"
    ckpt_path = "zeronvs.ckpt"
    image_path = "smallmoto.png"

    model, config = load_model(config_path, ckpt_path, device)
    scheduler = DDIMScheduler(
        num_train_timesteps=config.model.params.timesteps,
        beta_start=config.model.params.linear_start,
        beta_end=config.model.params.linear_end,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    c_crossattn, c_concat = encode_condition_image(model, image_path, device)
    c2w = np.eye(4)  # Change this to desired camera pose
    cond = make_condition(model, c_crossattn, c_concat, config, c2w, fov_deg=50.0, device=device)

    result_img = generate_image(model, cond, scheduler, device)
    Image.fromarray(result_img).save("output_view.png")
