import torch
from threestudio.models.guidance.zero123_guidance import Zero123Guidance
from types import SimpleNamespace
from torchvision.transforms import functional as TF
import PIL.Image as Image

# ---- Step 1: Define config ----
cfg = SimpleNamespace(
    pretrained_model_name_or_path="load/zero123/105000.ckpt",
    pretrained_config="load/zero123/sd-objaverse-finetune-c_concat-256.yaml",
    vram_O=False,  # Needed for decoding image
    cond_image_path="load/images/hamburger_rgba.png",
    cond_elevation_deg=0.0,
    cond_azimuth_deg=0.0,
    cond_camera_distance=1.2,
    guidance_scale=5.0,
    guidance_scale_aux=5.0,
    precomputed_scale=None,
    cond_fov_deg=30.0,
    grad_clip=None,
    half_precision_weights=False,
    min_step_percent=0.02,
    max_step_percent=0.98,
    max_items_eval=4,
    p_use_aux_cameras=0.0,
    use_anisotropic_schedule=False,
    anisotropic_offset=0,
    depth_threshold_for_anchor_guidance=0.,
)

# ---- Step 2: Load the model ----

guidance = Zero123Guidance(cfg)
guidance.cuda()
guidance.configure()

# ---- Step 3: Setup camera for new view ----
camera = {
    "azimuth": torch.tensor([60.0], device="cuda"),  # desired azimuth
    "elevation": torch.tensor([15.0], device="cuda"),  # desired elevation
    "camera_distances": torch.tensor([1.2], device="cuda"),  # same as training
    "fov_deg": torch.tensor([30.0], device="cuda"),
    "c2w": torch.eye(4).unsqueeze(0).to("cuda"),  # dummy identity
    "canonical_c2w": torch.eye(4).unsqueeze(0).to("cuda"),
    "aux_c2ws": [],  # no auxiliary views
}

# ---- Step 4: Generate new view ----
cond, _ = guidance.get_cond(camera)
new_view = guidance.gen_from_cond(cond, ddim_steps=50, scale=5.0)[0]  # shape: (H, W, 3)

# ---- Step 5: Save or visualize result ----
new_view_img = Image.fromarray((new_view * 255).astype("uint8"))
new_view_img.save("novel_view.png")
