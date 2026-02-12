import torch

from denoise import windowed_mask_fresnel_abs_sum
from fresnel_rec import FresnelReqTorch

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

#print cuda device info:
if device.type == "cuda":
    print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
#device = torch.device("cpu")
# --- experiment params (set your real ones) ---
d = 0.34             # meters
wavelength = 532e-9  # meters
pixelsize = 1.67e-6  # meters
thetain_deg = 0.0    # degrees

fresnel = FresnelReqTorch(d=d, wavelength=wavelength, pixelsize=pixelsize, thetain_deg=thetain_deg)

# example image
image = torch.randn(1024, 1024, device=device, dtype=torch.float32)

out = windowed_mask_fresnel_abs_sum(
    image=image,                 # (H,W) or (B,H,W)
    mask_type="tukey",
    fresnel_req=fresnel,         # <- plug-in here
    win_hw=(128, 128),
    step=(32, 32),
    mask_kwargs={"alpha": 0.5, "normalize": True},
    normalize_overlap=True,
)

print(out.shape)  # (1,H,W) if input was (H,W)
