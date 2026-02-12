import torch
import torch.nn.functional as F
from masks import mask_rect_torch, mask_circ_torch, mask_tukey_torch

def _get_mask_from_file(mask_type: str, win_hw, device, dtype, mask_kwargs=None):
    mask_kwargs = mask_kwargs or {}
    mt = mask_type.lower()
    if mt in ["rect", "rectangle", "box"]:
        m = mask_rect_torch(win_hw, device=device, dtype=dtype)
    elif mt in ["circ", "circle", "circular"]:
        m = mask_circ_torch(win_hw, device=device, dtype=dtype)
    elif mt in ["tukey", "taper", "tapered"]:
        alpha = mask_kwargs.get("alpha", 0.5)
        normalize = mask_kwargs.get("normalize", True)
        m = mask_tukey_torch(win_hw, alpha=alpha, device=device, dtype=dtype, normalize=normalize)
    else:
        raise ValueError(f"Unknown mask_type='{mask_type}'.")
    return m

def windowed_mask_fresnel_abs_sum(image, mask_type, fresnel_req, win_hw=(128, 128), step=(32, 32), mask_kwargs=None, normalize_overlap=True):
    if mask_kwargs is None: mask_kwargs = {}
    if image.ndim == 2: x = image[None, None, :, :]
    elif image.ndim == 3: x = image[:, None, :, :]
    else: x = image
    
    B, C, H, W = x.shape
    device = x.device
    dtype = x.dtype
    winH, winW = win_hw
    sy, sx = step

    mask = _get_mask_from_file(mask_type, win_hw, device=device, dtype=dtype, mask_kwargs=mask_kwargs)
    mask_ = mask.view(1, 1, winH, winW)

    # All on GPU
    cols = F.unfold(x, kernel_size=(winH, winW), stride=(sy, sx))
    L = cols.shape[-1]
    patches = cols.transpose(1, 2).reshape(B, L, winH, winW)
    
    masked = patches * mask_
    
    masked_flat = masked.reshape(B * L, winH, winW)
    recon_flat = fresnel_req(masked_flat)
    mag_flat = torch.abs(recon_flat)
    mag = mag_flat.reshape(B, L, winH, winW)
    
    mag_cols = mag.reshape(B, L, winH * winW).transpose(1, 2)
    out = F.fold(mag_cols, output_size=(H, W), kernel_size=(winH, winW), stride=(sy, sx))

    if normalize_overlap:
        w = mask.reshape(1, winH * winW, 1).to(device=device, dtype=dtype)
        wcols = w.expand(B, -1, L).contiguous()
        wsum = F.fold(wcols, output_size=(H, W), kernel_size=(winH, winW), stride=(sy, sx))
        out = out / (wsum + 1e-8)

    return out[:, 0]