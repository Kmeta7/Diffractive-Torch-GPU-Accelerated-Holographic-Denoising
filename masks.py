import math
import torch

def mask_rect_torch(win_hw, device=None, dtype=torch.float32):
    winH, winW = win_hw
    return torch.ones((winH, winW), device=device, dtype=dtype)

def mask_circ_torch(win_hw, device=None, dtype=torch.float32):
    winH, winW = win_hw
    center_y, center_x = winH / 2, winW / 2
    Y, X = torch.meshgrid(torch.arange(winH, device=device), torch.arange(winW, device=device))
    dist_from_center = torch.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    radius = min(center_x, center_y)
    mask = (dist_from_center <= radius).float()
    return mask.to(dtype)

def tukey_1d_torch(N, alpha=0.5, device=None, dtype=torch.float32):
    n = torch.arange(N, device=device, dtype=dtype)
    w = torch.ones(N, device=device, dtype=dtype)
    if alpha <= 0:
        return w
    if alpha >= 1:
        return 0.5 - 0.5 * torch.cos(2*math.pi*n/(N-1))

    edge = int(math.floor(alpha*(N-1)/2))
    if edge > 0:
        k = n[:edge] / edge
        w[:edge] = 0.5*(1 + torch.cos(math.pi*(k - 1)))
        k2 = (n[-edge:] - (N-1-edge)) / edge
        w[-edge:] = 0.5*(1 + torch.cos(math.pi*(k2)))
    return w

def mask_tukey_torch(win_hw, alpha=0.5, device=None, dtype=torch.float32, normalize=True):
    winH, winW = win_hw
    wy = tukey_1d_torch(winH, alpha, device=device, dtype=dtype)
    wx = tukey_1d_torch(winW, alpha, device=device, dtype=dtype)
    m = wy[:, None] * wx[None, :]
    if normalize:
        m = m / (m.sum() + 1e-12)
    return m