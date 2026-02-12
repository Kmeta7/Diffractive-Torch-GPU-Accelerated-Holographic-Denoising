#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg") 

import argparse
import math
import torch
import matplotlib.pyplot as plt


def tukey_1d_torch(N: int, alpha: float = 0.5, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Simple Tukey window (alpha in [0,1]) implemented in torch.
    """
    n = torch.arange(N, device=device, dtype=dtype)
    w = torch.ones(N, device=device, dtype=dtype)

    if alpha <= 0:
        return w
    if alpha >= 1:
        # Hann window
        return 0.5 - 0.5 * torch.cos(2 * math.pi * n / (N - 1))

    edge = int(math.floor(alpha * (N - 1) / 2))
    if edge > 0:
        k = n[:edge] / edge
        w[:edge] = 0.5 * (1 + torch.cos(math.pi * (k - 1)))
        k2 = (n[-edge:] - (N - 1 - edge)) / edge
        w[-edge:] = 0.5 * (1 + torch.cos(math.pi * k2))

    return w


def tukey_2d(H: int, W: int, alpha: float) -> torch.Tensor:
    wy = tukey_1d_torch(H, alpha=alpha)
    wx = tukey_1d_torch(W, alpha=alpha)
    return wy[:, None] * wx[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=128, help="window height")
    ap.add_argument("--w", type=int, default=128, help="window width")
    ap.add_argument("--alpha", type=float, default=0.5, help="Tukey alpha in [0,1]")
    ap.add_argument("--cmap", type=str, default="viridis", help="matplotlib colormap name")
    ap.add_argument("--no_slices", action="store_true", help="only show 2D image (no 1D slices)")
    args = ap.parse_args()

    alpha = max(0.0, min(1.0, args.alpha))
    m = tukey_2d(args.h, args.w, alpha).cpu()

    if args.no_slices:
        plt.figure()
        plt.title(f"2D Tukey window (H={args.h}, W={args.w}, alpha={alpha})")
        plt.imshow(m.numpy(), cmap=args.cmap)
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        return

    # 2D + slices
    fig = plt.figure(figsize=(10, 4))

    ax0 = fig.add_subplot(1, 2, 1)
    ax0.set_title(f"2D Tukey (alpha={alpha})")
    im = ax0.imshow(m.numpy(), cmap=args.cmap)
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")

    ax1 = fig.add_subplot(1, 2, 2)
    ax1.set_title("Center slices")
    cy = args.h // 2
    cx = args.w // 2
    ax1.plot(m[cy, :].numpy(), label="row @ center y")
    ax1.plot(m[:, cx].numpy(), label="col @ center x")
    ax1.set_xlabel("index")
    ax1.set_ylabel("amplitude")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    plt.tight_layout()
    out = "tukey_2d.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")



if __name__ == "__main__":
    main()
