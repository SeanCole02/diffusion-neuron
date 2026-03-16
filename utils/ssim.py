"""Differentiable SSIM metric for use as decoder training signal and REINFORCE reward."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Compute mean SSIM between two image batches.

    Args:
        pred, target: (B, C, H, W) float tensors in [0, 1]
        window_size:  Gaussian kernel size (default 11)

    Returns:
        Scalar tensor in [-1, 1]; higher is better.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    channels = pred.shape[1]
    window = _gaussian_window(window_size, channels, pred.device)
    pad = window_size // 2

    mu1 = F.conv2d(pred, window, padding=pad, groups=channels)
    mu2 = F.conv2d(target, window, padding=pad, groups=channels)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def _gaussian_window(size: int, channels: int, device: torch.device) -> torch.Tensor:
    sigma = 1.5
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    window_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)          # (size, size)
    window_2d = window_2d.unsqueeze(0).unsqueeze(0)               # (1, 1, size, size)
    return window_2d.expand(channels, 1, size, size).contiguous() # (C, 1, size, size)
