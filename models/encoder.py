"""REINFORCE policy encoder: frame + class → stim parameters.

Three stimulation modes per frame:
  'full'    — all 42 encoder channels, raw RGB frame
  'spatial' — 21 channels, edge-filtered frame (Sobel gradients)
  'color'   — 21 channels, chroma frame (UV channels of YUV)

All modes share the same CNN backbone but have separate policy heads
and receive differently preprocessed versions of the input frame.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import config


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StimEncoder(nn.Module):
    """CNN backbone + three policy heads (full / spatial / color).

    Input:  frame (B, 3, H, W) in [0, 1]  +  class_idx (B,)  +  mode str
    Output: policy_params (B, N_CH, 4)
            where dim-2 = [µ_freq, log_σ_freq, µ_amp, log_σ_amp]
            N_CH = ENCODER_CHANNEL_COUNT (full) or SPATIAL/COLOR_CHANNEL_COUNT
    """

    _SOBEL_X = torch.tensor(
        [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    ).view(1, 1, 3, 3)
    _SOBEL_Y = torch.tensor(
        [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    ).view(1, 1, 3, 3)

    def __init__(self):
        super().__init__()

        # Shared backbone: 2× stride-2 convs → AdaptiveAvgPool → 256-d vector
        self.backbone = nn.Sequential(
            _ConvBlock(3, 64, stride=2),    # 64 → 32
            _ConvBlock(64, 256, stride=2),  # 32 → 16
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),                   # (B, 256)
        )

        self.class_emb = nn.Embedding(config.NUM_CLASSES, config.CLASS_EMB_DIM)

        feat_dim = 256 + config.CLASS_EMB_DIM
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, config.ENCODER_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(config.ENCODER_HIDDEN, config.ENCODER_HIDDEN),
            nn.ReLU(inplace=True),
        )

        # Separate policy heads per mode
        self.full_head    = nn.Linear(config.ENCODER_HIDDEN, config.ENCODER_CHANNEL_COUNT * 4)
        self.spatial_head = nn.Linear(config.ENCODER_HIDDEN, config.SPATIAL_CHANNEL_COUNT * 4)
        self.color_head   = nn.Linear(config.ENCODER_HIDDEN, config.COLOR_CHANNEL_COUNT * 4)

    # ── Frame preprocessing ───────────────────────────────────────────────────

    def _spatial_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Edge magnitude via Sobel filter on grayscale. Returns (B, 3, H, W)."""
        gray = (0.299 * frame[:, 0:1] + 0.587 * frame[:, 1:2] + 0.114 * frame[:, 2:3])
        sx = self._SOBEL_X.to(frame.device)
        sy = self._SOBEL_Y.to(frame.device)
        ex = F.conv2d(gray, sx, padding=1)
        ey = F.conv2d(gray, sy, padding=1)
        edges = torch.sqrt(ex ** 2 + ey ** 2).clamp(0, 1)
        return edges.expand(-1, 3, -1, -1)

    def _color_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """YUV chroma frame — luminance + U + V channels. Returns (B, 3, H, W)."""
        r, g, b = frame[:, 0:1], frame[:, 1:2], frame[:, 2:3]
        y =  0.299  * r + 0.587  * g + 0.114  * b
        u = -0.147  * r - 0.289  * g + 0.436  * b + 0.5
        v =  0.615  * r - 0.515  * g - 0.100  * b + 0.5
        return torch.cat([y, u, v], dim=1).clamp(0, 1)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self, frame: torch.Tensor, class_idx: torch.Tensor, mode: str = "full"
    ) -> torch.Tensor:
        """Returns policy params: (B, N_CH, 4)."""
        if mode == "spatial":
            x_in = self._spatial_frame(frame)
            head  = self.spatial_head
            n_ch  = config.SPATIAL_CHANNEL_COUNT
        elif mode == "color":
            x_in = self._color_frame(frame)
            head  = self.color_head
            n_ch  = config.COLOR_CHANNEL_COUNT
        else:  # full
            x_in = frame
            head  = self.full_head
            n_ch  = config.ENCODER_CHANNEL_COUNT

        feat = self.backbone(x_in)
        emb  = self.class_emb(class_idx)
        x    = self.mlp(torch.cat([feat, emb], dim=-1))
        return head(x).view(-1, n_ch, 4)

    def sample_stim(
        self, policy_params: torch.Tensor
    ) -> Tuple[List[float], List[float], torch.Tensor]:
        """Sample stim actions from the policy distributions.

        Returns:
            freqs:    list of N_CH frequency values (Hz)
            amps:     list of N_CH amplitude values (µA)
            log_prob: scalar tensor for REINFORCE
        """
        p = policy_params.squeeze(0)  # (N_CH, 4)

        mu_freq    = p[:, 0]
        sigma_freq = F.softplus(p[:, 1]) + 1e-4
        mu_amp     = p[:, 2]
        sigma_amp  = F.softplus(p[:, 3]) + 1e-4

        freq_dist = Normal(mu_freq, sigma_freq)
        amp_dist  = Normal(mu_amp,  sigma_amp)

        freq_raw = freq_dist.sample()
        amp_raw  = amp_dist.sample()
        log_prob = freq_dist.log_prob(freq_raw).sum() + amp_dist.log_prob(amp_raw).sum()

        freqs = (
            torch.sigmoid(freq_raw) * (config.MAX_FREQ_HZ - config.MIN_FREQ_HZ)
            + config.MIN_FREQ_HZ
        ).tolist()
        amps = (
            torch.sigmoid(amp_raw) * (config.MAX_AMP_UA - config.MIN_AMP_UA)
            + config.MIN_AMP_UA
        ).tolist()

        return freqs, amps, log_prob
