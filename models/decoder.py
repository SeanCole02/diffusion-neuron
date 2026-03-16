"""Spike decoder: CL1 spike counts → predicted next video frame.

Architecture:
  spikes (177 = 3 rounds × 59ch) → reshape (3, 8, 8) [pad 59→64]
  → Conv2d spatial encoder → (64, 8, 8)
  → 3× bilinear upsample + Conv2d → (3, 64, 64) in [0, 1]

No class conditioning — the decoder must rely purely on spike patterns.
The spatial reshape preserves channel-to-position correspondence so the
decoder can learn that activity in a grid region maps to a spatial region
of the output, rather than flattening all spatial information through FC.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config

# Pad 59 active channels to 64 for clean 8×8 reshape
_SPIKE_PAD = 64 - config.ACTIVE_CHANNELS.__len__()  # 5 padding values


class SpikeDecoder(nn.Module):
    """Decode spike counts and class label into the next video frame.

    Input:
        spikes:    (B, SPIKE_DIM)  — normalized spike counts in [0, 1]
                   SPIKE_DIM = 59 * STIM_ROUNDS = 177
        class_idx: (B,)            — integer class index

    Output: (B, 3, H, W) predicted frame in [0, 1]
    """

    def __init__(self):
        super().__init__()
        assert config.FRAME_SIZE == (64, 64)
        assert config.STIM_ROUNDS == 3

        # Spatial spike encoder: (B, 3, 8, 8) → (B, 64, 8, 8)
        self.spike_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )

        # Upsample: 8→16→32→64
        self.upsample = nn.Sequential(
            # 8 → 16
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            # 16 → 32
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            # 32 → 64
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 8),
            nn.ReLU(inplace=True),
            # channel projection
            nn.Conv2d(8, 3, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, spikes: torch.Tensor, class_idx: torch.Tensor) -> torch.Tensor:
        B = spikes.shape[0]

        # Split into rounds: (B, 3, 59)
        rounds = spikes.view(B, config.STIM_ROUNDS, len(config.ACTIVE_CHANNELS))

        # Pad each round 59 → 64 for clean 8×8 reshape
        rounds = F.pad(rounds, (0, _SPIKE_PAD))  # (B, 3, 64)

        # Reshape to spatial grid: (B, 3, 8, 8)
        x = rounds.view(B, config.STIM_ROUNDS, 8, 8)

        x = self.spike_conv(x)  # (B, 64, 8, 8)
        return self.upsample(x)
