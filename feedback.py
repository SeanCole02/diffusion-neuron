"""SSIM-conditioned feedback stimulation for positive and negative channels.

Positive feedback (SSIM > POS_SSIM_THRESHOLD):
    Synchronous, predictable bursts at POS_FEEDBACK_FREQ_HZ.
    Amplitude scales linearly with SSIM within the threshold range.

Negative feedback (SSIM < 0.4):
    Chaotic, asynchronous noise.
    Maximum chaos frequency scales with the magnitude of the error.

In the neutral range [0.4, 0.8], feedback channels receive only the minimum
safe stimulus (no meaningful signal).
"""

from __future__ import annotations

import random
from typing import List, Tuple

import config


def _pos_feedback(ssim: float) -> Tuple[List[float], List[float]]:
    """Synchronous burst — amplitude proportional to how good SSIM is."""
    scale = (ssim - config.POS_SSIM_THRESHOLD) / (1.0 - config.POS_SSIM_THRESHOLD)
    scale = max(0.0, min(1.0, scale))
    amp = config.MIN_AMP_UA + scale * (config.MAX_AMP_UA - config.MIN_AMP_UA)
    freqs = [config.POS_FEEDBACK_FREQ_HZ] * config.POS_FEEDBACK_COUNT
    amps = [amp] * config.POS_FEEDBACK_COUNT
    return freqs, amps


def _neg_feedback(ssim: float) -> Tuple[List[float], List[float]]:
    """Asynchronous chaotic noise — chaos intensity proportional to error."""
    scale = (config.NEG_SSIM_THRESHOLD - ssim) / config.NEG_SSIM_THRESHOLD
    scale = max(0.0, min(1.0, scale))
    max_chaos_freq = (
        config.NEG_FEEDBACK_FREQ_MIN_HZ
        + scale * (config.MAX_FREQ_HZ - config.NEG_FEEDBACK_FREQ_MIN_HZ)
    )
    freqs = [
        random.uniform(config.MIN_FREQ_HZ, max_chaos_freq)
        for _ in range(config.NEG_FEEDBACK_COUNT)
    ]
    amps = [
        random.uniform(config.MIN_AMP_UA, config.MAX_AMP_UA)
        for _ in range(config.NEG_FEEDBACK_COUNT)
    ]
    return freqs, amps


def _neutral(count: int) -> Tuple[List[float], List[float]]:
    """Safe minimum stimulus — no meaningful feedback signal."""
    return [config.MIN_FREQ_HZ] * count, [config.MIN_AMP_UA] * count


def compute_feedback(
    ssim: float,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Compute feedback stim arrays for positive and negative channels.

    Args:
        ssim: SSIM value from previous decoding step (scalar in [-1, 1])

    Returns:
        (pos_freqs, pos_amps, neg_freqs, neg_amps) — each a list of 8 values
    """
    if ssim >= config.POS_SSIM_THRESHOLD:
        pos_freqs, pos_amps = _pos_feedback(ssim)
    else:
        pos_freqs, pos_amps = _neutral(config.POS_FEEDBACK_COUNT)

    if ssim <= config.NEG_SSIM_THRESHOLD:
        neg_freqs, neg_amps = _neg_feedback(ssim)
    else:
        neg_freqs, neg_amps = _neutral(config.NEG_FEEDBACK_COUNT)

    return pos_freqs, pos_amps, neg_freqs, neg_amps
