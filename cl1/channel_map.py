"""Maps role-specific stim parameters into the full 64-channel array."""

from typing import List, Tuple

import config


def build_stim_arrays(
    encoder_freqs: List[float],
    encoder_amps: List[float],
    pos_fb_freqs: List[float],
    pos_fb_amps: List[float],
    neg_fb_freqs: List[float],
    neg_fb_amps: List[float],
    encoder_channels: List[int] = None,
) -> Tuple[List[float], List[float]]:
    """Pack per-role stim params into full 64-channel arrays.

    encoder_channels: which physical channels to map encoder stim to.
                      Defaults to config.ENCODER_CHANNELS (all 42).
                      Pass config.SPATIAL_ENCODER_CHANNELS or
                      config.COLOR_ENCODER_CHANNELS for partial rounds.
    Dead channels receive safe minimum values.
    Returns (frequencies, amplitudes), both length TOTAL_CHANNELS.
    """
    if encoder_channels is None:
        encoder_channels = config.ENCODER_CHANNELS

    frequencies = [config.MIN_FREQ_HZ] * config.TOTAL_CHANNELS
    amplitudes  = [0.0]              * config.TOTAL_CHANNELS  # silent by default

    for idx, ch in enumerate(encoder_channels):
        frequencies[ch] = encoder_freqs[idx]
        amplitudes[ch]  = encoder_amps[idx]

    for idx, ch in enumerate(config.POS_FEEDBACK_CHANNELS):
        frequencies[ch] = pos_fb_freqs[idx]
        amplitudes[ch]  = pos_fb_amps[idx]

    for idx, ch in enumerate(config.NEG_FEEDBACK_CHANNELS):
        frequencies[ch] = neg_fb_freqs[idx]
        amplitudes[ch]  = neg_fb_amps[idx]

    return frequencies, amplitudes
