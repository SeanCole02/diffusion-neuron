"""Binary UDP protocol helpers for 64-channel CL1 stimulation.

Packet layout (little-endian):

  STIM  (520 bytes):  uint64 timestamp_us | float32[64] frequencies | float32[64] amplitudes
  SPIKE (264 bytes):  uint64 timestamp_us | float32[64] spike_counts
"""

from __future__ import annotations

import struct
import time
from typing import List, Sequence, Tuple

import config

# ── Packet formats ─────────────────────────────────────────────────────────────
_N = config.TOTAL_CHANNELS

STIM_FORMAT  = f"<Q{_N}f{_N}f"   # timestamp + frequencies + amplitudes
SPIKE_FORMAT = f"<Q{_N}f"         # timestamp + spike_counts

STIM_PACKET_SIZE  = struct.calcsize(STIM_FORMAT)   # 520 bytes
SPIKE_PACKET_SIZE = struct.calcsize(SPIKE_FORMAT)  # 264 bytes


def now_us() -> int:
    return int(time.time() * 1_000_000)


def latency_ms(timestamp_us: int) -> float:
    return (now_us() - timestamp_us) / 1000.0


def validate_stim(frequencies: Sequence[float], amplitudes: Sequence[float]) -> None:
    if len(frequencies) != _N:
        raise ValueError(f"Expected {_N} frequencies, got {len(frequencies)}")
    if len(amplitudes) != _N:
        raise ValueError(f"Expected {_N} amplitudes, got {len(amplitudes)}")
    dead = set(config.DEAD_CHANNELS)
    for ch, (f, a) in enumerate(zip(frequencies, amplitudes)):
        if ch in dead:
            continue
        if a == 0.0:
            continue  # amp=0 means silent — skip validation for this channel
        if not (config.MIN_FREQ_HZ <= f <= config.MAX_FREQ_HZ):
            raise ValueError(
                f"ch={ch} freq={f:.2f} outside [{config.MIN_FREQ_HZ}, {config.MAX_FREQ_HZ}] Hz"
            )
        if not (config.MIN_AMP_UA <= a <= config.MAX_AMP_UA):
            raise ValueError(
                f"ch={ch} amp={a:.2f} outside [{config.MIN_AMP_UA}, {config.MAX_AMP_UA}] uA"
            )


def pack_stim(frequencies: Sequence[float], amplitudes: Sequence[float]) -> bytes:
    """Pack a stim command into a 520-byte UDP payload."""
    validate_stim(frequencies, amplitudes)
    return struct.pack(STIM_FORMAT, now_us(), *frequencies, *amplitudes)


def unpack_stim(packet: bytes) -> Tuple[int, List[float], List[float]]:
    """Unpack a stim packet; returns (timestamp_us, frequencies, amplitudes)."""
    if len(packet) != STIM_PACKET_SIZE:
        raise ValueError(f"Expected {STIM_PACKET_SIZE} bytes, got {len(packet)}")
    values = struct.unpack(STIM_FORMAT, packet)
    ts = int(values[0])
    freqs = list(values[1 : 1 + _N])
    amps  = list(values[1 + _N :])
    return ts, freqs, amps


def pack_spike(spike_counts: Sequence[float]) -> bytes:
    """Pack a spike response into a 264-byte UDP payload."""
    if len(spike_counts) != _N:
        raise ValueError(f"Expected {_N} spike counts, got {len(spike_counts)}")
    return struct.pack(SPIKE_FORMAT, now_us(), *spike_counts)


def unpack_spike(packet: bytes) -> Tuple[int, List[float]]:
    """Unpack a spike packet; returns (timestamp_us, spike_counts)."""
    if len(packet) != SPIKE_PACKET_SIZE:
        raise ValueError(f"Expected {SPIKE_PACKET_SIZE} bytes, got {len(packet)}")
    values = struct.unpack(SPIKE_FORMAT, packet)
    return int(values[0]), list(values[1:])
