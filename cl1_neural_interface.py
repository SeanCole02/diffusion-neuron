"""CL1 Neural Interface — runs ON the CL1 device.

Bridges our UDP training protocol to the real CL1 hardware via the `cl` SDK.

Flow per stimulation cycle:
  1. Wait for a 520-byte UDP stim packet on STIM_BIND:STIM_PORT
  2. Parse 64-channel frequencies + amplitudes
  3. Build a stim plan via create_stim_plan() and run it in one batch
  4. Artifact rejection: ignore spikes for ARTIFACT_TICKS ticks (~10 ms at 1000 Hz)
  5. Collect spikes for COLLECT_TICKS ticks (~50 ms), count per channel
  6. Send 264-byte UDP spike packet back to sender on SPIKE_REPLY_PORT

Usage (on the CL1 device):
    python cl1_neural_interface.py
    python cl1_neural_interface.py --stim-port 12345 --spike-port 12346 --tick-rate 1000
"""

from __future__ import annotations

import argparse
import socket
import struct
import time
from collections import defaultdict
from enum import Enum, auto
from typing import Optional

import cl
from cl import BurstDesign, ChannelSet, StimDesign

# ── Protocol constants (must match cl1/protocol.py) ────────────────────────────
TOTAL_CHANNELS = 64
DEAD_CHANNELS = {0, 4, 7, 56, 63}
STIM_FORMAT  = f"<Q{TOTAL_CHANNELS}f{TOTAL_CHANNELS}f"
SPIKE_FORMAT = f"<Q{TOTAL_CHANNELS}f"
STIM_PACKET_SIZE  = struct.calcsize(STIM_FORMAT)   # 520 bytes
SPIKE_PACKET_SIZE = struct.calcsize(SPIKE_FORMAT)  # 264 bytes

# ── Interface defaults ─────────────────────────────────────────────────────────
STIM_BIND        = "127.1.0.1"
STIM_PORT        = 12345
SPIKE_REPLY_PORT = 12346

# ── Timing (at 1000 Hz loop = 1 ms per tick) ──────────────────────────────────
DEFAULT_TICK_RATE   = 1000   # Hz
ARTIFACT_TICKS      = 10     # ticks to skip after stim  (~10 ms)
COLLECT_TICKS       = 50     # ticks to count spikes over (~50 ms)

# ── Biphasic pulse defaults ────────────────────────────────────────────────────
PHASE_WIDTH_US = 200   # µs per phase (symmetric biphasic)


class State(Enum):
    IDLE       = auto()   # waiting for a stim packet
    ARTIFACT   = auto()   # post-stim artifact rejection window
    COLLECTING = auto()   # counting spikes
    SENDING    = auto()   # ready to send spike response


def _parse_stim(packet: bytes):
    """Unpack a stim UDP packet → (timestamp_us, frequencies, amplitudes)."""
    values = struct.unpack(STIM_FORMAT, packet)
    ts    = int(values[0])
    freqs = list(values[1 : 1 + TOTAL_CHANNELS])
    amps  = list(values[1 + TOTAL_CHANNELS :])
    return ts, freqs, amps


def _pack_spike(spike_counts: list[float]) -> bytes:
    """Pack spike counts into a 264-byte UDP payload."""
    return struct.pack(SPIKE_FORMAT, int(time.time() * 1_000_000), *spike_counts)


def _apply_stim(neurons, freqs: list[float], amps: list[float]) -> None:
    """Send stimulation to all non-dead channels via the CL1 SDK."""
    plan = neurons.create_stim_plan()
    for ch in range(TOTAL_CHANNELS):
        if ch in DEAD_CHANNELS:
            continue
        freq = freqs[ch]
        amp  = amps[ch]
        if amp <= 0 or freq <= 0:
            continue
        plan.stim(
            ChannelSet(ch),
            StimDesign(PHASE_WIDTH_US, -amp, PHASE_WIDTH_US, amp),
            BurstDesign(1, freq),
        )
    plan.run()


def run(stim_port: int, spike_reply_port: int, tick_rate: int, reply_host: Optional[str]) -> None:
    # ── UDP socket: bind for stim, reply for spikes ────────────────────────────
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((STIM_BIND, stim_port))
    sock.setblocking(False)   # non-blocking so loop() can drive timing

    reply_desc = f"{reply_host}:{spike_reply_port}" if reply_host else f"sender_ip:{spike_reply_port}"
    print(
        f"[CL1-INTERFACE] Listening for stim on {STIM_BIND}:{stim_port}\n"
        f"[CL1-INTERFACE] Sending spikes to {reply_desc}\n"
        f"[CL1-INTERFACE] Loop rate: {tick_rate} Hz  |  "
        f"artifact: {ARTIFACT_TICKS} ticks  |  collect: {COLLECT_TICKS} ticks"
    )

    with cl.open() as neurons:
        state       = State.IDLE
        tick_count  = 0
        spike_counts: dict[int, float] = defaultdict(float)
        sender_addr: Optional[tuple]   = None
        pending_freqs: list[float]     = []
        pending_amps:  list[float]     = []

        for tick in neurons.loop(ticks_per_second=tick_rate):

            # ── IDLE: poll for incoming stim packet ───────────────────────────
            if state == State.IDLE:
                try:
                    packet, sender = sock.recvfrom(STIM_PACKET_SIZE)
                    if len(packet) == STIM_PACKET_SIZE:
                        _, pending_freqs, pending_amps = _parse_stim(packet)
                        # Use fixed reply host+port if provided; otherwise echo
                        # the full sender tuple (preserving the ephemeral source
                        # port so Cloudflare tunnel can route the reply back).
                        sender_addr = (reply_host, spike_reply_port) if reply_host else sender
                        spike_counts = defaultdict(float)
                        tick_count   = 0
                        _apply_stim(neurons, pending_freqs, pending_amps)
                        state = State.ARTIFACT
                except BlockingIOError:
                    pass   # no packet yet, stay idle

            # ── ARTIFACT: skip spikes for ARTIFACT_TICKS ticks ───────────────
            elif state == State.ARTIFACT:
                tick_count += 1
                if tick_count >= ARTIFACT_TICKS:
                    tick_count = 0
                    state = State.COLLECTING

            # ── COLLECTING: count spikes per channel ──────────────────────────
            elif state == State.COLLECTING:
                for spike in tick.analysis.spikes:
                    spike_counts[spike.channel] += 1
                tick_count += 1
                if tick_count >= COLLECT_TICKS:
                    state = State.SENDING

            # ── SENDING: pack and transmit spike packet ───────────────────────
            elif state == State.SENDING:
                counts = [spike_counts.get(ch, 0.0) for ch in range(TOTAL_CHANNELS)]
                # Zero out dead channels
                for ch in DEAD_CHANNELS:
                    counts[ch] = 0.0
                packet = _pack_spike(counts)
                sock.sendto(packet, sender_addr)
                print(
                    f"[CL1-INTERFACE] Spike packet sent to {sender_addr} | "
                    f"total spikes: {sum(counts):.0f}"
                )
                state = State.IDLE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CL1 neural interface (runs on CL1 device)")
    parser.add_argument("--stim-port",  type=int, default=STIM_PORT)
    parser.add_argument("--spike-port", type=int, default=SPIKE_REPLY_PORT)
    parser.add_argument("--tick-rate",  type=int, default=DEFAULT_TICK_RATE,
                        help="Loop rate in Hz (max 25000)")
    parser.add_argument("--reply-host", type=str, default=None,
                        help="Fixed IP to send spike replies to (overrides sender IP)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.stim_port, args.spike_port, args.tick_rate, args.reply_host)
