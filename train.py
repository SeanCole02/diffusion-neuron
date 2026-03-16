"""Main training loop: frame_t + class → CL1 stim → spikes → predicted frame_t+1.

Architecture
------------
Encoder  (REINFORCE policy, non-differentiable bottleneck through CL1):
    frame_t, class → (µ, σ) per channel → sample stim → UDP → CL1 → spikes

Decoder  (standard backprop, differentiable):
    spikes, class → predicted frame_t+1

Update rules
------------
  Decoder:  MSE loss vs. actual frame_t+1  (gradient descent)
  Encoder:  REINFORCE  —  -log_prob × advantage
            advantage = SSIM(pred, actual) − exponential_moving_average_baseline

Feedback stims (packed into the same 64-channel stim message):
  SSIM > 0.8  →  synchronous positive burst on positive channels
  SSIM < 0.4  →  chaotic asynchronous noise on negative channels

Session management:
  After MAX_TRAIN_SECONDS the loop waits REST_SECONDS before resuming,
  to respect mandatory CL1 rest requirements.

Usage:
    # development (local simulator)
    python cl1_test_server.py &
    python train.py

    # real hardware
    python train.py --cl1-host <cl1-device-host> [--stim-port 12345] [--spike-port 12346]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

import config
from cl1.channel_map import build_stim_arrays
from cl1.interface import CL1Interface
from data.ucf101_subset import get_dataloader
from feedback import compute_feedback
from models.decoder import SpikeDecoder
from models.encoder import StimEncoder
from utils.session import SessionManager
from utils.ssim import ssim as compute_ssim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_SPIKE_NORM_MAX = 10.0  # clamp outlier spikes before normalising to [0, 1]


def _normalize_spikes(spike_counts: list[float]) -> torch.Tensor:
    """Extract active-channel spikes and normalise to [0, 1]."""
    active = [spike_counts[ch] for ch in config.ACTIVE_CHANNELS]
    t = torch.tensor(active, dtype=torch.float32)
    return torch.clamp(t, 0.0, _SPIKE_NORM_MAX) / _SPIKE_NORM_MAX  # (SPIKE_DIM,)


def _load_checkpoint(
    ckpt_path: Path,
    encoder: StimEncoder,
    decoder: SpikeDecoder,
    enc_opt: optim.Optimizer,
    dec_opt: optim.Optimizer,
    device: torch.device,
) -> tuple[int, float]:
    """Load checkpoint if it exists; returns (step, reinforce_baseline)."""
    if not ckpt_path.exists():
        return 0, 0.0
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    enc_opt.load_state_dict(ckpt["enc_opt"])
    dec_opt.load_state_dict(ckpt["dec_opt"])
    step = ckpt.get("step", 0)
    baseline = ckpt.get("baseline", 0.0)
    logger.info("Resumed from checkpoint at step %d", step)
    return step, baseline


async def train(cl1_host: str, stim_port: int, spike_port: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Models ────────────────────────────────────────────────────────────────
    encoder = StimEncoder().to(device)
    decoder = SpikeDecoder().to(device)
    enc_opt = optim.Adam(encoder.parameters(), lr=config.LR_ENCODER)
    dec_opt = optim.Adam(decoder.parameters(), lr=config.LR_DECODER)

    # ── Checkpoint ────────────────────────────────────────────────────────────
    ckpt_dir = Path(config.CHECKPOINT_DIR)
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "latest.pt"
    step, baseline = _load_checkpoint(ckpt_path, encoder, decoder, enc_opt, dec_opt, device)

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Loading UCF101 subset into memory…")
    dataloader = get_dataloader(shuffle=True)
    logger.info("Dataset ready: %d frame pairs", len(dataloader.dataset))

    session = SessionManager()

    # Feedback stims from the previous step (initialised to neutral)
    prev_pos_freqs = [config.MIN_FREQ_HZ] * config.POS_FEEDBACK_COUNT
    prev_pos_amps = [config.MIN_AMP_UA] * config.POS_FEEDBACK_COUNT
    prev_neg_freqs = [config.MIN_FREQ_HZ] * config.NEG_FEEDBACK_COUNT
    prev_neg_amps = [config.MIN_AMP_UA] * config.NEG_FEEDBACK_COUNT

    # ── Training loop ─────────────────────────────────────────────────────────
    async with CL1Interface(cl1_host=cl1_host, stim_port=stim_port, spike_port=spike_port) as cl1:
        encoder.train()
        decoder.train()

        for epoch in range(100_000):
            for frame_t, frame_t1, class_idx, is_new_clip in dataloader:
                # Pause between clips to let cells settle
                if is_new_clip.item():
                    await asyncio.sleep(config.INTER_CLIP_PAUSE_S)

                # Respect mandatory CL1 rest window
                await session.rest_if_needed()

                frame_t = frame_t.to(device)        # (1, 3, H, W)
                frame_t1 = frame_t1.to(device)      # (1, 3, H, W)
                class_idx = class_idx.to(device)    # (1,)

                # ── Encoder: sample stim policy (3 modes) ─────────────────────
                enc_opt.zero_grad()

                _rounds = [
                    ("full",    config.ENCODER_CHANNELS),
                    ("spatial", config.SPATIAL_ENCODER_CHANNELS),
                    ("color",   config.COLOR_ENCODER_CHANNELS),
                ]

                round_spikes: list[torch.Tensor] = []
                log_prob = torch.tensor(0.0, device=device)
                timed_out = False
                for mode, enc_channels in _rounds:
                    policy = encoder(frame_t, class_idx, mode=mode)
                    enc_freqs, enc_amps, lp = encoder.sample_stim(policy)
                    log_prob = log_prob + lp

                    all_freqs, all_amps = build_stim_arrays(
                        enc_freqs, enc_amps,
                        prev_pos_freqs, prev_pos_amps,
                        prev_neg_freqs, prev_neg_amps,
                        encoder_channels=enc_channels,
                    )

                    for _attempt in range(3):
                        try:
                            spike_counts = await cl1.stimulate(all_freqs, all_amps)
                            break
                        except TimeoutError:
                            print(f"[step {step} {mode}] timeout (attempt {_attempt+1}/3), retrying…", flush=True)
                    else:
                        print(f"[step {step} {mode}] timeout after 3 attempts, skipping step", flush=True)
                        timed_out = True
                        break

                    round_spikes.append(_normalize_spikes(spike_counts).to(device))

                if timed_out:
                    continue

                # ── Decoder: concatenated spikes → predicted next frame ────────
                spikes = torch.cat(round_spikes).unsqueeze(0)  # (1, 59*STIM_ROUNDS)
                dec_opt.zero_grad()
                pred_frame = decoder(spikes, class_idx)          # (1, 3, H, W) in [0, 1]

                # ── Losses ────────────────────────────────────────────────────
                dec_loss = F.mse_loss(pred_frame, frame_t1)
                dec_loss.backward()
                dec_opt.step()

                ssim_val = compute_ssim(pred_frame.detach(), frame_t1).item()

                # REINFORCE: advantage = reward − baseline
                baseline = (
                    config.REINFORCE_BASELINE_DECAY * baseline
                    + (1 - config.REINFORCE_BASELINE_DECAY) * ssim_val
                )
                advantage = ssim_val - baseline
                enc_loss = -log_prob * advantage
                enc_loss.backward()
                enc_opt.step()

                # ── Prepare feedback for next step ────────────────────────────
                prev_pos_freqs, prev_pos_amps, prev_neg_freqs, prev_neg_amps = (
                    compute_feedback(ssim_val)
                )

                step += 1

                # ── Logging ───────────────────────────────────────────────────
                if step % config.LOG_INTERVAL == 0:
                    print(
                        f"[step {step:>6}] epoch={epoch} | "
                        f"ssim={ssim_val:.4f}  mse={dec_loss.item():.4f}  "
                        f"enc_loss={enc_loss.item():.4f}  baseline={baseline:.4f}  "
                        f"train_time={session.segment_elapsed_s / 60:.1f}min",
                        flush=True,
                    )

                # ── Checkpoint ────────────────────────────────────────────────
                if step % config.CHECKPOINT_INTERVAL == 0:
                    torch.save(
                        {
                            "encoder": encoder.state_dict(),
                            "decoder": decoder.state_dict(),
                            "enc_opt": enc_opt.state_dict(),
                            "dec_opt": dec_opt.state_dict(),
                            "step": step,
                            "baseline": baseline,
                        },
                        ckpt_path,
                    )
                    logger.info("Checkpoint saved (step %d)", step)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CL1 video generation training")
    parser.add_argument("--cl1-host", default=config.CL1_HOST, help="CL1 interface host")
    parser.add_argument("--stim-port", type=int, default=config.STIM_PORT, help="Stim UDP port")
    parser.add_argument("--spike-port", type=int, default=config.SPIKE_PORT, help="Spike UDP port")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(train(args.cl1_host, args.stim_port, args.spike_port))
