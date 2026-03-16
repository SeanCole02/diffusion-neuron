"""Ablation: train the decoder with random noise in place of real CL1 spikes.

If the decoder achieves similar SSIM to real training, it is ignoring the spikes
and learning from the class embedding alone. Run alongside or after real training
and compare SSIM curves.

No CL1 connection required.

Usage:
    python ablation_noise.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

import config
from data.ucf101_subset import get_dataloader
from models.decoder import SpikeDecoder
from utils.ssim import ssim as compute_ssim

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CKPT_PATH = Path("checkpoints/ablation_noise.pt")


def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    decoder = SpikeDecoder().to(device)
    opt = optim.Adam(decoder.parameters(), lr=config.LR_DECODER)

    CKPT_PATH.parent.mkdir(exist_ok=True)
    step = 0
    if CKPT_PATH.exists():
        ckpt = torch.load(CKPT_PATH, map_location=device)
        decoder.load_state_dict(ckpt["decoder"])
        opt.load_state_dict(ckpt["opt"])
        step = ckpt.get("step", 0)
        logger.info("Resumed from step %d", step)

    dataloader = get_dataloader(shuffle=True)
    logger.info("Dataset: %d frame pairs", len(dataloader.dataset))

    decoder.train()
    for epoch in range(100_000):
        for frame_t, frame_t1, class_idx, _ in dataloader:
            frame_t1  = frame_t1.to(device)
            class_idx = class_idx.to(device)

            # Random noise in place of CL1 spikes
            spikes = torch.rand(1, config.SPIKE_DIM, device=device)

            opt.zero_grad()
            pred = decoder(spikes, class_idx)
            loss = F.mse_loss(pred, frame_t1)
            loss.backward()
            opt.step()

            step += 1

            if step % config.LOG_INTERVAL == 0:
                ssim_val = compute_ssim(pred.detach(), frame_t1).item()
                print(
                    f"[ablation noise | step {step:>6}] epoch={epoch}  "
                    f"ssim={ssim_val:.4f}  mse={loss.item():.4f}",
                    flush=True,
                )

            if step % config.CHECKPOINT_INTERVAL == 0:
                torch.save({"decoder": decoder.state_dict(), "opt": opt.state_dict(), "step": step}, CKPT_PATH)


if __name__ == "__main__":
    train()
