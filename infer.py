"""Inference: feed frames + class through CL1 and decode predicted next frames.

Usage:
    python infer.py --class IceDancing --frames 30 --cl1-host <host>
    python infer.py --class Biking --image my_frame.png --cl1-host <host>
    python infer.py --test-set --cl1-host <host>
    python infer.py --train-set --class IceDancing --max-samples 30 --cl1-host <host>

    # Ablation: random noise instead of CL1 spikes (no hardware required)
    python infer.py --class IceDancing --frames 30 --ablation
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

import config
from cl1.channel_map import build_stim_arrays
from cl1.interface import CL1Interface
from data.ucf101_subset import UCF101FramePairDataset, get_test_dataloader, get_train_dataloader
from models.decoder import SpikeDecoder
from models.encoder import StimEncoder

_SPIKE_NORM_MAX = 10.0
_ABLATION_CHECKPOINT = "checkpoints/ablation_noise.pt"


def _normalize_spikes(spike_counts: list[float]) -> torch.Tensor:
    active = [spike_counts[ch] for ch in config.ACTIVE_CHANNELS]
    t = torch.tensor(active, dtype=torch.float32)
    return torch.clamp(t, 0.0, _SPIKE_NORM_MAX) / _SPIKE_NORM_MAX


def _load_models(checkpoint: str, device: torch.device, ablation: bool):
    decoder = SpikeDecoder().to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()

    if ablation:
        print(f"Loaded ablation checkpoint: {checkpoint}  (step {ckpt.get('step', '?')})")
        return None, decoder

    encoder = StimEncoder().to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    print(f"Loaded checkpoint: {checkpoint}  (step {ckpt.get('step', '?')})")
    return encoder, decoder


def _to_pil(t: torch.Tensor) -> Image.Image:
    return TF.to_pil_image(t.squeeze(0).clamp(0, 1).cpu())


def _make_comparison(frame_t: torch.Tensor, pred: torch.Tensor, scale: int = 6) -> Image.Image:
    panels = [_to_pil(frame_t), _to_pil(pred)]
    w, h = panels[0].size
    canvas = Image.new("RGB", (w * 2, h))
    for i, p in enumerate(panels):
        canvas.paste(p, (i * w, 0))
    return canvas.resize((canvas.width * scale, canvas.height * scale), Image.NEAREST)


def _load_image(path: str, frame_size: tuple) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = TF.resize(img, list(frame_size))
    return TF.to_tensor(img).unsqueeze(0)


async def _collect_spikes(
    cl1: CL1Interface,
    encoder: StimEncoder,
    frame_t: torch.Tensor,
    class_tensor: torch.Tensor,
    device: torch.device,
    label: str = "",
) -> torch.Tensor | None:
    """Run 3-round encoding via CL1. Returns (1, SPIKE_DIM) tensor or None on timeout."""
    _rounds = [
        ("full",    config.ENCODER_CHANNELS),
        ("spatial", config.SPATIAL_ENCODER_CHANNELS),
        ("color",   config.COLOR_ENCODER_CHANNELS),
    ]
    round_spikes = []
    for mode, enc_channels in _rounds:
        with torch.no_grad():
            policy = encoder(frame_t, class_tensor, mode=mode)
            enc_freqs, enc_amps, _ = encoder.sample_stim(policy)
        all_freqs, all_amps = build_stim_arrays(
            enc_freqs, enc_amps,
            [config.MIN_FREQ_HZ] * config.POS_FEEDBACK_COUNT,
            [config.MIN_AMP_UA]  * config.POS_FEEDBACK_COUNT,
            [config.MIN_FREQ_HZ] * config.NEG_FEEDBACK_COUNT,
            [config.MIN_AMP_UA]  * config.NEG_FEEDBACK_COUNT,
            encoder_channels=enc_channels,
        )
        for attempt in range(3):
            try:
                spike_counts = await cl1.stimulate(all_freqs, all_amps)
                break
            except TimeoutError:
                print(f"  {label}[{mode}] timeout (attempt {attempt+1}/3), retrying...")
        else:
            return None
        round_spikes.append(_normalize_spikes(spike_counts).to(device))
    return torch.cat(round_spikes).unsqueeze(0)


def _save_gif(frames: list[Image.Image], out: str) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if out.endswith(".gif") and len(frames) > 1:
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=100, loop=0)
        print(f"Saved GIF: {out}  ({len(frames)} frames, left=input  right=predicted)")
    else:
        for i, img in enumerate(frames):
            p = out if len(frames) == 1 else out.replace(".png", f"_{i}.png")
            img.save(p)
        print(f"Saved {len(frames)} PNG(s)  (left=input  right=predicted)")


def _save_class_gifs(class_frames: dict[str, list[Image.Image]], out: str) -> None:
    base = out.replace(".gif", "") if out.endswith(".gif") else out
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for cls_name, imgs in class_frames.items():
        if not imgs:
            continue
        path = f"{base}_{cls_name}.gif"
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=100, loop=0)
        print(f"Saved: {path}  ({len(imgs)} frames)")


async def run(
    class_name: str,
    image_path: str | None,
    frame_idx: int,
    frames: int,
    checkpoint: str,
    cl1_host: str,
    stim_port: int,
    spike_port: int,
    out: str,
    ablation: bool = False,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder = _load_models(checkpoint, device, ablation)

    class_idx = config.UCF101_CLASSES.index(class_name)
    class_tensor_cpu = torch.tensor([class_idx])

    if image_path:
        base_frame = _load_image(image_path, config.FRAME_SIZE)
        source_frames = [base_frame] * frames
        print(f"Using image: {image_path}  ({frames} frame(s), class='{class_name}')")
    else:
        dataset = UCF101FramePairDataset(seed=0)
        class_indices = [i for i, (_, _, c) in enumerate(dataset._pairs) if c == class_idx]
        if not class_indices:
            raise RuntimeError(f"No frames found for class '{class_name}'")
        start = frame_idx % len(class_indices)
        end   = min(start + frames, len(class_indices))
        source_frames = [dataset[i][0].unsqueeze(0) for i in class_indices[start:end]]
        print(f"Running on {len(source_frames)} frame(s) of '{class_name}' (idx {start}-{end-1})")

    result_frames: list[Image.Image] = []

    async def _infer_frame(frame_t, cl1=None):
        frame_t = frame_t.to(device)
        class_tensor = class_tensor_cpu.to(device)
        if ablation:
            spikes = torch.rand(1, config.SPIKE_DIM, device=device)
        else:
            spikes = await _collect_spikes(cl1, encoder, frame_t, class_tensor, device)
            if spikes is None:
                return None
        with torch.no_grad():
            pred = decoder(spikes, class_tensor)
        return _make_comparison(frame_t, pred)

    if ablation:
        for n, frame_t in enumerate(source_frames):
            img = await _infer_frame(frame_t)
            if img:
                result_frames.append(img)
                print(f"  frame {n+1}/{len(source_frames)} done", flush=True)
    else:
        async with CL1Interface(cl1_host=cl1_host, stim_port=stim_port, spike_port=spike_port) as cl1:
            for n, frame_t in enumerate(source_frames):
                img = await _infer_frame(frame_t, cl1)
                if img:
                    result_frames.append(img)
                    print(f"  frame {n+1}/{len(source_frames)} done", flush=True)

    if not result_frames:
        print("No frames produced.")
        return
    _save_gif(result_frames, out)


async def _run_dataloader(
    encoder: StimEncoder | None,
    decoder: SpikeDecoder,
    cl1: CL1Interface | None,
    dataloader,
    device: torch.device,
    max_samples: int,
    label: str,
    class_filter: str | None = None,
    ablation: bool = False,
) -> dict[str, list[Image.Image]]:
    class_frames: dict[str, list[Image.Image]] = {c: [] for c in config.UCF101_CLASSES}
    total = min(max_samples, len(dataloader)) if max_samples else len(dataloader)
    seen = 0

    for n, (frame_t, _, class_idx, is_new_clip) in enumerate(dataloader):
        cls_name = config.UCF101_CLASSES[class_idx.item()]
        if class_filter and cls_name != class_filter:
            continue
        if max_samples and seen >= max_samples:
            break
        seen += 1
        if is_new_clip.item():
            await asyncio.sleep(config.INTER_CLIP_PAUSE_S)

        frame_t      = frame_t.to(device)
        class_tensor = class_idx.to(device)

        if ablation:
            spikes = torch.rand(1, config.SPIKE_DIM, device=device)
        else:
            spikes = await _collect_spikes(cl1, encoder, frame_t, class_tensor, device, label=f"[{label}] sample {n+1} ")
            if spikes is None:
                print(f"  [{label}] sample {n+1}: skipped")
                continue

        with torch.no_grad():
            pred_frame = decoder(spikes, class_tensor)

        class_frames[cls_name].append(_make_comparison(frame_t, pred_frame))
        print(f"  [{label}] [{cls_name}] {seen}/{total} done", flush=True)

    return class_frames


async def run_test_set(
    checkpoint: str,
    cl1_host: str,
    stim_port: int,
    spike_port: int,
    out: str,
    max_samples: int = 0,
    class_filter: str | None = None,
    ablation: bool = False,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder = _load_models(checkpoint, device, ablation)
    dataloader = get_test_dataloader()
    print(f"Test set: {len(dataloader.dataset)} frame pairs")

    if ablation:
        class_frames = await _run_dataloader(None, decoder, None, dataloader, device, max_samples, "test", class_filter, ablation=True)
    else:
        async with CL1Interface(cl1_host=cl1_host, stim_port=stim_port, spike_port=spike_port) as cl1:
            class_frames = await _run_dataloader(encoder, decoder, cl1, dataloader, device, max_samples, "test", class_filter)
    _save_class_gifs(class_frames, out)


async def run_train_set(
    checkpoint: str,
    cl1_host: str,
    stim_port: int,
    spike_port: int,
    out: str,
    max_samples: int = 100,
    class_filter: str | None = None,
    ablation: bool = False,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder = _load_models(checkpoint, device, ablation)
    dataloader = get_train_dataloader()
    print(f"Train set: {len(dataloader.dataset)} frame pairs (capped at {max_samples})")

    if ablation:
        class_frames = await _run_dataloader(None, decoder, None, dataloader, device, max_samples, "train", class_filter, ablation=True)
    else:
        async with CL1Interface(cl1_host=cl1_host, stim_port=stim_port, spike_port=spike_port) as cl1:
            class_frames = await _run_dataloader(encoder, decoder, cl1, dataloader, device, max_samples, "train", class_filter)
    _save_class_gifs(class_frames, out)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CL1 inference")
    parser.add_argument("--class",       dest="class_name", default=None,
                        choices=config.UCF101_CLASSES,
                        help="Class to use (required for single-frame mode; optional filter for dataset modes)")
    parser.add_argument("--image",       default=None,      help="Path to a custom input image")
    parser.add_argument("--frame-idx",   type=int, default=0)
    parser.add_argument("--frames",      type=int, default=30)
    parser.add_argument("--checkpoint",  default="checkpoints/latest.pt")
    parser.add_argument("--out",         default="inference_out/out.gif")
    parser.add_argument("--test-set",    action="store_true")
    parser.add_argument("--train-set",   action="store_true")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--cl1-host",    default=config.CL1_HOST)
    parser.add_argument("--stim-port",   type=int, default=config.STIM_PORT)
    parser.add_argument("--spike-port",  type=int, default=config.SPIKE_PORT)
    parser.add_argument("--ablation",    action="store_true",
                        help="Use random noise instead of CL1 spikes (no hardware required). "
                             "Defaults checkpoint to ablation_noise.pt.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    checkpoint = args.checkpoint
    if args.ablation and checkpoint == "checkpoints/latest.pt":
        checkpoint = _ABLATION_CHECKPOINT

    if args.test_set:
        asyncio.run(run_test_set(
            checkpoint=checkpoint, cl1_host=args.cl1_host, stim_port=args.stim_port,
            spike_port=args.spike_port, out=args.out, max_samples=args.max_samples,
            class_filter=args.class_name, ablation=args.ablation,
        ))
    elif args.train_set:
        asyncio.run(run_train_set(
            checkpoint=checkpoint, cl1_host=args.cl1_host, stim_port=args.stim_port,
            spike_port=args.spike_port, out=args.out, max_samples=args.max_samples,
            class_filter=args.class_name, ablation=args.ablation,
        ))
    else:
        if not args.class_name:
            raise SystemExit("error: --class is required for single-frame inference")
        asyncio.run(run(
            class_name=args.class_name, image_path=args.image, frame_idx=args.frame_idx,
            frames=args.frames, checkpoint=checkpoint, cl1_host=args.cl1_host,
            stim_port=args.stim_port, spike_port=args.spike_port, out=args.out,
            ablation=args.ablation,
        ))
