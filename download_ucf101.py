"""Download and extract the UCF101 subset needed for training.

Downloads UCF101.rar (~6.5 GB) from the official source, then extracts
only the 5 target classes into ./data/UCF101/<ClassName>/*.avi.

Requirements
------------
    pip install rarfile requests tqdm
    # Windows: also install UnRAR from https://www.rarlab.com/rar_add.htm
    #          and ensure `UnRAR.exe` is on your PATH (or set UNRAR_TOOL below).
    # Linux:   sudo apt install unrar   (or brew install rar on macOS)

Usage
-----
    python download_ucf101.py
    python download_ucf101.py --data-dir ./data --classes Basketball Biking Diving
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import requests
import rarfile
from tqdm import tqdm

import config

# ── Constants ─────────────────────────────────────────────────────────────────
UCF101_URL = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
UCF101_RAR_NAME = "UCF101.rar"

# If UnRAR.exe is not on PATH, set the full path here (Windows only).
UNRAR_TOOL: str | None = None  # e.g. r"C:\Program Files\WinRAR\UnRAR.exe"


# ── Helpers ───────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path) -> None:
    """Stream-download url to dest, showing a tqdm progress bar."""
    if dest.exists():
        print(f"[download] {dest.name} already exists, skipping download.")
        return

    print(f"[download] {url}")
    print(f"           -> {dest}  (~6.5 GB, this will take a while)")
    print("[download] Note: SSL verification disabled (UCF server uses non-standard cert)")

    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    response = requests.get(url, stream=True, timeout=60, verify=False)
    response.raise_for_status()

    total = int(response.headers.get("Content-Length", 0))
    chunk = 1 << 20  # 1 MB chunks

    with dest.open("wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=dest.name
    ) as bar:
        for data in response.iter_content(chunk_size=chunk):
            f.write(data)
            bar.update(len(data))


def extract_classes(
    rar_path: Path,
    out_dir: Path,
    classes: list[str],
) -> None:
    """Extract only the target class folders from the RAR archive."""
    if UNRAR_TOOL:
        rarfile.UNRAR_TOOL = UNRAR_TOOL

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract] Opening {rar_path.name} …")
    with rarfile.RarFile(str(rar_path)) as rf:
        all_names = rf.namelist()

        for cls in classes:
            cls_dest = out_dir / cls
            if cls_dest.exists() and any(cls_dest.glob("*.avi")):
                print(f"[extract] {cls}/ already present, skipping.")
                continue

            # UCF101 RAR uses 'UCF-101/<ClassName>/<video>.avi'
            prefix = f"UCF-101/{cls}/"
            members = [n for n in all_names if n.startswith(prefix) and n.endswith(".avi")]

            if not members:
                print(f"[extract] WARNING: no files found for class '{cls}' in archive.")
                continue

            print(f"[extract] {cls}: extracting {len(members)} videos …")
            cls_dest.mkdir(exist_ok=True)

            for member in tqdm(members, desc=cls, unit="file"):
                # Extract to a temp location then move (avoids recreating UCF-101/ tree)
                rf.extract(member, path=out_dir / "_tmp")
                src = out_dir / "_tmp" / member
                dst = cls_dest / src.name
                shutil.move(str(src), str(dst))

        # Clean up temp extraction tree
        tmp = out_dir / "_tmp"
        if tmp.exists():
            shutil.rmtree(tmp)

    print(f"[extract] Done. Classes available in {out_dir}")


def verify(out_dir: Path, classes: list[str], min_clips: int) -> bool:
    ok = True
    for cls in classes:
        clips = list((out_dir / cls).glob("*.avi"))
        status = "OK" if len(clips) >= min_clips else "WARN (too few clips)"
        print(f"  {cls:20s} {len(clips):4d} clips  [{status}]")
        if len(clips) < min_clips:
            ok = False
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download UCF101 subset for CL1 training")
    parser.add_argument(
        "--data-dir", default="./data", help="Root directory for downloaded data"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=config.UCF101_CLASSES,
        help="UCF101 class names to extract",
    )
    parser.add_argument(
        "--keep-rar",
        action="store_true",
        help="Keep the UCF101.rar after extraction (default: delete to free ~6.5 GB)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    ucf_dir = data_dir / "UCF101"
    rar_path = data_dir / UCF101_RAR_NAME

    data_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Download ──────────────────────────────────────────────────────
    download_file(UCF101_URL, rar_path)

    # ── Step 2: Extract target classes ────────────────────────────────────────
    extract_classes(rar_path, ucf_dir, args.classes)

    # ── Step 3: Verify ────────────────────────────────────────────────────────
    print("\n[verify] Class clip counts:")
    ok = verify(ucf_dir, args.classes, min_clips=config.SAMPLES_PER_CLASS)

    if not ok:
        print(
            "\n[verify] Some classes have fewer clips than SAMPLES_PER_CLASS "
            f"({config.SAMPLES_PER_CLASS}). Reduce SAMPLES_PER_CLASS in config.py "
            "or choose classes with more available clips."
        )

    # ── Step 4: Optionally delete the RAR ────────────────────────────────────
    if not args.keep_rar and rar_path.exists():
        print(f"\n[cleanup] Deleting {rar_path} to free disk space …")
        rar_path.unlink()
        print("[cleanup] Done.")

    print(
        f"\nDataset ready at: {ucf_dir.resolve()}\n"
        f"Update UCF101_ROOT in config.py if you moved it elsewhere.\n"
        f"Run training with:  python train.py"
    )


if __name__ == "__main__":
    main()
