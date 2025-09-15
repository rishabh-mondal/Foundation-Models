#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, random
from pathlib import Path
import numpy as np

try:
    import rasterio as rio
except Exception:
    rio = None
try:
    import tifffile as tiff
except Exception:
    tiff = None

from PIL import Image
from tqdm import tqdm

def load_rgb(path: Path) -> np.ndarray:
    # [H,W,3] in [0,255] uint8 (don’t scale; we’ll report raw range)
    img = Image.open(path).convert("RGB")
    return np.asarray(img)

def load_aef(path: Path) -> np.ndarray:
    # [C,H,W] float32, expect C==64; fall back to tifffile if rasterio missing
    arr = None
    if rio is not None:
        try:
            with rio.open(path) as ds:
                arr = ds.read().astype(np.float32)  # [C,H,W]
        except Exception:
            arr = None
    if arr is None and tiff is not None:
        arr = tiff.imread(str(path)).astype(np.float32)   # [H,W,C] or [C,H,W]
        if arr.ndim == 3 and arr.shape[-1] == 64 and arr.shape[0] != 64:
            arr = np.moveaxis(arr, -1, 0)
    if arr is None or arr.ndim != 3:
        raise RuntimeError(f"Failed to load AEF: {path}")
    return arr  # [C,H,W]

def get_filelist(root: Path, split: str, modality: str):
    if modality == "rgb":
        img_dir = root / split / "images"
        exts = ("*.jpg","*.jpeg","*.png","*.tif","*.tiff")
    else:
        img_dir = root / split / "embeddings"
        exts = ("*.tif","*.tiff",)
    assert img_dir.is_dir(), f"Missing {img_dir}"
    files = []
    for p in exts:
        files.extend(img_dir.glob(p))
    files = sorted(files)
    if not files:
        raise RuntimeError(f"No files found in {img_dir}")
    return files

def describe(vals: np.ndarray):
    # vals: 1D float array
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return dict(count=0, min=np.nan, max=np.nan, p1=np.nan, p50=np.nan, p99=np.nan)
    return dict(
        count=vals.size,
        min=float(vals.min()),
        max=float(vals.max()),
        p1=float(np.percentile(vals, 1)),
        p50=float(np.percentile(vals, 50)),
        p99=float(np.percentile(vals, 99)),
    )

def main():
    ap = argparse.ArgumentParser("Check per-channel min/max (and percentiles)")
    ap.add_argument("--root", required=True, help="Region root, e.g. .../uttar_pradesh")
    ap.add_argument("--split", default="train", help="train|val|test")
    ap.add_argument("--modality", choices=["rgb","aef"], default="rgb")
    ap.add_argument("--sample", type=int, default=0, help="random sample size (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    files = get_filelist(root, args.split, args.modality)

    if args.sample and args.sample < len(files):
        random.seed(args.seed)
        files = random.sample(files, args.sample)

    print(f"[INFO] Modality={args.modality}  Split={args.split}  Files={len(files)}")

    if args.modality == "rgb":
        # Accumulate per-channel stats without loading everything into RAM
        # We’ll collect a stratified sample of pixels to estimate percentiles robustly
        ch_min = np.full(3, np.inf)
        ch_max = np.full(3, -np.inf)
        samples = []  # list of [N,3] rows
        for fp in tqdm(files, desc="Scanning RGB"):
            arr = load_rgb(fp).astype(np.float32)  # [H,W,3], 0..255
            ch_min = np.minimum(ch_min, arr.reshape(-1,3).min(axis=0))
            ch_max = np.maximum(ch_max, arr.reshape(-1,3).max(axis=0))
            # take up to 10k random pixels per image for percentile estimate
            n = min(10000, arr.shape[0]*arr.shape[1])
            idx = np.random.choice(arr.shape[0]*arr.shape[1], size=n, replace=False)
            samples.append(arr.reshape(-1,3)[idx])
        sam = np.concatenate(samples, axis=0) if samples else np.zeros((0,3), np.float32)

        print("\nRGB per-channel stats (raw 0..255):")
        for c, name in enumerate(["R","G","B"]):
            d = describe(sam[:,c])
            d["min_raw"] = float(ch_min[c]); d["max_raw"] = float(ch_max[c])
            print(f"  {name}: min={d['min_raw']:.3f} max={d['max_raw']:.3f} "
                  f"| p1={d['p1']:.3f} p50={d['p50']:.3f} p99={d['p99']:.3f} (count={int(d['count'])})")

        print("\nTIP:")
        print("  • If you use ImageNet norm, expect roughly p1~0, p99~255 before scaling.")
        print("  • If values already 0..1, divide by 255 first or pass --rgb_norm none in your detector.\n")

    else:
        # AEF: true per-channel stats (C==64 ideally)
        ch = None
        ch_samples = []  # for percentiles
        ch_min = None
        ch_max = None

        for fp in tqdm(files, desc="Scanning AEF"):
            arr = load_aef(fp)  # [C,H,W], float32
            if ch is None:
                ch = arr.shape[0]
                ch_min = np.full(ch, np.inf, dtype=np.float64)
                ch_max = np.full(ch, -np.inf, dtype=np.float64)

            # flatten per channel
            flat = arr.reshape(arr.shape[0], -1).astype(np.float32)
            # handle NaNs
            flat = np.where(np.isfinite(flat), flat, 0.0)

            ch_min = np.minimum(ch_min, flat.min(axis=1))
            ch_max = np.maximum(ch_max, flat.max(axis=1))

            # sample for percentiles
            n = min(20000, flat.shape[1])
            idx = np.random.choice(flat.shape[1], size=n, replace=False)
            ch_samples.append(flat[:, idx])  # [C,n]

        if ch_samples:
            sam = np.concatenate(ch_samples, axis=1)  # [C, total_n]
        else:
            sam = np.zeros((64,0), np.float32)

        print(f"\nAEF per-channel stats (C={sam.shape[0]}):")
        for c in range(sam.shape[0]):
            d = describe(sam[c])
            print(f"  C{c:02d}: min={ch_min[c]:.6f} max={ch_max[c]:.6f} "
                  f"| p1={d['p1']:.6f} p50={d['p50']:.6f} p99={d['p99']:.6f}")

        print("\nTIP:")
        print("  • Your training code assumes AEF ~ [-1,1]. If you see wider ranges, consider clamping or scaling.")
        print("  • Large outliers can hurt training; p1/p99 are a good sanity check.\n")

if __name__ == "__main__":
    main()


# python check_minmax.py --root /home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/uttar_pradesh --split train --modality rgb    