#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prep per-image AEF vectors & CSVs for GeoContrast Phase-1

Given a structure like:
  <ROOT>/<REGION>/<SPLIT>/{images, labels, embeddings}
…this script:
  1) Matches each image to its per-image 64-band AEF GeoTIFF (exact stem, else nearest lat/lon).
  2) Converts the GeoTIFF to a [64] float32 vector (bandwise mean) and saves as .npy.
  3) Writes a CSV: region,filename,aef_npy  (what your Phase-1 code expects).

It saves outputs to BOTH:
  A) --out-base/<region>/<split>/aef_vecs/ + CSVs in --out-base
  B) <root>/<region>/<split>/aef_vecs/     + CSVs in <root>/<region>

Examples:
  python prep_per_image_aef_csvs.py \
    --root /home/.../final_data \
    --region uttar_pradesh bangladesh pak_punjab \
    --splits train val test \
    --out-base /home/.../aef_phase1

By default it expects:
  <root>/<region>/<split>/images
  <root>/<region>/<split>/embeddings (with .tif files)
"""

import argparse
import csv
import math
import re
from pathlib import Path
import numpy as np
import shutil

try:
    import rasterio as rio
except Exception:
    rio = None
    print("[WARN] rasterio not found; will try tifffile")
try:
    import tifffile as tiff
except Exception:
    tiff = None
    print("[WARN] tifffile not found; only rasterio will be used")

# ---------------------
# Defaults
# ---------------------
DEFAULT_REGIONS = ["uttar_pradesh", "bangladesh", "pak_punjab"]
DEFAULT_SPLITS  = ["train", "val", "test"]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# Filenames like "25.2879_80.4283.*"
LATLON_RE = re.compile(r"^(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)(?:\.[^.]+)?$")

def parse_lat_lon(stem: str):
    m = LATLON_RE.match(stem)
    if not m:
        return None
    lat = float(m.group(1))
    lon = float(m.group(2))
    return lat, lon

def read_aef_tif_to_vec(tif_path: Path) -> np.ndarray:
    """
    Read 64-band AEF GeoTIFF and return [64] vector (band-wise spatial mean).
    Tries rasterio first, then tifffile. Moves channels to first axis if needed.
    """
    arr = None
    if rio is not None:
        try:
            with rio.open(tif_path) as ds:
                arr = ds.read().astype(np.float32)  # [C,H,W]
        except Exception:
            arr = None
    if arr is None and tiff is not None:
        arr = tiff.imread(str(tif_path)).astype(np.float32)  # [H,W,C] or [C,H,W]
        if arr.ndim == 3 and arr.shape[0] != 64 and arr.shape[-1] == 64:
            arr = np.moveaxis(arr, -1, 0)
    if arr is None or arr.ndim != 3:
        raise RuntimeError(f"Bad AEF TIF (cannot read 3D array): {tif_path}")
    C = arr.shape[0]
    if C != 64:
        print(f"[WARN] Expected 64 bands, got {C} for {tif_path.name}")
    vec = arr.reshape(C, -1).mean(axis=1).astype(np.float32)  # [C] -> [64]
    return vec

def build_stem_map(paths):
    """Return dict: stem -> full path (first wins)."""
    d = {}
    for p in paths:
        d.setdefault(p.stem, p)
    return d

def nearest_by_latlon(target_latlon, candidates_latlon, tol_deg=5e-4):
    """Return index of nearest candidate within tol_deg (degrees). Else None."""
    if not candidates_latlon:
        return None
    lat, lon = target_latlon
    best_i, best_d = None, 1e9
    for i, cl in enumerate(candidates_latlons := candidates_latlon):
        if cl is None:
            continue
        clat, clon = cl
        d = math.hypot(lat - clat, lon - clon)
        if d < best_d:
            best_d, best_i = d, i
    return best_i if best_d <= tol_deg else None

def process_region_split(root: Path, region: str, split: str, out_base: Path) -> Path:
    """
    Process a single region/split:
      - convert all embedding TIFFs to [64] .npy (cached) into BOTH locations
      - write CSV rows 'region,filename,aef_npy'
      - return the CSV path in --out-base
    """
    img_dir = root / region / split / "images"
    emb_dir = root / region / split / "embeddings"

    if not img_dir.is_dir():
        raise RuntimeError(f"Expected folder not found: {img_dir}")
    if not emb_dir.is_dir():
        raise RuntimeError(f"Expected folder not found: {emb_dir}")

    # Output dirs
    out_vec_dirA = out_base / region / split / "aef_vecs"           # A) under --out-base
    out_vec_dirB = root / region / split / "aef_vecs"                # B) inside dataset tree
    out_vec_dirA.mkdir(parents=True, exist_ok=True)
    out_vec_dirB.mkdir(parents=True, exist_ok=True)

    # Collect files (use .glob for broad compatibility)
    imgs = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXTS])
    embs = sorted([p for p in emb_dir.glob("*.tif")])

    if len(imgs) == 0:
        print(f"[WARN] No images found in {img_dir}")
    if len(embs) == 0:
        print(f"[WARN] No AEF GeoTIFFs found in {emb_dir}")

    emb_stem_map = build_stem_map(embs)
    img_latlons  = {p.stem: parse_lat_lon(p.stem) for p in imgs}
    emb_latlons  = [parse_lat_lon(p.stem) for p in embs]

    # CSV paths
    csv_pathA = out_base / f"{region}_{split}_per_image_aef.csv"   # A) under --out-base
    csv_pathB = root / region / f"{region}_{split}_per_image_aef.csv"  # B) inside dataset tree

    rows = []
    matched, fallback_matched = 0, 0

    for img in imgs:
        stem = img.stem

        # 1) exact stem match
        emb_path = emb_stem_map.get(stem)

        # 2) nearest lat/lon fallback (if stems differ)
        if emb_path is None:
            latlon = img_latlons.get(stem)
            if latlon is not None and all(x is not None for x in latlon):
                idx = nearest_by_latlon(latlon, emb_latlons, tol_deg=5e-4)
                if idx is not None:
                    emb_path = embs[idx]
                    fallback_matched += 1

        if emb_path is None:
            print(f"[MISS] No embedding GeoTIFF found for image {img.name}")
            continue

        # Convert TIF -> vec .npy (cached) to location A, then mirror to B
        out_npyA = out_vec_dirA / f"{stem}.npy"
        if not out_npyA.exists():
            try:
                vec = read_aef_tif_to_vec(emb_path)
                np.save(out_npyA, vec)
            except Exception as e:
                print(f"[ERR] {emb_path.name}: {e}")
                continue

        # Mirror to dataset tree (location B)
        out_npyB = out_vec_dirB / f"{stem}.npy"
        if not out_npyB.exists():
            # faster than re-reading: copy or re-save
            shutil.copy2(out_npyA, out_npyB)

        # For CSV row, point to the A copy (you can swap to B if preferred)
        rows.append([region, img.name, str(out_npyA)])
        matched += 1

    # write CSVs to BOTH locations
    for csv_path in (csv_pathA, csv_pathB):
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["region", "filename", "aef_npy"])
            w.writerows(rows)
        print(f"[OK] Wrote CSV -> {csv_path}")

    print(f"[{region} / {split}] Images: {len(imgs)} | matched: {matched} "
          f"(exact:{matched - fallback_matched}, nearest:{fallback_matched})")

    return csv_pathA  # returning the --out-base CSV path by convention

def parse_args():
    ap = argparse.ArgumentParser("Prep per-image AEF vectors & CSVs for GeoContrast Phase-1")
    ap.add_argument("--root", required=True,
                    help="Root folder containing <region>/<split>/{images,labels,embeddings}")
    ap.add_argument("--region", nargs="+", default=DEFAULT_REGIONS,
                    help=f"Region keys (default: {' '.join(DEFAULT_REGIONS)})")
    ap.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS,
                    help=f"Splits to process (default: {' '.join(DEFAULT_SPLITS)})")
    ap.add_argument("--out-base", required=True,
                    help="Output base where .npy vectors and CSVs will be written (also mirrored into <root>/<region>)")
    return ap.parse_args()

def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    out_base = Path(args.out_base).expanduser().resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    for region in args.region:
        for split in args.splits:
            try:
                process_region_split(root, region, split, out_base)
            except Exception as e:
                print(f"[SKIP] {region}/{split}: {e}")

if __name__ == "__main__":
    main()