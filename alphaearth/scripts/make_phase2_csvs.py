#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make CSVs for Phase-2 EMA training.

Produces:
  1) uttar_pradesh_train.csv   (labeled)
  2) uttar_pradesh_val.csv     (labeled)
  3) targets_mix_train.csv     (unlabeled; Bangladesh + PakPunjab)

Each CSV has two columns: region,filename
'filename' is the image filename under <root>/<split>/images/.

Usage:
  python make_phase2_csvs.py --out_dir /path/to/csvs
"""

import os
import csv
from pathlib import Path
from typing import List, Tuple
import argparse

# ---- Edit if your roots move ----
REGION_ROOTS = {
    "uttar_pradesh": "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/uttar_pradesh",
    "bangladesh":    "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/bangladesh",
    "pak_punjab":    "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/pak_punjab",
}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

def list_images(root: Path, split: str, require_labels: bool) -> List[str]:
    """
    Return a list of image filenames in <root>/<split>/images.
    If require_labels=True, only keep those that also have
    a txt in <root>/<split>/labels with the same stem.
    """
    img_dir = root / split / "images"
    lab_dir = root / split / "labels"
    assert img_dir.is_dir(), f"Missing images dir: {img_dir}"
    if require_labels:
        assert lab_dir.is_dir(), f"Missing labels dir for labeled split: {lab_dir}"

    files = []
    for f in sorted(os.listdir(img_dir)):
        if Path(f).suffix.lower() not in IMG_EXTS:
            continue
        if require_labels:
            stem = Path(f).stem
            if not (lab_dir / f"{stem}.txt").exists():
                # Skip unlabeled image for labeled CSVs
                continue
        files.append(f)
    return files

def write_csv(path: Path, rows: List[Tuple[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "filename"])
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser("Generate Phase-2 CSVs")
    ap.add_argument("--out_dir", default="phase2_csvs", help="Where to write the CSVs")
    ap.add_argument("--src_region", default="uttar_pradesh", choices=list(REGION_ROOTS.keys()))
    ap.add_argument("--tgt_regions", nargs="*", default=["bangladesh", "pak_punjab"],
                    help="Target regions to mix for targets_mix_train.csv")
    ap.add_argument("--src_train_split", default="train")
    ap.add_argument("--src_val_split", default="val")
    ap.add_argument("--tgt_split", default="train")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # --- Source (labeled) ---
    src_root = Path(REGION_ROOTS[args.src_region])

    src_train_imgs = list_images(src_root, args.src_train_split, require_labels=True)
    src_val_imgs   = list_images(src_root, args.src_val_split,   require_labels=True)

    up_train_csv = out_dir / "uttar_pradesh_train.csv"
    up_val_csv   = out_dir / "uttar_pradesh_val.csv"

    write_csv(up_train_csv, [(args.src_region, f) for f in src_train_imgs])
    write_csv(up_val_csv,   [(args.src_region, f) for f in src_val_imgs])

    print(f"✅ {up_train_csv}  ({len(src_train_imgs)} rows)")
    print(f"✅ {up_val_csv}    ({len(src_val_imgs)} rows)")

    # --- Targets (unlabeled, mixed) ---
    tgt_rows: List[Tuple[str, str]] = []
    for r in args.tgt_regions:
        root = Path(REGION_ROOTS[r])
        imgs = list_images(root, args.tgt_split, require_labels=False)
        tgt_rows += [(r, f) for f in imgs]
        print(f"  • {r}/{args.tgt_split}: {len(imgs)} images")

    tgt_csv = out_dir / "targets_mix_train.csv"
    write_csv(tgt_csv, tgt_rows)
    print(f"✅ {tgt_csv}  ({len(tgt_rows)} rows)")

    print("\nAll CSVs ready.\n"
          f"  --train_csv_src {up_train_csv}\n"
          f"  --val_csv       {up_val_csv}\n"
          f"  --train_csv_tgt {tgt_csv}")

if __name__ == "__main__":
    main()