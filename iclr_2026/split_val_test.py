#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
LABEL_EXTS = {".txt", ".json", ".xml", ".png", ".tif", ".tiff"}  # allow mask labels too


def index_by_stem(root: Path, allow_exts: set[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in allow_exts:
            stem = p.stem
            if stem in mapping:
                raise ValueError(
                    f"Multiple files share the same stem '{stem}' in {root}:\n- {mapping[stem]}\n- {p}"
                )
            mapping[stem] = p
    return mapping


def pair_images_labels(images_dir: Path, labels_dir: Path) -> List[Tuple[str, Path, Path]]:
    img_map = index_by_stem(images_dir, IMAGE_EXTS)
    lbl_map = index_by_stem(labels_dir, LABEL_EXTS)

    common = sorted(set(img_map.keys()) & set(lbl_map.keys()))
    if not common:
        raise ValueError("No matching image/label stems found. Check directories and extensions.")

    pairs: List[Tuple[str, Path, Path]] = [(stem, img_map[stem], lbl_map[stem]) for stem in common]
    return pairs


def split_pairs(pairs: List[Tuple[str, Path, Path]], ratio: float, seed: int | None, shuffle: bool) -> Tuple[List, List]:
    if not 0.0 < ratio < 1.0:
        raise ValueError("ratio must be between 0 and 1")
    idxs = list(range(len(pairs)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(idxs)
    # 50-50 default -> first half val, second half test (or according to ratio)
    cut = int(round(len(idxs) * ratio))
    left = [pairs[i] for i in idxs[:cut]]
    right = [pairs[i] for i in idxs[cut:]]
    return left, right


def copy_pairs(pairs: List[Tuple[str, Path, Path]], images_out: Path, labels_out: Path) -> None:
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    for stem, img, lbl in pairs:
        shutil.copy2(img, images_out / img.name)
        shutil.copy2(lbl, labels_out / lbl.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split paired images/labels into val and test sets (50-50 by default).")
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory containing images")
    parser.add_argument("--labels-dir", type=Path, required=True, help="Directory containing labels")
    parser.add_argument("--out-dir", type=Path, required=False, help="Output root directory for split (will create val/ and test/)")
    parser.add_argument("--ratio", type=float, default=0.5, help="Fraction for val split (default 0.5)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed when shuffling")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before splitting (deterministic with --seed)")

    args = parser.parse_args()

    images_dir: Path = args.images_dir
    labels_dir: Path = args.labels_dir
    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"Labels directory not found: {labels_dir}")

    # Default out-dir: common parent / split_50_50
    if args.out_dir is None:
        # try to find a common parent two levels up if siblings, else use images parent
        common_parent = images_dir.parent
        if labels_dir.parent == images_dir.parent:
            common_parent = images_dir.parent
        out_dir = common_parent / "split_50_50"
    else:
        out_dir = args.out_dir
    (out_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)
    (out_dir / "test" / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "test" / "labels").mkdir(parents=True, exist_ok=True)

    pairs = pair_images_labels(images_dir, labels_dir)
    val_pairs, test_pairs = split_pairs(pairs, args.ratio, args.seed, args.shuffle)

    copy_pairs(val_pairs, out_dir / "val" / "images", out_dir / "val" / "labels")
    copy_pairs(test_pairs, out_dir / "test" / "images", out_dir / "test" / "labels")

    print(f"Total pairs: {len(pairs)}")
    print(f"Val pairs:   {len(val_pairs)} -> {out_dir / 'val'}")
    print(f"Test pairs:  {len(test_pairs)} -> {out_dir / 'test'}")


if __name__ == "__main__":
    main()

