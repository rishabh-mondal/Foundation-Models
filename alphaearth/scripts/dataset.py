import os, csv, math, json, copy, argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import nms
from torchmetrics.detection import MeanAveragePrecision

REGION_ROOTS: Dict[str, str] = {
    "uttar_pradesh": "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/uttar_pradesh",
    "bangladesh":    "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/bangladesh",
    "pak_punjab":    "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/pak_punjab",
}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

try:
    import rasterio as rio
except Exception:
    rio = None
try:
    import tifffile as tiff
except Exception:
    tiff = None



# =========================
# I/O helpers
# =========================
def resolve_image_path(region: str, filename: str, split: str) -> Path:
    base = Path(REGION_ROOTS[region])
    for p in [base / split / "images" / filename, base / "images" / filename, base / filename]:
        if p.exists(): return p
    raise FileNotFoundError(f"Image not found: {filename} (region={region}, split={split})")

def resolve_label_path(region: str, filename: str, split: str, labels_dir_name="labels") -> Path:
    base = Path(REGION_ROOTS[region]); stem = Path(filename).stem
    for p in [base / split / labels_dir_name / f"{stem}.txt",
              base / labels_dir_name / f"{stem}.txt"]:
        if p.parent.exists(): return p
    return base / split / labels_dir_name / f"{stem}.txt"


def tif_to_vec64(tif_path: Path) -> np.ndarray:
    arr = None
    if rio is not None:
        try:
            with rio.open(tif_path) as ds:
                arr = ds.read().astype(np.float32)  # [C,H,W]
        except Exception:
            arr = None
    if arr is None and tiff is not None:
        arr = tiff.imread(str(tif_path)).astype(np.float32)
        if arr.ndim == 3 and arr.shape[0] != 64 and arr.shape[-1] == 64:
            arr = np.moveaxis(arr, -1, 0)
    if arr is None or arr.ndim != 3:
        raise RuntimeError(f"Bad AEF TIF: {tif_path}")
    C = arr.shape[0]
    if C != 64:
        print(f"[WARN] expected 64 bands; got {C} for {tif_path.name}")
    return arr.reshape(C, -1).mean(-1).astype(np.float32)  # [64]




def try_load_aef_vec(row: dict, region: str, filename: str, split: str) -> np.ndarray:
    if "aef_npy" in row and row["aef_npy"]:
        p = Path(row["aef_npy"]);  # per-image vector .npy
        if p.exists(): return np.load(p).astype(np.float32).reshape(-1)
    if "aef_tif" in row and row["aef_tif"]:
        p = Path(row["aef_tif"])   # per-image 64ch .tif
        if p.exists(): return tif_to_vec64(p)
    base = Path(REGION_ROOTS[region]); stem = Path(filename).stem
    npy = base / split / "aef_vecs" / f"{stem}.npy"
    if npy.exists(): return np.load(npy).astype(np.float32).reshape(-1)
    tif = base / split / "embeddings" / f"{stem}.tif"
    if tif.exists(): return tif_to_vec64(tif)
    raise FileNotFoundError(f"AEF vec not found for {region}/{split}/{filename}")


def yolo_obb_to_xyxy_line(parts: List[str], W: int, H: int) -> Tuple[List[float], int]:
    cls_id = int(float(parts[0])) + 1  # shift to 1..K (0 is background)
    obb = np.array([float(x) for x in parts[1:]], dtype=np.float32)
    xs = obb[0::2] * W; ys = obb[1::2] * H
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())], cls_id

def xyxy_to_yolo_obb_line(box: List[float], cls_id0: int, W: int, H: int) -> str:
    xmin, ymin, xmax, ymax = box
    pts = [xmin/W, ymin/H, xmax/W, ymin/H, xmax/W, ymax/H, xmin/W, ymax/H]
    return " ".join([str(int(cls_id0))] + [f"{p:.6f}" for p in pts])

# =========================
# Dataset
# =========================
class BrickKilnDetCSV(Dataset):
    """Returns (image, target, aef_vec, is_labeled). If labeled=False, target may be empty."""
    def __init__(self, csv_path: str, split: str, image_size: int = 800, labeled: bool = True,
                 labels_dir_name: str = "labels"):
        with open(csv_path, "r") as f:
            self.rows = list(csv.DictReader(f))
        assert self.rows, f"No rows in {csv_path}"
        self.split = split
        self.labeled = labeled
        self.labels_dir_name = labels_dir_name
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
        ])

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        region, filename = r["region"], r["filename"]
        img_path  = resolve_image_path(region, filename, self.split)
        img = Image.open(img_path).convert("RGB")
        x   = self.tf(img)
        _, Ht, Wt = x.shape

        boxes, labels = [], []
        if self.labeled:
            lab_path = resolve_label_path(region, filename, self.split, labels_dir_name=self.labels_dir_name)
            if lab_path.exists():
                with open(lab_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 9:
                            b, c = yolo_obb_to_xyxy_line(parts, Wt, Ht)
                            if b[2] > b[0] and b[3] > b[1]:
                                boxes.append(b); labels.append(c)

        target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1,4),
                  "labels": torch.as_tensor(labels, dtype=torch.int64),
                  "image_id": torch.tensor([idx])}

        v = try_load_aef_vec(r, region, filename, self.split)  # [64]
        v = torch.from_numpy(v).float()
        v = v / (v.norm(p=2) + 1e-6)

        return x, target, v, torch.tensor(int(self.labeled), dtype=torch.uint8)

def collate_fn(batch):
    imgs, tgts, conds, labs = zip(*batch)
    return list(imgs), list(tgts), torch.stack(conds, 0), torch.stack(labs, 0)