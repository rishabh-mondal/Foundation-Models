#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 2: Faster R-CNN + DINOv3 backbone (GeoContrast-init) + FiLM conditioning with per-image AEF

What this does
--------------
- Loads DINOv3 weights produced by Phase 1 (save_dino_encoder output).
- Builds a Faster R-CNN with a wrapper backbone that:
    * extracts patch tokens from DINOv3
    * reshapes to feature map [B,C,H,W]
    * FiLM-modulates features using a 64-D per-image AEF vector (mean-mode)
      (If you provide a .tif map, it reduces to [64] by global mean.)
- Trains/validates on one region; evaluates IN-REGION + OOR regions.

Input data layout
-----------------
<REGION_ROOTS[region]>/<split>/{images,labels,embeddings}
- images/: RGB tiles
- labels/: YOLO-OBB txt (class x1 y1 x2 y2 x3 y3 x4 y4) in [0,1]
- embeddings/: per-image AEF GeoTIFFs (optional if you have .npy vectors)

AEF sources for Phase 2 (priority order per image)
1) CSV row (region,filename,aef_npy or aef_tif)
2) Auto-resolve: <region>/<split>/aef_vecs/<stem>.npy
3) Auto-resolve: <region>/<split>/embeddings/<stem>.tif  (then mean over H,W)

Run (example)
-------------
CUDA_VISIBLE_DEVICES=0 nohup python -u geocontrast_phase2_fasterrcnn_film.py \
  --geocontrast_ckpt /path/to/dinov3_geocontrast_mean_all.pth \
  --train_region pak_punjab \
  --in_region   pak_punjab \
  --oor_regions uttar_pradesh bangladesh \
  --train_split train --val_split val --test_split test \
  --image_size 800 --batch_size 8 --epochs 10 \
  --results_dir /path/to/phase2_runs \
  --train_csv   /.../pak_punjab_train_per_image_aef.csv \
  --val_csv     /.../pak_punjab_val_per_image_aef.csv \
  > phase2_frcnn.log 2>&1 &
"""

import os, csv, math, logging, argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

# -------------------
# Paths (EDIT if needed)
# -------------------
DINOV3_GITHUB_LOCATION = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/dinov3"
DINOV3_LOCATION = os.getenv("DINOV3_LOCATION") or DINOV3_GITHUB_LOCATION
DINO_MODEL_NAME = "dinov3_vitl16"
DINO_WEIGHTS    = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/dinov3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"

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

# -------------------
# Utils
# -------------------
def build_img_transform(size: int):
    return transforms.Compose([
        transforms.Resize((size, size), antialias=True),
        transforms.ToTensor(),
    ])

def load_dino():
    # base DINO (we will LOAD Phase-1 weights on top of this)
    return torch.hub.load(
        repo_or_dir=DINOV3_LOCATION,
        model=DINO_MODEL_NAME,
        source="local",
        weights=DINO_WEIGHTS,
        skip_validation=True,
    )

def load_geocontrast_weights(dino_model: nn.Module, ckpt_path: str):
    """
    Load Phase-1 saved encoder weights: keys are prefixed with 'dino.'.
    """
    sd = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = dino_model.load_state_dict(
        {k.replace("dino.", ""): v for k, v in sd.items() if k.startswith("dino.")},
        strict=False
    )
    print(f"[GeoContrast] loaded -> {ckpt_path}")
    if missing:
        print(f"[WARN] missing keys: {len(missing)} (ok if non-critical)")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)}")

def resolve_image_path(region: str, filename: str, split: str) -> Path:
    base = Path(REGION_ROOTS[region])
    cand = base / split / "images" / filename
    if cand.exists(): return cand
    if (base / "images" / filename).exists(): return base / "images" / filename
    if (base / filename).exists(): return base / filename
    raise FileNotFoundError(f"Image not found: {filename} (region={region}, split={split})")

def resolve_label_path(region: str, filename: str, split: str) -> Optional[Path]:
    base = Path(REGION_ROOTS[region])
    stem = Path(filename).stem
    cand = base / split / "labels" / f"{stem}.txt"
    if cand.exists(): 
        return cand
    cand2 = base / "labels" / f"{stem}.txt"
    if cand2.exists(): 
        return cand2
    # IMPORTANT: return None instead of raising
    return None

def try_load_aef_vec_from_csv(row: dict) -> Optional[np.ndarray]:
    # prefer explicit vector
    if "aef_npy" in row and row["aef_npy"]:
        p = Path(row["aef_npy"])
        if p.exists():
            return np.load(p).astype(np.float32).reshape(-1)
    # or reduce a tif to 64-D vector
    if "aef_tif" in row and row["aef_tif"]:
        p = Path(row["aef_tif"])
        if p.exists():
            return tif_to_vec64(p)
    return None

def auto_resolve_aef(region: str, filename: str, split: str) -> Optional[np.ndarray]:
    base = Path(REGION_ROOTS[region])
    stem = Path(filename).stem
    # cached vec
    npy = base / split / "aef_vecs" / f"{stem}.npy"
    if npy.exists():
        return np.load(npy).astype(np.float32).reshape(-1)
    # tif fallback
    tif = base / split / "embeddings" / f"{stem}.tif"
    if tif.exists():
        return tif_to_vec64(tif)
    return None

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
    v = arr.reshape(C, -1).mean(-1).astype(np.float32)  # [64]
    return v

def read_yolo_obb_to_xyxy(txt_path: Path, W: int, H: int) -> Tuple[List[List[float]], List[int]]:
    boxes, labels = [], []
    with open(txt_path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 9:  # cls + 8 coords
                continue
            cls_id = int(float(p[0])) + 1  # shift: 1..K (bg=0)
            obb = np.array([float(x) for x in p[1:]], dtype=np.float32)
            xs = obb[0::2] * W
            ys = obb[1::2] * H
            xmin, ymin = float(xs.min()), float(ys.min())
            xmax, ymax = float(xs.max()), float(ys.max())
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(cls_id)
    return boxes, labels

# -------------------
# Datasets
# -------------------
class BrickKilnDetCSV(Dataset):
    """
    Detection dataset driven by a per-image AEF CSV:
      CSV: region,filename,(aef_npy|aef_tif)
      Reads images/labels from REGION_ROOTS[region]/<split>/{images,labels}
      Builds per-image 64D AEF vector:
        - from CSV aef_npy, or
        - from CSV aef_tif (reduced to 64 by mean), or
        - auto-resolve <split>/aef_vecs/<stem>.npy else <split>/embeddings/<stem>.tif
    """
    def __init__(self, csv_path: str, split: str, image_size: int = 800):
        self.rows = []
        with open(csv_path, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                self.rows.append(row)
        assert len(self.rows) > 0, f"No rows in {csv_path}"

        self.split = split
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
        ])

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        region, filename = r["region"], r["filename"]

        img_path  = resolve_image_path(region, filename, self.split)
        label_path= resolve_label_path(region, filename, self.split)

        img = Image.open(img_path).convert("RGB")
        x   = self.tf(img)  # [3,H,W]
        _, Ht, Wt = x.shape

        if label_path is not None:
            boxes, labels = read_yolo_obb_to_xyxy(label_path, Wt, Ht)
        else:
            # Background-only image: no boxes, no labels
            boxes, labels = [], []

        target = {
            "boxes":  torch.as_tensor(boxes,  dtype=torch.float32).reshape(-1, 4),  # [0,4] if empty
            "labels": torch.as_tensor(labels, dtype=torch.int64),                    # [0] if empty
            "image_id": torch.tensor([idx]),
        }

        # per-image 64-D AEF (keep your existing logic)
        vec = try_load_aef_vec_from_csv(r)
        if vec is None:
            vec = auto_resolve_aef(region, filename, self.split)
        v = torch.from_numpy(vec)
        v = v / (v.norm(p=2) + 1e-6)

        return x, target, v

def collate_fn(batch):
    # keep everything, including empty-target (background) images
    imgs, tgts, conds = zip(*batch)
    return list(imgs), list(tgts), torch.stack(conds, 0)

# -------------------
# FiLM backbone wrapper
# -------------------
class FiLMAdapter(nn.Module):
    def __init__(self, feat_dim: int, cond_dim: int = 64, hidden: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, 2*feat_dim)
        )
    def forward(self, feat: torch.Tensor, cond: torch.Tensor):
        # feat: [B,C,H,W], cond: [B,64]
        gb = self.mlp(cond)               # [B,2C]
        gamma, beta = gb.chunk(2, dim=1)  # [B,C],[B,C]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta .unsqueeze(-1).unsqueeze(-1)
        return gamma * feat + beta

class DinoV3BackboneWrapper(nn.Module):
    """
    Wrap DINO to return features { '0': [B,C,H,W] } expected by Faster R-CNN,
    with FiLM modulation driven by a per-batch [B,64] conditioning vector.
    """
    def __init__(self, dino_model: nn.Module, patch_stride: int = 16, cond_dim: int = 64):
        super().__init__()
        self.dino = dino_model
        self.patch_stride = patch_stride
        C = getattr(dino_model, "embed_dim", None) or getattr(dino_model, "num_features", None)
        if C is None:
            C = 1024
        self.out_channels = C
        self.film = FiLMAdapter(feat_dim=C, cond_dim=cond_dim, hidden=min(4*C, 1024))
        self._cond = None

    def set_conditioning(self, conds: torch.Tensor):
        self._cond = conds

    @torch.no_grad()
    def _maybe_hw(self, x: torch.Tensor):
        _, _, H, W = x.shape
        return math.ceil(H/self.patch_stride), math.ceil(W/self.patch_stride)

    def _get_patch_tokens(self, x: torch.Tensor):
        out = self.dino.forward_features(x)
        if isinstance(out, dict):
            if "x_norm_patchtokens" in out:
                t = out["x_norm_patchtokens"]
                Ht = out.get("H") or self._maybe_hw(x)[0]
                Wt = out.get("W") or self._maybe_hw(x)[1]
                return t, Ht, Wt
            if "tokens" in out and out["tokens"] is not None:
                t = out["tokens"]
                Ht, Wt = self._maybe_hw(x)
                if t.shape[1] == Ht*Wt + 1: t = t[:,1:,:]
                return t, Ht, Wt
        if hasattr(self.dino, "get_intermediate_layers"):
            t = self.dino.get_intermediate_layers(x, 1, False)[0]
            Ht, Wt = self._maybe_hw(x)
            return t, Ht, Wt
        # fallback
        t = self.dino(x)
        Ht, Wt = self._maybe_hw(x)
        if t.dim() == 3 and t.shape[1] == Ht*Wt + 1: t = t[:,1:,:]
        return t, Ht, Wt

    def forward(self, x: torch.Tensor):
        tokens, Ht, Wt = self._get_patch_tokens(x)   # [B,N,C]
        B, N, C = tokens.shape
        feat = tokens.transpose(1, 2).contiguous().view(B, C, Ht, Wt)  # [B,C,H,W]

        if self._cond is None:
            cond = torch.zeros(B, 64, device=feat.device, dtype=feat.dtype)
        else:
            cond = self._cond.to(feat.device, dtype=feat.dtype)
            if cond.shape != (B, 64):
                raise RuntimeError(f"FiLM cond shape mismatch: {cond.shape} vs (B= {B}, 64)")
        feat = self.film(feat, cond)
        return {"0": feat}

def build_detector(dino_model: nn.Module, num_classes: int, image_size: int = 800) -> FasterRCNN:
    backbone = DinoV3BackboneWrapper(dino_model, patch_stride=16, cond_dim=64)
    anchor_generator = AnchorGenerator(
        sizes=((16, 32, 64, 128, 256),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        min_size=image_size,
        max_size=image_size,
    )
    return model

# -------------------
# Train / Validate / Evaluate
# -------------------
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total, steps = 0.0, 0
    pbar = tqdm(data_loader, desc=f"Train ep{epoch+1}")
    for batch in pbar:
        if batch is None: continue
        images, targets, conds = batch
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
        model.backbone.set_conditioning(conds.to(device))

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += float(losses)
        steps += 1
        pbar.set_postfix(loss=f"{float(losses):.4f}")
    return total / max(1, steps)

@torch.no_grad()
def validate(model, data_loader, device, epoch):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=False)
    for batch in tqdm(data_loader, desc=f"Val ep{epoch+1}"):
        if batch is None: continue
        images, targets, conds = batch
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
        model.backbone.set_conditioning(conds.to(device))
        out = model(images)
        out = [{k: v.detach().cpu() for k,v in o.items()} for o in out]
        tg  = [{k: v.detach().cpu() for k,v in t.items()} for t in targets]
        metric.update(out, tg)
    res = metric.compute()
    return float(res.get("map", torch.tensor(0.0))), float(res.get("map_50", torch.tensor(0.0)))

@torch.no_grad()
def evaluate_region(model, region_key: str, split: str, device, batch_size=8, num_workers=8, image_size=800,
                    csv_path: Optional[str]=None, pretty=""):
    if csv_path is not None and Path(csv_path).exists():
        ds = BrickKilnDetCSV(csv_path, split=split, image_size=image_size)
    else:
        base = Path(REGION_ROOTS[region_key]) / split / "images"
        rows = []
        for fname in os.listdir(base):
            if Path(fname).suffix.lower() in IMG_EXTS:
                rows.append({"region": region_key, "filename": fname})
        tmp = Path(f"_tmp_{region_key}_{split}.csv")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["region","filename"])
            w.writeheader(); w.writerows(rows)
        ds = BrickKilnDetCSV(str(tmp), split=split, image_size=image_size)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=True, collate_fn=collate_fn)
    model.eval()
    metric_class = MeanAveragePrecision(box_format='xyxy', class_metrics=True,  iou_thresholds=[0.5])
    metric_agn   = MeanAveragePrecision(box_format='xyxy', class_metrics=False, iou_thresholds=[0.5])
    for batch in tqdm(dl, desc=f"Test [{pretty or region_key}]"):
        if batch is None: continue
        images, targets, conds = batch
        images = [i.to(device) for i in images]
        model.backbone.set_conditioning(conds.to(device))
        preds  = model(images)
        preds = [{k: v.to('cpu') for k,v in p.items()} for p in preds]
        tgts  = [{k: v.to('cpu') for k,v in t.items()} for t in targets]
        metric_class.update(preds, tgts)
        preds_agn = [{'boxes': p['boxes'], 'scores': p['scores'], 'labels': torch.ones_like(p['labels'])} for p in preds]
        tgts_agn  = [{'boxes': t['boxes'], 'labels': torch.ones_like(t['labels'])} for t in tgts]
        metric_agn.update(preds_agn, tgts_agn)

    res_c = metric_class.compute()
    res_a = metric_agn.compute()
    ca50  = float(res_a['map_50'])*100.0
    mc50 = float(res_c.get('map', torch.tensor(0.0)))*100.0
    classes = res_c.get('classes', torch.tensor([])).tolist() if 'classes' in res_c else []
    mpc     = res_c.get('map_per_class', torch.tensor([])).tolist() if 'map_per_class' in res_c else []
    per_cls = {int(c): v*100.0 for c,v in zip(classes, mpc)}

    print("\n" + "="*84)
    print(f" Region: {pretty or region_key} — {split}")
    print("="*84)
    print(f"{'CA mAP@50':<12}{'MC mAP@50':<12}")
    print("-"*84)
    print(f"{ca50:<12.2f}{mc50:<12.2f}")
    print("="*84 + "\n")
    return ca50, mc50, per_cls

# -------------------
# CLI / Main
# -------------------
def parse_args():
    ap = argparse.ArgumentParser("Phase 2: FasterRCNN + DINOv3 (GeoContrast-init) + FiLM(AEF)")
    ap.add_argument("--geocontrast_ckpt", default="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/alphaearth/scripts/geocontrast_dinov3_vitl16_map_224.pth", help="Path to Phase-1 saved encoder (.pth)")
    ap.add_argument("--num_classes", type=int, default=4, help="bg + 3 kiln classes")
    ap.add_argument("--image_size",  type=int, default=800)
    ap.add_argument("--batch_size",  type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--epochs",      type=int, default=10)
    ap.add_argument("--backbone_lr", type=float, default=1e-5)
    ap.add_argument("--head_lr",     type=float, default=1e-4)
    ap.add_argument("--weight_decay",type=float, default=0.04)
    ap.add_argument("--train_region", default="uttar_pradesh", choices=list(REGION_ROOTS.keys()))
    ap.add_argument("--in_region",    default="uttar_pradesh", choices=list(REGION_ROOTS.keys()))
    ap.add_argument("--oor_regions",  nargs="*", default=["pak_punjab","bangladesh"])
    ap.add_argument("--train_split",  default="train")
    ap.add_argument("--val_split",    default="val")
    ap.add_argument("--test_split",   default="test")
    # Eval/control flags

    ap.add_argument("--eval_only", action="store_true",
                    help="If set, skip training and only run evaluation")
    ap.add_argument("--detector_ckpt", default="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/alphaearth/scripts/phase2_runs/best_film_geocontrast_map_all_val_map50_uttar_pradesh.pth",
                    help="Optional: path to a trained Phase-2 detector .pth to load before eval")

    # CSVs that carry per-image AEF paths for train/val (better than auto)
    ap.add_argument("--train_csv", default="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/uttar_pradesh/uttar_pradesh_train_per_image_aef.csv")
    ap.add_argument("--val_csv",   default="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/uttar_pradesh/uttar_pradesh_val_per_image_aef.csv")

    # optional CSVs for evaluation regions (if you want explicit mapping)
    ap.add_argument("--eval_csv_map", nargs="*", default=[],
                    help="pairs like: region=/abs/path/to/csv")

    ap.add_argument("--results_dir",  default="phase2_runs")
    return ap.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build DINO and load Phase-1 weights
    dino = load_dino()
    load_geocontrast_weights(dino, args.geocontrast_ckpt)

    # Build detector
    model = build_detector(dino, num_classes=args.num_classes, image_size=args.image_size).to(device)

    # If evaluating only and a trained detector checkpoint is given, load it
    if args.eval_only and args.detector_ckpt:
        state = torch.load(args.detector_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"[INFO] Loaded detector checkpoint for eval_only: {args.detector_ckpt}")

    # Datasets / loaders for train/val (still needed for eval on val? we keep building anyway)
    train_ds = BrickKilnDetCSV(args.train_csv if args.train_csv else
                               str(Path(args.results_dir)/f"{args.train_region}_{args.train_split}_auto.csv"),
                               split=args.train_split, image_size=args.image_size)
    if not args.train_csv:
        base = Path(REGION_ROOTS[args.train_region]) / args.train_split / "images"
        rows = [{"region": args.train_region, "filename": f}
                for f in os.listdir(base) if Path(f).suffix.lower() in IMG_EXTS]
        Path(args.results_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.results_dir)/f"{args.train_region}_{args.train_split}_auto.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["region","filename"])
            w.writeheader(); w.writerows(rows)

    val_ds = BrickKilnDetCSV(args.val_csv if args.val_csv else
                             str(Path(args.results_dir)/f"{args.train_region}_{args.val_split}_auto.csv"),
                             split=args.val_split, image_size=args.image_size)
    if not args.val_csv:
        base = Path(REGION_ROOTS[args.train_region]) / args.val_split / "images"
        rows = [{"region": args.train_region, "filename": f}
                for f in os.listdir(base) if Path(f).suffix.lower() in IMG_EXTS]
        with open(Path(args.results_dir)/f"{args.train_region}_{args.val_split}_auto.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["region","filename"])
            w.writeheader(); w.writerows(rows)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    # Param groups (only used if we train)
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if name.startswith("backbone.dino"):
            backbone_params.append(p)
        else:
            head_params.append(p)

    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.backbone_lr},
         {"params": head_params,     "lr": args.head_lr}],
        weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = Path(args.results_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / f"best_film_geocontrast_map_all_val_map50_{args.train_region}.pth"

    # ======== TRAIN or SKIP ========
    if not args.eval_only:
        best_map50 = -1.0
        for ep in range(args.epochs):
            tl = train_one_epoch(model, optimizer, train_loader, device, ep)
            mv, mv50 = validate(model, val_loader, device, ep)
            print(f"[E{ep+1:02d}] train_loss={tl:.4f}  val_mAP={mv:.4f}  val_mAP50={mv50:.4f}")
            if mv50 > best_map50:
                best_map50 = mv50
                torch.save(model.state_dict(), best_ckpt)
                print(f"[CKPT] saved -> {best_ckpt} (val mAP50={best_map50:.4f})")
            sched.step()
    else:
        if args.detector_ckpt:
            print(f"[INFO] eval_only=True, using detector ckpt: {args.detector_ckpt}")
        else:
            print("[WARN] eval_only=True but no --detector_ckpt given; evaluating un-finetuned head.")

    # ======== PREP FOR EVAL ========
    if not args.eval_only:
        if best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, map_location="cpu"))
            print(f"[INFO] Loaded best ckpt: {best_ckpt}")
    model.to(device).eval()

    # Optional explicit CSV map for eval regions
    eval_csv_map = {}
    for pair in args.eval_csv_map:
        if "=" in pair:
            k, v = pair.split("=", 1)
            eval_csv_map[k.strip()] = v.strip()

    # IN-REGION
    evaluate_region(
        model,
        region_key=args.in_region,
        split=args.test_split,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        csv_path=eval_csv_map.get(args.in_region),
        pretty=f"{args.in_region} (IN-REGION)"
    )

    # OOR
    for r in args.oor_regions:
        evaluate_region(
            model,
            region_key=r,
            split=args.test_split,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            csv_path=eval_csv_map.get(r),
            pretty=f"{r} (OOR)"
        )

if __name__ == "__main__":
    main()



# CUDA_VISIBLE_DEVICES=2 nohup python -u geocontrast_phase2_fasterrcnn_film.py \
#   --geocontrast_ckpt /home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/checkpoints/dinov3_geocontrast_mean_all.pth \
#   --train_region uttar_pradesh \
#   --in_region uttar_pradesh \
#   --oor_regions pak_punjab bangladesh \
#   --train_csv /home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/uttar_pradesh_train_per_image_aef.csv \
#   --val_csv   /home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/uttar_pradesh_val_per_image_aef.csv \
#   --results_dir /home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/checkpoints/phase2_runs_up \
#   > phase2_frcnn_up.log 2>&1 &    


# CUDA_VISIBLE_DEVICES=3 nohup python -u geocontrast_phase2_fasterrcnn_film.py  > phase2_frcnn_bangladesh_map.log 2>&1 &    



# CUDA_VISIBLE_DEVICES=0 python -u geocontrast_phase2_fasterrcnn_film.py \
#   --geocontrast_ckpt /home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/checkpoints/dinov3_geocontrast_mean_all.pth \
#   --eval_only \
#   --detector_ckpt /home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/alphaearth/scripts/phase2_runs/best_film_geocontrast_map_all_val_map50_pak_punjab.pth \
#   --in_region pak_punjab \
#   --oor_regions bangladesh uttar_pradesh