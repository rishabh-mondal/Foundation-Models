#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 2: Faster R-CNN + DINOv3 (GeoContrast-init) + FiLM(AEF) with FLEXIBLE FREEZING

Key additions:
- --freeze_backbone_for N : number of initial epochs to keep DINOv3 frozen
- --freeze_mode {none,all,last_n} : what to freeze during the freeze window
- --unfreeze_last_blocks K : when leaving the freeze window, unfreeze only the last K ViT blocks (if available)
- --freeze_film_during_freeze : also freeze FiLM during the freeze window
- separate LRs: --backbone_lr, --film_lr, --head_lr
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
    return torch.hub.load(
        repo_or_dir=DINOV3_LOCATION,
        model=DINO_MODEL_NAME,
        source="local",
        weights=DINO_WEIGHTS,
        skip_validation=True,
    )

def load_geocontrast_weights(dino_model: nn.Module, ckpt_path: str):
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
    if cand.exists(): return cand
    cand2 = base / "labels" / f"{stem}.txt"
    if cand2.exists(): return cand2
    return None

def try_load_aef_vec_from_csv(row: dict) -> Optional[np.ndarray]:
    if "aef_npy" in row and row["aef_npy"]:
        p = Path(row["aef_npy"])
        if p.exists():
            return np.load(p).astype(np.float32).reshape(-1)
    if "aef_tif" in row and row["aef_tif"]:
        p = Path(row["aef_tif"])
        if p.exists():
            return tif_to_vec64(p)
    return None

def auto_resolve_aef(region: str, filename: str, split: str) -> Optional[np.ndarray]:
    base = Path(REGION_ROOTS[region])
    stem = Path(filename).stem
    npy = base / split / "aef_vecs" / f"{stem}.npy"
    if npy.exists():
        return np.load(npy).astype(np.float32).reshape(-1)
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
    v = arr.reshape(C, -1).mean(-1).astype(np.float32)
    return v

def read_yolo_obb_to_xyxy(txt_path: Path, W: int, H: int) -> Tuple[List[List[float]], List[int]]:
    boxes, labels = [], []
    with open(txt_path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 9:
                continue
            cls_id = int(float(p[0])) + 1  # shift 1..K
            obb = np.array([float(x) for x in p[1:]], dtype=np.float32)
            xs = obb[0::2] * W; ys = obb[1::2] * H
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
        x   = self.tf(img)
        _, Ht, Wt = x.shape
        if label_path is not None:
            boxes, labels = read_yolo_obb_to_xyxy(label_path, Wt, Ht)
        else:
            boxes, labels = [], []
        target = {
            "boxes":  torch.as_tensor(boxes,  dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        vec = try_load_aef_vec_from_csv(r)
        if vec is None:
            vec = auto_resolve_aef(region, filename, self.split)
        v = torch.from_numpy(vec)
        v = v / (v.norm(p=2) + 1e-6)
        return x, target, v

def collate_fn(batch):
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
        gb = self.mlp(cond)               # [B,2C]
        gamma, beta = gb.chunk(2, dim=1)  # [B,C],[B,C]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta .unsqueeze(-1).unsqueeze(-1)
        return gamma * feat + beta

class DinoV3BackboneWrapper(nn.Module):
    def __init__(self, dino_model: nn.Module, patch_stride: int = 16, cond_dim: int = 64):
        super().__init__()
        self.dino = dino_model
        self.patch_stride = patch_stride
        C = getattr(dino_model, "embed_dim", None) or getattr(dino_model, "num_features", None)
        if C is None: C = 1024
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
                t = out["x_norm_patchtokens"]; Ht = out.get("H"); Wt = out.get("W")
                if Ht is None or Wt is None: Ht, Wt = self._maybe_hw(x)
                return t, Ht, Wt
            if "tokens" in out and out["tokens"] is not None:
                t = out["tokens"]; Ht, Wt = self._maybe_hw(x)
                if t.shape[1] == Ht*Wt + 1: t = t[:,1:,:]
                return t, Ht, Wt
        if hasattr(self.dino, "get_intermediate_layers"):
            t = self.dino.get_intermediate_layers(x, 1, False)[0]
            Ht, Wt = self._maybe_hw(x)
            return t, Ht, Wt
        t = self.dino(x)
        Ht, Wt = self._maybe_hw(x)
        if t.dim() == 3 and t.shape[1] == Ht*Wt + 1: t = t[:,1:,:]
        return t, Ht, Wt

    def forward(self, x: torch.Tensor):
        tokens, Ht, Wt = self._get_patch_tokens(x)   # [B,N,C]
        B, N, C = tokens.shape
        feat = tokens.transpose(1, 2).contiguous().view(B, C, Ht, Wt)
        cond = torch.zeros(B, 64, device=feat.device, dtype=feat.dtype) if self._cond is None \
               else self._cond.to(feat.device, dtype=feat.dtype)
        if cond.shape != (B, 64):
            raise RuntimeError(f"FiLM cond shape mismatch: {cond.shape} vs (B={B},64)")
        feat = self.film(feat, cond)
        return {"0": feat}

def build_detector(dino_model: nn.Module, num_classes: int, image_size: int = 800) -> FasterRCNN:
    backbone = DinoV3BackboneWrapper(dino_model, patch_stride=16, cond_dim=64)
    anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 256),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    model = FasterRCNN(backbone=backbone, num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       min_size=image_size, max_size=image_size)
    return model

# -------------------
# FREEZE CONTROL
# -------------------
def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters(): p.requires_grad = flag

def freeze_all_backbone(model: nn.Module, freeze_film: bool):
    # Freeze DINO
    set_requires_grad(model.backbone.dino, False)
    # Optionally freeze FiLM too
    if freeze_film and hasattr(model.backbone, "film"):
        set_requires_grad(model.backbone.film, False)

def unfreeze_all_backbone(model: nn.Module, unfreeze_film: bool = True):
    set_requires_grad(model.backbone.dino, True)
    if unfreeze_film and hasattr(model.backbone, "film"):
        set_requires_grad(model.backbone.film, True)

def unfreeze_last_n_blocks(model: nn.Module, n: int, keep_film_trainable: bool = True):
    """Freeze all DINO blocks except the last n (if available)."""
    dino = model.backbone.dino
    # Default: freeze everything
    set_requires_grad(dino, False)
    # Try to find transformer blocks container
    blocks = None
    for attr in ["blocks", "layers", "encoder_blocks"]:
        if hasattr(dino, attr):
            b = getattr(dino, attr)
            if isinstance(b, (nn.ModuleList, list)) and len(b) > 0:
                blocks = b; break
    if blocks is None:
        print("[FREEZE] Could not find blocks; unfreezing entire backbone instead.")
        set_requires_grad(dino, True)
        return
    # Unfreeze last n
    n = max(1, min(n, len(blocks)))
    for m in blocks[-n:]:
        set_requires_grad(m, True)
    # also unfreeze norm/head layers if present
    for extra in ["norm", "head", "fc_norm", "proj"]:
        if hasattr(dino, extra):
            set_requires_grad(getattr(dino, extra), True)
    # Keep FiLM trainable by default
    if keep_film_trainable and hasattr(model.backbone, "film"):
        set_requires_grad(model.backbone.film, True)

def build_param_groups(model, backbone_lr, film_lr, head_lr, weight_decay):
    back_params = []
    film_params = []
    head_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone.dino"):
            back_params.append(p)
        elif n.startswith("backbone.film"):
            film_params.append(p)
        else:
            head_params.append(p)
    return [
        {"params": back_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": film_params, "lr": film_lr,     "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr,     "weight_decay": weight_decay},
    ]

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
        total += float(losses); steps += 1
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
    # (unchanged)
    base = Path(REGION_ROOTS[region_key]) / split / "images"
    if csv_path is not None and Path(csv_path).exists():
        ds = BrickKilnDetCSV(csv_path, split=split, image_size=image_size)
    else:
        rows = [{"region": region_key, "filename": f}
                for f in os.listdir(base) if Path(f).suffix.lower() in IMG_EXTS]
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
    mc50  = float(res_c.get('map', torch.tensor(0.0)))*100.0
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
    ap = argparse.ArgumentParser("Phase 2: FasterRCNN + DINOv3 (GeoContrast-init) + FiLM(AEF) + Freezing")
    ap.add_argument("--geocontrast_ckpt", default="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/alphaearth/scripts/geocontrast_dinov3_vitl16_map_224.pth", help="Path to Phase-1 saved encoder (.pth)")
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--image_size",  type=int, default=800)
    ap.add_argument("--batch_size",  type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--epochs",      type=int, default=10)
    ap.add_argument("--backbone_lr", type=float, default=1e-5)
    ap.add_argument("--film_lr",     type=float, default=1e-4)
    ap.add_argument("--head_lr",     type=float, default=1e-4)
    ap.add_argument("--weight_decay",type=float, default=0.04)

    # Regions / splits
    ap.add_argument("--train_region", default="uttar_pradesh", choices=list(REGION_ROOTS.keys()))
    ap.add_argument("--in_region",    default="uttar_pradesh", choices=list(REGION_ROOTS.keys()))
    ap.add_argument("--oor_regions",  nargs="*", default=["pak_punjab","bangladesh"])
    ap.add_argument("--train_split",  default="train")
    ap.add_argument("--val_split",    default="val")
    ap.add_argument("--test_split",   default="test")

    # Eval/control flags
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--detector_ckpt", default="",
                    help="Optional trained Phase-2 detector .pth to load before eval")

    # CSVs for per-image AEF
    ap.add_argument("--train_csv", default="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/uttar_pradesh/uttar_pradesh_train_per_image_aef.csv")
    ap.add_argument("--val_csv",   default="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/uttar_pradesh/uttar_pradesh_val_per_image_aef.csv")

    ap.add_argument("--eval_csv_map", nargs="*", default=[],
                    help="pairs like: region=/abs/path/to/csv")

    ap.add_argument("--results_dir",  default="phase2_runs")

    # ==== FREEZE CONTROL ====
    ap.add_argument("--freeze_backbone_for", type=int, default=0,
                    help="Freeze backbone for first N epochs (0 = no initial freeze)")
    ap.add_argument("--freeze_mode", choices=["none","all","last_n"], default="none",
                    help="What to freeze during the freeze window")
    ap.add_argument("--unfreeze_last_blocks", type=int, default=0,
                    help="When exiting the freeze window, unfreeze only last K blocks (if >0). Otherwise unfreeze all.")
    ap.add_argument("--freeze_film_during_freeze", action="store_true",
                    help="Also freeze FiLM during the freeze window")
    return ap.parse_args()

def rebuild_optimizer(model, args):
    param_groups = build_param_groups(model, args.backbone_lr, args.film_lr, args.head_lr, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    return optimizer

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build DINO and load Phase-1 weights
    dino = load_dino()
    load_geocontrast_weights(dino, args.geocontrast_ckpt)

    # Build detector
    model = build_detector(dino, num_classes=args.num_classes, image_size=args.image_size).to(device)

    # Load trained detector if eval_only
    if args.eval_only and args.detector_ckpt:
        state = torch.load(args.detector_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"[INFO] Loaded detector checkpoint for eval_only: {args.detector_ckpt}")

    # Datasets / loaders
    def _auto_csv(region, split, default_path):
        if default_path: return default_path
        base = Path(REGION_ROOTS[region]) / split / "images"
        rows = [{"region": region, "filename": f}
                for f in os.listdir(base) if Path(f).suffix.lower() in IMG_EXTS]
        Path(args.results_dir).mkdir(parents=True, exist_ok=True)
        tmp = Path(args.results_dir)/f"{region}_{split}_auto.csv"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["region","filename"])
            w.writeheader(); w.writerows(rows)
        return str(tmp)

    train_csv = _auto_csv(args.train_region, args.train_split, args.train_csv)
    val_csv   = _auto_csv(args.train_region, args.val_split,   args.val_csv)

    train_ds = BrickKilnDetCSV(train_csv, split=args.train_split, image_size=args.image_size)
    val_ds   = BrickKilnDetCSV(val_csv,   split=args.val_split,   image_size=args.image_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    # ======== INITIAL FREEZE STATE ========
    # Apply freeze state for epoch 0 depending on args
    if args.freeze_backbone_for > 0 and args.freeze_mode != "none":
        if args.freeze_mode == "all":
            freeze_all_backbone(model, freeze_film=args.freeze_film_during_freeze)
            print(f"[FREEZE] Epochs 1..{args.freeze_backbone_for}: backbone frozen (mode=all), "
                  f"FiLM {'frozen' if args.freeze_film_during_freeze else 'trainable'}")
        elif args.freeze_mode == "last_n":
            # During freeze window, we still freeze *all* (standard practice),
            # then after the window we unfreeze last K blocks.
            freeze_all_backbone(model, freeze_film=args.freeze_film_during_freeze)
            print(f"[FREEZE] Epochs 1..{args.freeze_backbone_for}: backbone frozen (mode=last_n; unfreeze last {args.unfreeze_last_blocks} afterwards), "
                  f"FiLM {'frozen' if args.freeze_film_during_freeze else 'trainable'}")
    else:
        print("[FREEZE] No initial freezing.")

    # Build optimizer and scheduler
    optimizer = rebuild_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    ckpt_dir = Path(args.results_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / f"best_film_geocontrast_map_all_val_map50_3efbf_t_all_{args.train_region}.pth"

    # ======== TRAIN or SKIP ========
    if not args.eval_only:
        best_map50 = -1.0
        for ep in range(args.epochs):
            # ---- Handle freeze window exit at epoch boundary ----
            if args.freeze_backbone_for > 0 and ep == args.freeze_backbone_for:
                if args.unfreeze_last_blocks and args.freeze_mode in ("all","last_n"):
                    unfreeze_last_n_blocks(model, args.unfreeze_last_blocks, keep_film_trainable=True)
                    print(f"[FREEZE->UNFREEZE] Epoch {ep+1}: unfreeze last {args.unfreeze_last_blocks} blocks (FiLM trainable).")
                else:
                    unfreeze_all_backbone(model, unfreeze_film=True)
                    print(f"[FREEZE->UNFREEZE] Epoch {ep+1}: unfreeze entire backbone (and FiLM).")
                # Rebuild optimizer to include newly trainable params
                optimizer = rebuild_optimizer(model, args)
                # Rebuild scheduler for the remaining epochs (simple, robust)
                remaining = max(1, args.epochs - ep)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)

            tl = train_one_epoch(model, optimizer, train_loader, device, ep)
            mv, mv50 = validate(model, val_loader, device, ep)
            print(f"[E{ep+1:02d}] train_loss={tl:.4f}  val_mAP={mv:.4f}  val_mAP50={mv50:.4f}")
            if mv50 > best_map50:
                best_map50 = mv50
                torch.save(model.state_dict(), best_ckpt)
                print(f"[CKPT] saved -> {best_ckpt} (val mAP50={best_map50:.4f})")
            scheduler.step()
    else:
        if args.detector_ckpt:
            print(f"[INFO] eval_only=True, using detector ckpt: {args.detector_ckpt}")
        else:
            print("[WARN] eval_only=True but no --detector_ckpt given; evaluating un-finetuned head.")

    # ======== PREP FOR EVAL ========
    if not args.eval_only and best_ckpt.exists():
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


# Freeze both backbone and FiLM for first 2 epochs, then train everything


# CUDA_VISIBLE_DEVICES=1 nohup python -u geocontrast_phase2_fasterrcnn_film_freeze.py \
#   --epochs 5 \
#   --freeze_backbone_for 2 \
#   --freeze_mode all \
#   --freeze_film_during_freeze
# > Freeze_both_backbone_and_FiLM_for_first_2_epochs_then_train_everything 2>&1 &



# Train 10 epochs; freeze first 4 epochs, then unfreeze only the last 6 ViT blocks

# CUDA_VISIBLE_DEVICES=3 nohup python -u geocontrast_phase2_fasterrcnn_film_freeze.py \
#   --epochs 5 \
#   --freeze_backbone_for 3 \
#   --freeze_mode last_n --unfreeze_last_blocks 6
# > Train_5_epochs_freeze_backbone_for_first_3_train_FiLM_heads_during_freeze_then_unfreeze_entire_backbone.log 2>&1 &


# CUDA_VISIBLE_DEVICES=1 nohup python -u geocontrast_phase2_fasterrcnn_film_freeze.py \
#   --epochs 5 \
#   --freeze_backbone_for 3 \
#   --freeze_mode all \
#   --backbone_lr 1e-5 --film_lr 1e-4 --head_lr 1e-4
# > Train_5_epochs_freeze_backbone_first_3_train_FiLM_heads_during_freeze_then_unfreeze_backbone 2>&1 &



# === Train: freeze backbone for first 2 epochs; also freeze FiLM ===
# CUDA_VISIBLE_DEVICES=1 nohup python -u geocontrast_phase2_fasterrcnn_film_freeze.py \
#   --epochs 5 \
#   --freeze_backbone_for 2 \
#   --freeze_mode all \
#   --freeze_film_during_freeze \
#   > phase2_up_freezeALL_e5_f2_gpu1.log 2>&1 &
