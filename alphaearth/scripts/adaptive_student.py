#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 2 (End-to-End, upgraded):
  - Pseudo labels from AEF head-only detector AND RGB DINOv3 detector (yours)
  - Optional fusion of AEF+RGB pseudo labels via NMS union
  - Student: Faster R-CNN with DINOv3 backbone + FiLM(AEF)
  - Dual loaders (source GT + target pseudo)
  - Alignment losses: L_sim-DINO, L_sim-AEF, optional CLIP-style image↔AEF
"""

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

# =========================
# Config (edit as needed)
# =========================
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

# =========================
# DINOv3 backbones
# =========================
def load_dinov3():
    return torch.hub.load(
        repo_or_dir=DINOV3_LOCATION,
        model=DINO_MODEL_NAME,
        source="local",
        weights=DINO_WEIGHTS,
        skip_validation=True,
    )

class FiLMAdapter(nn.Module):
    def __init__(self, feat_dim: int, cond_dim: int = 64, hidden: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cond_dim, hidden), nn.ReLU(True),
                                 nn.Linear(hidden, 2*feat_dim))
    def forward(self, feat, cond):
        gb = self.mlp(cond); gamma, beta = gb.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta .unsqueeze(-1).unsqueeze(-1)
        return gamma * feat + beta

class DinoV3BackboneFiLM(nn.Module):
    def __init__(self, dino_model: nn.Module, patch_stride: int = 16, cond_dim: int = 64):
        super().__init__()
        self.dino = dino_model
        self.patch_stride = patch_stride
        C = getattr(dino_model, "embed_dim", None) or getattr(dino_model, "num_features", None) or 1024
        self.out_channels = C
        self.film = FiLMAdapter(feat_dim=C, cond_dim=cond_dim, hidden=min(4*C, 1024))
        self._cond = None

    def set_conditioning(self, conds: torch.Tensor): self._cond = conds

    @torch.no_grad()
    def _maybe_hw(self, x): _, _, H, W = x.shape; return math.ceil(H/16), math.ceil(W/16)

    def _get_patch_tokens(self, x):
        out = self.dino.forward_features(x)
        if isinstance(out, dict):
            if "x_norm_patchtokens" in out:
                t = out["x_norm_patchtokens"]; Ht = out.get("H") or self._maybe_hw(x)[0]; Wt = out.get("W") or self._maybe_hw(x)[1]
                return t, Ht, Wt
            if out.get("tokens") is not None:
                t = out["tokens"]; Ht, Wt = self._maybe_hw(x)
                if t.shape[1] == Ht*Wt + 1: t = t[:,1:,:]
                return t, Ht, Wt
        if hasattr(self.dino, "get_intermediate_layers"):
            t = self.dino.get_intermediate_layers(x, 1, False)[0]; Ht, Wt = self._maybe_hw(x); return t, Ht, Wt
        t = self.dino(x); Ht, Wt = self._maybe_hw(x)
        if t.dim() == 3 and t.shape[1] == Ht*Wt + 1: t = t[:,1:,:]
        return t, Ht, Wt

    def forward(self, x):
        tokens, Ht, Wt = self._get_patch_tokens(x)
        B, N, C = tokens.shape
        feat = tokens.transpose(1, 2).contiguous().view(B, C, Ht, Wt)
        cond = torch.zeros(B, 64, device=feat.device, dtype=feat.dtype) if self._cond is None else self._cond.to(feat.device, dtype=feat.dtype)
        feat = self.film(feat, cond)
        return {"0": feat}

class DinoV3BackbonePlain(nn.Module):
    """DINOv3 backbone without FiLM (for the RGB pseudo-label teacher)."""
    def __init__(self, dino_model: nn.Module, patch_stride: int = 16):
        super().__init__()
        self.dino = dino_model
        self.patch_stride = patch_stride
        C = getattr(dino_model, "embed_dim", None) or getattr(dino_model, "num_features", None) or 1024
        self.out_channels = C

    @torch.no_grad()
    def _maybe_hw(self, x): _, _, H, W = x.shape; return math.ceil(H/16), math.ceil(W/16)

    def _get_patch_tokens(self, x):
        out = self.dino.forward_features(x)
        if isinstance(out, dict) and "x_norm_patchtokens" in out:
            t = out["x_norm_patchtokens"]; Ht = out.get("H") or self._maybe_hw(x)[0]; Wt = out.get("W") or self._maybe_hw(x)[1]
            return t, Ht, Wt
        if hasattr(self.dino, "get_intermediate_layers"):
            t = self.dino.get_intermediate_layers(x, 1, False)[0]; Ht, Wt = self._maybe_hw(x); return t, Ht, Wt
        t = self.dino(x); Ht, Wt = self._maybe_hw(x)
        if t.dim() == 3 and t.shape[1] == Ht*Wt + 1: t = t[:,1:,:]
        return t, Ht, Wt

    def forward(self, x):
        tokens, Ht, Wt = self._get_patch_tokens(x)
        B, N, C = tokens.shape
        feat = tokens.transpose(1, 2).contiguous().view(B, C, Ht, Wt)
        return {"0": feat}

def build_student_detector(dino_model: nn.Module, num_classes: int, image_size: int = 800) -> FasterRCNN:
    backbone = DinoV3BackboneFiLM(dino_model, patch_stride=16, cond_dim=64)
    anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 256),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    return FasterRCNN(backbone=backbone, num_classes=num_classes,
                      rpn_anchor_generator=anchor_generator,
                      min_size=image_size, max_size=image_size)

def build_rgb_teacher(num_classes: int, image_size: int = 800) -> FasterRCNN:
    dino_rgb = load_dinov3()
    backbone = DinoV3BackbonePlain(dino_rgb, patch_stride=16)
    anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 256),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    return FasterRCNN(backbone=backbone, num_classes=num_classes,
                      rpn_anchor_generator=anchor_generator,
                      min_size=image_size, max_size=image_size)

# =========================
# Alignment heads / losses
# =========================
class FeatProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, out_dim))
    def forward(self, z): return F.normalize(self.net(z), dim=-1)

class AEFProjector(nn.Module):
    def __init__(self, in_dim: int = 64, out_dim: int = 256, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(True), nn.Linear(hidden, out_dim))
    def forward(self, v): return F.normalize(self.net(v), dim=-1)

def global_pool(feat): return F.adaptive_avg_pool2d(feat, 1).flatten(1)
def cosine_loss(u, v): return (1.0 - (F.normalize(u,-1) * F.normalize(v,-1)).sum(-1)).mean()
def clip_loss(z_i, z_a, logit_scale):
    s = logit_scale.exp().clamp(1e-3, 100.0); logits = s * (z_i @ z_a.t())
    B = z_i.size(0); y = torch.arange(B, device=z_i.device)
    return 0.5*(F.cross_entropy(logits, y) + F.cross_entropy(logits.t(), y))

# =========================
# Pseudo-label generators
# =========================
class AEFHeadOnlyBackbone(nn.Module):
    def __init__(self, in_ch=64, out_ch=128):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.out_channels = out_ch
    def forward(self, x): return {"0": self.conv(x)}

def load_aef_head_only_detector(num_classes: int, ckpt_path: str, image_size: int) -> FasterRCNN:
    backbone = AEFHeadOnlyBackbone(64, 128)
    ag = AnchorGenerator(sizes=((12, 16, 24, 32, 48, 64, 96, 128, 192, 256),),
                         aspect_ratios=((0.5, 1.0, 2.0),))
    model = FasterRCNN(backbone=backbone, num_classes=num_classes,
                       rpn_anchor_generator=ag, min_size=image_size, max_size=image_size,
                       image_mean=[0.0]*64, image_std=[1.0]*64)
    if ckpt_path and Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"[PL-AEF] loaded: {ckpt_path}")
    return model

def _load_aef_tensor(region: str, filename: str, split: str, size: int) -> torch.Tensor:
    base = Path(REGION_ROOTS[region]); stem = Path(filename).stem
    tif = base / split / "embeddings" / f"{stem}.tif"
    arr = None
    if rio is not None and tif.exists():
        try:
            with rio.open(tif) as ds:
                arr = ds.read().astype(np.float32)
        except Exception:
            arr = None
    if arr is None and tiff is not None and tif.exists():
        arr = tiff.imread(str(tif)).astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 64 and arr.shape[0] != 64:
            arr = np.moveaxis(arr, -1, 0)
    if arr is None:
        # fallback: vector → tiny map
        v = tif_to_vec64(tif)
        arr = np.tile(v[:,None,None], (1, 32, 32))
    x = torch.from_numpy(arr)
    x = F.interpolate(x.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)
    return x.clamp_(-1.0, 1.0)

def _write_pl_files(out_dir: Path, stem: str, boxes: List[List[float]], scores: List[float],
                    labels_1based: List[int], W: int, H: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    # YOLO-OBB .txt (no score)
    with open(out_dir / f"{stem}.txt", "w") as f:
        for b, s, c1 in zip(boxes, scores, labels_1based):
            cls0 = max(0, int(c1)-1)
            f.write(xyxy_to_yolo_obb_line(b, cls0, W, H) + "\n")
    # .json with scores for later fusion
    with open(out_dir / f"{stem}.json", "w") as jf:
        json.dump([{"bbox": [float(x) for x in b], "score": float(s), "cls": int(c1)} for b,s,c1 in zip(boxes, scores, labels_1based)], jf)

@torch.no_grad()
def generate_pseudo_labels_aef(target_csv: str, split: str, save_dir_name: str,
                               aef_ckpt: str, num_classes: int, image_size: int,
                               score_thr: float = 0.5, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    det = load_aef_head_only_detector(num_classes, aef_ckpt, image_size).to(device).eval()

    rows = list(csv.DictReader(open(target_csv)))
    by_region: Dict[str, List[dict]] = {}
    for r in rows: by_region.setdefault(r["region"], []).append(r)

    for region, rlist in by_region.items():
        out_dir = Path(REGION_ROOTS[region]) / split / save_dir_name
        print(f"[PL-AEF] writing → {out_dir}")
        for r in tqdm(rlist, desc=f"PL-AEF {region}/{split}"):
            fn = r["filename"]; stem = Path(fn).stem
            x = _load_aef_tensor(region, fn, split, image_size)  # [64,H,W]
            p = det([x.to(device)])[0]
            bxs = p["boxes"].detach().cpu().tolist()
            scs = p["scores"].detach().cpu().tolist()
            lbs = p["labels"].detach().cpu().tolist()
            # filter
            keep = [(b,s,c) for b,s,c in zip(bxs, scs, lbs) if s >= score_thr]
            if keep:
                B,S,C = zip(*keep)
                _write_pl_files(out_dir, stem, list(B), list(S), list(C), image_size, image_size)

@torch.no_grad()
def generate_pseudo_labels_rgb(target_csv: str, split: str, save_dir_name: str,
                               rgb_ckpt: str, num_classes: int, image_size: int,
                               score_thr: float = 0.5, device=None):
    """
    Runs a DINOv3+FRCNN RGB teacher (your ckpt) on target images and writes YOLO-OBB + JSON.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = build_rgb_teacher(num_classes=num_classes, image_size=image_size).to(device).eval()
    if rgb_ckpt and Path(rgb_ckpt).exists():
        state = torch.load(rgb_ckpt, map_location="cpu")
        teacher.load_state_dict(state, strict=False)
        print(f"[PL-RGB] loaded: {rgb_ckpt}")

    tf = transforms.Compose([transforms.Resize((image_size, image_size), antialias=True),
                             transforms.ToTensor()])

    rows = list(csv.DictReader(open(target_csv)))
    by_region: Dict[str, List[dict]] = {}
    for r in rows: by_region.setdefault(r["region"], []).append(r)

    for region, rlist in by_region.items():
        out_dir = Path(REGION_ROOTS[region]) / split / save_dir_name
        print(f"[PL-RGB] writing → {out_dir}")
        for r in tqdm(rlist, desc=f"PL-RGB {region}/{split}"):
            fn = r["filename"]; stem = Path(fn).stem
            img = Image.open(resolve_image_path(region, fn, split)).convert("RGB")
            x = tf(img).to(device)
            p = teacher([x])[0]
            bxs = p["boxes"].detach().cpu().tolist()
            scs = p["scores"].detach().cpu().tolist()
            lbs = p["labels"].detach().cpu().tolist()
            keep = [(b,s,c) for b,s,c in zip(bxs, scs, lbs) if s >= score_thr]
            if keep:
                B,S,C = zip(*keep)
                _write_pl_files(out_dir, stem, list(B), list(S), list(C), image_size, image_size)

# =========================
# Fuse (AEF + RGB) pseudo
# =========================
def _read_pl_json(json_path: Path) -> List[dict]:
    if json_path.exists():
        try:
            return json.load(open(json_path))
        except Exception:
            return []
    # fallback: read .txt (no scores)
    txt = json_path.with_suffix(".txt")
    if not txt.exists(): return []
    lines = [l.strip().split() for l in open(txt)]
    H = W = 1.0
    out = []
    for parts in lines:
        if len(parts) == 9:
            # reconstruct xyxy from normalized corners
            xs = np.array([float(parts[i]) for i in [1,3,5,7]])
            ys = np.array([float(parts[i]) for i in [2,4,6,8]])
            xmin, ymin, xmax, ymax = float(xs.min()*W), float(ys.min()*H), float(xs.max()*W), float(ys.max()*H)
            out.append({"bbox":[xmin,ymin,xmax,ymax], "score":0.5, "cls": int(parts[0])+1})
    return out

def fuse_pseudo_label_dirs(target_csv: str, split: str,
                           aef_dir_name: str, rgb_dir_name: str, out_dir_name: str,
                           image_size: int, iou_thr: float = 0.5, prefer: Optional[str]=None):
    """
    Merge AEF+RGB pseudo labels with class-wise NMS. If prefer is 'rgb' or 'aef',
    we keep preferred boxes when overlaps tie on score.
    """
    rows = list(csv.DictReader(open(target_csv)))
    by_region: Dict[str, List[dict]] = {}
    for r in rows: by_region.setdefault(r["region"], []).append(r)

    for region, rlist in by_region.items():
        base = Path(REGION_ROOTS[region]) / split
        da = base / aef_dir_name
        dr = base / rgb_dir_name
        out = base / out_dir_name
        out.mkdir(parents=True, exist_ok=True)
        print(f"[PL-FUSE] {region}/{split} → {out}")

        for r in tqdm(rlist, desc=f"FUSE {region}/{split}"):
            stem = Path(r["filename"]).stem
            a_json = da / f"{stem}.json"
            r_json = dr / f"{stem}.json"
            A = _read_pl_json(a_json)
            R = _read_pl_json(r_json)
            all_classes = set([d["cls"] for d in A] + [d["cls"] for d in R])
            fused_boxes, fused_scores, fused_cls = [], [], []

            for c in all_classes:
                a_c = [d for d in A if d["cls"]==c]
                r_c = [d for d in R if d["cls"]==c]
                boxes = torch.tensor([d["bbox"] for d in (a_c + r_c)], dtype=torch.float32)
                if boxes.numel() == 0: continue
                scores= torch.tensor([d["score"] for d in (a_c + r_c)], dtype=torch.float32)
                # tiny tie preference
                if prefer in ("rgb","aef"):
                    bias = torch.zeros_like(scores)
                    bias[len(a_c):] = 1e-4 if prefer=="rgb" else 0.0
                    bias[:len(a_c)] = 1e-4 if prefer=="aef" else 0.0
                    scores = scores + bias
                keep = nms(boxes, scores, iou_thr)
                for i in keep.tolist():
                    fused_boxes.append(boxes[i].tolist())
                    fused_scores.append(float(scores[i]))
                    fused_cls.append(int(c))

            # write fused .txt (no score) + .json (with scores)
            _write_pl_files(out, stem, fused_boxes, fused_scores, fused_cls, image_size, image_size)

# =========================
# Train / Validate (dual)
# =========================
def build_param_groups(model, proj_feats, proj_aef, backbone_lr, film_lr, head_lr, proj_lr, weight_decay):
    back_params, film_params, head_params, proj_params = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if n.startswith("backbone.dino"): back_params.append(p)
        elif n.startswith("backbone.film"): film_params.append(p)
        else: head_params.append(p)
    proj_params += list(proj_feats.parameters()) + list(proj_aef.parameters())
    return [
        {"params": back_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": film_params, "lr": film_lr,     "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr,     "weight_decay": weight_decay},
        {"params": proj_params, "lr": proj_lr,     "weight_decay": weight_decay},
    ]

def forward_feats(model, images_tensor, conds_tensor):
    model.backbone.set_conditioning(conds_tensor)
    return model.backbone(images_tensor)["0"]

def step_detector(model, images, targets, conds):
    model.backbone.set_conditioning(conds)
    loss_dict = model(images, targets)
    return sum(loss for loss in loss_dict.values()), loss_dict

def train_epoch_dual(model, align_copy, proj_feats, proj_aef, logit_scale,
                     dl_src, dl_tgt, device, epoch, args, optimizer):
    model.train(); proj_feats.train(); proj_aef.train()
    total = 0.0; steps = 0
    it_src = iter(dl_src) if dl_src else None
    it_tgt = iter(dl_tgt) if dl_tgt else None
    num_iters = max(len(dl_src) if dl_src else 0, len(dl_tgt) if dl_tgt else 0)
    pbar = tqdm(range(num_iters), desc=f"Train ep{epoch+1}")

    w_pl   = min(1.0, (epoch+1)/max(1,args.warmup_pl))   * args.lambda_unsup
    w_simr = min(1.0, (epoch+1)/max(1,args.warmup_sim))
    w_dino = w_simr * args.lambda_sim_dino
    w_aef  = w_simr * args.lambda_sim_aef
    w_clip = w_simr * args.lambda_clip_aef

    for _ in pbar:
        optimizer.zero_grad(set_to_none=True)
        det_loss = 0.0; align_loss = 0.0

        bS = next(it_src, None) if it_src else None
        bT = next(it_tgt, None) if it_tgt else None

        if bS:
            ims, tgts, conds, _ = bS
            ims = [i.to(device) for i in ims]
            tgts = [{k:v.to(device) for k,v in t.items()} for t in tgts]
            conds = conds.to(device)
            l_s, _ = step_detector(model, ims, tgts, conds)
            det_loss += l_s

        if bT:
            imt, tgtt, condt, _ = bT
            imt = [i.to(device) for i in imt]
            tgtt = [{k:v.to(device) for k,v in t.items()} for t in tgtt]
            condt = condt.to(device)
            l_t, _ = step_detector(model, imt, tgtt, condt)
            det_loss += w_pl * l_t

        def align_once(images_list, conds_tensor):
            if not images_list: return 0.0
            x = torch.stack([im.to(device) for im in images_list], 0)
            c = conds_tensor.to(device)
            F_stu = forward_feats(model, x, c)
            z_stu = global_pool(F_stu)
            z_stu_p = proj_feats(z_stu)

            with torch.no_grad():
                out = align_copy.forward_features(x)
                if isinstance(out, dict) and "x_norm_patchtokens" in out:
                    t = out["x_norm_patchtokens"]
                elif hasattr(align_copy, "get_intermediate_layers"):
                    t = align_copy.get_intermediate_layers(x, 1, False)[0]
                else:
                    t = out if isinstance(out, torch.Tensor) else None
                if t is not None and t.dim()==3:
                    B, N, C = t.shape
                    gh = int(round(N**0.5)); gw = max(1, N//max(1,gh))
                    F_ref = t[:, :gh*gw, :].transpose(1,2).reshape(B, C, gh, gw)
                else:
                    F_ref = F_stu.detach()
            z_ref = global_pool(F_ref); z_ref_p = F.normalize(z_ref, dim=-1)
            z_aef = proj_aef(c)

            loss = 0.0
            if w_dino>0: loss += w_dino * cosine_loss(z_stu_p, z_ref_p)
            if w_aef >0: loss += w_aef  * cosine_loss(z_stu_p, z_aef)
            if w_clip>0: loss += w_clip * clip_loss(z_stu_p, z_aef, logit_scale)
            return loss

        if bS: align_loss += align_once(bS[0], bS[2])
        if bT: align_loss += align_once(bT[0], bT[2])

        loss = det_loss + align_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(proj_feats.parameters())+list(proj_aef.parameters()), 1.0)
        optimizer.step()

        total += float(loss); steps += 1
        pbar.set_postfix(loss=f"{float(loss):.4f}")
    return total / max(1, steps)

@torch.no_grad()
def validate(model, data_loader, device, epoch):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", iou_thresholds=[0.5], class_metrics=False)
    for images, targets, conds, _ in tqdm(data_loader, desc=f"Val ep{epoch+1}"):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        model.backbone.set_conditioning(conds.to(device))
        preds = model(images)
        metric.update([{k:v.detach().cpu() for k,v in p.items()} for p in preds],
                      [{k:v.detach().cpu() for k,v in t.items()} for t in targets])
    res = metric.compute()
    return float(res.get("map", torch.tensor(0.0))), float(res.get("map", torch.tensor(0.0)))

# =========================
# Evaluation (pretty)
# =========================
@torch.no_grad()
def evaluate_region(detector, root: str, split: str, device,
                    batch_size=8, num_workers=8, image_size=800,
                    title="", labels_dir_name="labels"):
    img_dir = Path(root) / split / "images"
    rows = [{"region": Path(root).name, "filename": f} for f in os.listdir(img_dir) if Path(f).suffix.lower() in IMG_EXTS]
    tmp = Path("phase2_runs"); tmp.mkdir(exist_ok=True, parents=True)
    csv_path = tmp / f"_eval_{Path(root).name}_{split}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["region","filename"]); w.writeheader(); w.writerows(rows)

    ds = BrickKilnDetCSV(str(csv_path), split=split, image_size=image_size, labeled=True, labels_dir_name=labels_dir_name)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

    detector.eval()
    metric_c = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=True,  iou_thresholds=[0.5])
    metric_a = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=False, iou_thresholds=[0.5])

    for images, targets, conds, _ in tqdm(dl, desc=f"Test [{title or split}]"):
        images = [i.to(device) for i in images]
        detector.backbone.set_conditioning(conds.to(device))
        preds = detector(images)
        pc = [{k:v.to('cpu') for k,v in p.items()} for p in preds]
        tc = [{k:v.to('cpu') for k,v in t.items()} for t in targets]
        metric_c.update(pc, tc)
        metric_a.update(
            [{'boxes': p['boxes'], 'scores': p['scores'], 'labels': torch.ones_like(p['labels'])} for p in pc],
            [{'boxes': t['boxes'], 'labels': torch.ones_like(t['labels'])} for t in tc]
        )

    rc = metric_c.compute(); ra = metric_a.compute()
    ca50 = float(ra.get('map', torch.tensor(0.0))) * 100.0
    classes = rc.get('classes', torch.tensor([])).tolist() if 'classes' in rc else []
    ap_pc   = rc.get('map_per_class', torch.tensor([])).tolist() if 'map_per_class' in rc else []
    per_cls = {int(c): float(ap)*100.0 for c, ap in zip(classes, ap_pc) if ap is not None and float(ap)>=0.0}
    mc50 = (sum(per_cls.values())/max(1,len(per_cls))) if per_cls else 0.0

    def g(k): return float(per_cls.get(k, 0.0))
    print("\n" + "="*84)
    print(f" Region: {title or (Path(root).name + ' — ' + split)}")
    print("="*84)
    print(f"{'CA mAP@50':<12}{'MC mAP@50':<12}{'CFCBK@50':<12}{'FCBK@50':<12}{'Zigzag@50':<12}")
    print("-"*84)
    print(f"{ca50:<12.2f}{mc50:<12.2f}{g(1):<12.2f}{g(2):<12.2f}{g(3):<12.2f}")
    print("="*84 + "\n")

# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser("Phase2 End-to-End (DINOv3 + FiLM + Alignment + AEF+RGB pseudo)")
    # Regions / splits
    ap.add_argument("--train_region_src", default="bangladesh", choices=list(REGION_ROOTS.keys()))
    ap.add_argument("--train_region_tgt", default="pak_punjab", choices=list(REGION_ROOTS.keys()))
    ap.add_argument("--val_region",       default="pak_punjab", choices=list(REGION_ROOTS.keys()))
    ap.add_argument("--split_src", default="train")
    ap.add_argument("--split_tgt", default="train")
    ap.add_argument("--split_val", default="val")
    ap.add_argument("--eval_split", default="test")
    ap.add_argument("--image_size", type=int, default=800)
    # CSVs
    ap.add_argument("--csv_src", default="")
    ap.add_argument("--csv_tgt", default="")
    ap.add_argument("--csv_val", default="")
    # Pseudo-labeling
    ap.add_argument("--pl_ckpt_aef", default="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/checkpoints/head_only_s800_e6_b8/best_head_only_uttar_pradesh.pth")
    ap.add_argument("--pl_ckpt_rgb", default="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/checkpoints/best_up_val_map50_dinov3.pth")
    ap.add_argument("--enable_rgb_pl", action="store_true", help="Also generate RGB pseudo labels with DINOv3 teacher.")
    ap.add_argument("--pl_aef_dir", default="labels_pseudo_aef")
    ap.add_argument("--pl_rgb_dir", default="labels_pseudo_rgb")
    ap.add_argument("--pl_out_dir", default="labels_pseudo_merged")
    ap.add_argument("--pl_score_thr", type=float, default=0.5)
    ap.add_argument("--pl_merge_policy", choices=["aef","rgb","nms_union","prefer_rgb","prefer_aef"], default="nms_union")
    ap.add_argument("--pl_iou_merge", type=float, default=0.5)
    ap.add_argument("--gen_pseudo_only", action="store_true")
    # Optim / losses
    ap.add_argument("--epochs",      type=int, default=12)
    ap.add_argument("--batch_size",  type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--backbone_lr", type=float, default=1e-5)
    ap.add_argument("--film_lr",     type=float, default=1e-4)
    ap.add_argument("--head_lr",     type=float, default=1e-4)
    ap.add_argument("--proj_lr",     type=float, default=1e-4)
    ap.add_argument("--weight_decay",type=float, default=0.04)
    ap.add_argument("--lambda_unsup",    type=float, default=1.0)
    ap.add_argument("--lambda_sim_dino", type=float, default=0.5)
    ap.add_argument("--lambda_sim_aef",  type=float, default=0.5)
    ap.add_argument("--lambda_clip_aef", type=float, default=0.0)
    ap.add_argument("--warmup_pl",       type=int, default=2)
    ap.add_argument("--warmup_sim",      type=int, default=2)
    # Modes
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--results_dir", default="phase2_runs")
    return ap.parse_args()

def auto_csv(region, split, preferred=""):
    if preferred: return preferred
    img_dir = Path(REGION_ROOTS[region]) / split / "images"
    rows = [{"region": region, "filename": f} for f in os.listdir(img_dir) if Path(f).suffix.lower() in IMG_EXTS]
    Path("phase2_runs").mkdir(parents=True, exist_ok=True)
    tmp = Path("phase2_runs")/f"{region}_{split}_auto.csv"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["region","filename"]); w.writeheader(); w.writerows(rows)
    return str(tmp)

# =========================
# Main
# =========================
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Student
    dino = load_dinov3()
    model = build_student_detector(dino, num_classes=4, image_size=args.image_size).to(device)

    # CSVs
    csv_src = auto_csv(args.train_region_src, args.split_src, args.csv_src)
    csv_tgt = auto_csv(args.train_region_tgt, args.split_tgt, args.csv_tgt)
    csv_val = auto_csv(args.val_region,       args.split_val, args.csv_val)

    # Stage A: Pseudo labels (AEF → required; RGB → optional)
    print("\n[Stage A] Generating pseudo labels on TARGET...")
    generate_pseudo_labels_aef(
        target_csv=csv_tgt, split=args.split_tgt, save_dir_name=args.pl_aef_dir,
        aef_ckpt=args.pl_ckpt_aef, num_classes=4, image_size=args.image_size,
        score_thr=args.pl_score_thr, device=device
    )
    if args.enable_rgb_pl:
        generate_pseudo_labels_rgb(
            target_csv=csv_tgt, split=args.split_tgt, save_dir_name=args.pl_rgb_dir,
            rgb_ckpt=args.pl_ckpt_rgb, num_classes=4, image_size=args.image_size,
            score_thr=args.pl_score_thr, device=device
        )

    # Optionally fuse
    tgt_labels_dir_for_training = args.pl_aef_dir
    if args.enable_rgb_pl:
        if args.pl_merge_policy == "rgb":
            tgt_labels_dir_for_training = args.pl_rgb_dir
        elif args.pl_merge_policy == "aef":
            tgt_labels_dir_for_training = args.pl_aef_dir
        else:
            prefer = None
            if args.pl_merge_policy in ("prefer_rgb","prefer_aef"):
                prefer = "rgb" if args.pl_merge_policy=="prefer_rgb" else "aef"
            fuse_pseudo_label_dirs(
                target_csv=csv_tgt, split=args.split_tgt,
                aef_dir_name=args.pl_aef_dir, rgb_dir_name=args.pl_rgb_dir, out_dir_name=args.pl_out_dir,
                image_size=args.image_size, iou_thr=args.pl_iou_merge, prefer=prefer
            )
            tgt_labels_dir_for_training = args.pl_out_dir

    if args.gen_pseudo_only:
        print("[Stage A] Pseudo-label generation finished.")
        return

    # Datasets
    ds_src = BrickKilnDetCSV(csv_src, split=args.split_src, image_size=args.image_size, labeled=True,  labels_dir_name="labels")
    ds_tgt = BrickKilnDetCSV(csv_tgt, split=args.split_tgt, image_size=args.image_size, labeled=True,  labels_dir_name=tgt_labels_dir_for_training)
    ds_val = BrickKilnDetCSV(csv_val, split=args.split_val, image_size=args.image_size, labeled=True,  labels_dir_name="labels")

    dl_src = DataLoader(ds_src, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    dl_tgt = DataLoader(ds_tgt, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    # Frozen align copy
    align_copy = copy.deepcopy(model.backbone.dino).eval().to(device)
    for p in align_copy.parameters(): p.requires_grad = False

    # Projectors
    C = model.backbone.out_channels
    proj_feats = FeatProjector(in_dim=C, out_dim=256).to(device)
    proj_aef   = AEFProjector(in_dim=64, out_dim=256, hidden=512).to(device)
    logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07), device=device))

    # Optim & sched
    optimizer = torch.optim.AdamW(
        build_param_groups(model, proj_feats, proj_aef,
                           args.backbone_lr, args.film_lr, args.head_lr, args.proj_lr, args.weight_decay),
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    # Train
    ckpt_dir = Path(args.results_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / f"best_phase2_dinov3_film_align_{args.train_region_src}_to_{args.train_region_tgt}.pth"
    best_m = -1.0

    if not args.eval_only:
        for ep in range(args.epochs):
            tl = train_epoch_dual(model, align_copy, proj_feats, proj_aef, logit_scale,
                                  dl_src, dl_tgt, device, ep, args, optimizer)
            mv, _ = validate(model, dl_val, device, ep)
            print(f"[E{ep+1:02d}] train_loss={tl:.4f}  val_mAP@50={mv:.4f}")
            if mv > best_m:
                best_m = mv
                torch.save({
                    "detector": model.state_dict(),
                    "proj_feats": proj_feats.state_dict(),
                    "proj_aef": proj_aef.state_dict(),
                    "logit_scale": logit_scale.detach().cpu()
                }, best_ckpt)
                print(f"[CKPT] saved -> {best_ckpt}")
            scheduler.step()

    # Load best and evaluate
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location="cpu")
        model.load_state_dict(state["detector"], strict=False)
        proj_feats.load_state_dict(state["proj_feats"])
        proj_aef.load_state_dict(state["proj_aef"])
        logit_scale.data.copy_(state["logit_scale"].to(device))
        print(f"[INFO] Loaded best ckpt: {best_ckpt}")

    # In-region = target
    evaluate_region(model, REGION_ROOTS[args.train_region_tgt], args.eval_split, device,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                    image_size=args.image_size, title=f"{args.train_region_tgt} (IN-REGION)", labels_dir_name="labels")

    # OODs
    for r in [k for k in REGION_ROOTS if k != args.train_region_tgt]:
        evaluate_region(model, REGION_ROOTS[r], args.eval_split, device,
                        batch_size=args.batch_size, num_workers=args.num_workers,
                        image_size=args.image_size, title=f"{r} (OOD)", labels_dir_name="labels")

if __name__ == "__main__":
    main()