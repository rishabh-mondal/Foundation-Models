#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase-2: Train + Validate + Evaluate (Target) — DINOv3 Faster R-CNN (Gram Anchoring)
-------------------------------------------------------------------------------------

• Student: Faster R-CNN with DINOv3 backbone (RGB).
• Teacher: optional EMA of student for online pseudo-labelling.
• Loss:  L = L_det^S + λ_unsup * L_det^T + λ_sim * L_align
    - L_det^S : supervised detection loss on source (weak & strong views)
    - L_det^T : detection loss on target (offline PL before ninitPL; EMA online PL after ema_start_iter)
    - L_align : **Gram anchoring** between student & frozen DINO features (projection matched)
        · Source aligned from iter 0
        · Target added after ninitSim

Use --sim_type gram (default) or cosine to switch alignment flavor.
"""

import os, math, csv, copy, random, inspect
from pathlib import Path
from typing import List, Dict, Optional, Tuple

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
import argparse

# ----------------------------
# Utilities
# ----------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def yolo_txt_to_xyxy(txt_path: Path, W: int, H: int) -> Tuple[List[List[float]], List[int]]:
    boxes, labels = [], []
    if not txt_path or not Path(txt_path).exists(): return boxes, labels
    with open(txt_path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 5:  # cls cx cy w h
                continue
            c = int(float(p[0])) + 1
            cx, cy, w, h = [float(v) for v in p[1:]]
            xmin = (cx - w/2) * W; ymin = (cy - h/2) * H
            xmax = (cx + w/2) * W; ymax = (cy + h/2) * H
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax]); labels.append(c)
    return boxes, labels

# ----------------------------
# Datasets
# ----------------------------
class SourceDetDataset(Dataset):
    def __init__(self, root: str, split: str, image_size: int = 800):
        base = Path(root) if (Path(root) / "images").is_dir() else (Path(root) / split)
        self.img_dir = base / "images"
        self.lab_dir = base / "labels"
        assert self.img_dir.is_dir(), f"Missing images: {self.img_dir}"
        assert self.lab_dir.is_dir(), f"Missing labels: {self.lab_dir}"
        self.files = sorted([f for f in os.listdir(self.img_dir) if Path(f).suffix.lower() in IMG_EXTS])
        self.tf = transforms.Compose([transforms.Resize((image_size, image_size), antialias=True),
                                      transforms.ToTensor()])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx: int):
        name = self.files[idx]
        img_path = self.img_dir / name
        img = Image.open(img_path).convert("RGB")
        x = self.tf(img)
        _, H, W = x.shape

        boxes, labels = [], []
        txt = self.lab_dir / f"{Path(name).stem}.txt"
        if txt.exists():
            with open(txt, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 9:
                        cls_id = int(float(parts[0])) + 1
                        obb = np.array([float(u) for u in parts[1:]], dtype=np.float32)
                        xs, ys = obb[0::2]*W, obb[1::2]*H
                        xmin, ymin, xmax, ymax = xs.min(), ys.min(), xs.max(), ys.max()
                        if xmax > xmin and ymax > ymin:
                            boxes.append([xmin, ymin, xmax, ymax]); labels.append(cls_id)
                    elif len(parts) == 5:
                        b, l = yolo_txt_to_xyxy(txt, W, H); boxes, labels = b, l; break

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1,4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "filename": name,
            "path": str(img_path),
        }
        return x, target

class TargetDetDataset(Dataset):
    def __init__(self, root: str, split: str, image_size: int = 800, offline_pl_dir: Optional[str] = None):
        base = Path(root) if (Path(root) / "images").is_dir() else (Path(root) / split)
        self.img_dir = base / "images"
        self.offline_dir = Path(offline_pl_dir) if offline_pl_dir else None
        assert self.img_dir.is_dir(), f"Missing images: {self.img_dir}"
        self.files = sorted([f for f in os.listdir(self.img_dir) if Path(f).suffix.lower() in IMG_EXTS])
        self.tf = transforms.Compose([transforms.Resize((image_size, image_size), antialias=True),
                                      transforms.ToTensor()])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx: int):
        name = self.files[idx]
        img_path = self.img_dir / name
        img = Image.open(img_path).convert("RGB")
        x = self.tf(img)
        _, H, W = x.shape

        boxes, labels = [], []
        if self.offline_dir:
            txt = self.offline_dir / f"{Path(name).stem}.txt"
            b, l = yolo_txt_to_xyxy(txt, W, H)
            boxes, labels = b, l

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1,4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "filename": name,
            "path": str(img_path),
        }
        return x, target

def collate_fn(batch): imgs, tgts = zip(*batch); return list(imgs), list(tgts)

# ----------------------------
# Backbone wrapper (DINOv3 → FRCNN)
# ----------------------------
class DinoV3Backbone(nn.Module):
    def __init__(self, dino_model: nn.Module, patch_stride: int = 16):
        super().__init__()
        self.dino = dino_model
        self.patch_stride = patch_stride
        C = getattr(dino_model, "embed_dim", None) or getattr(dino_model, "num_features", 1024)
        self.out_channels = int(C)

    def _hw(self, x):
        _, _, H, W = x.shape
        return math.ceil(H/self.patch_stride), math.ceil(W/self.patch_stride)

    def _patch_tokens(self, x):
        try:
            out = self.dino.forward_features(x)
            if isinstance(out, dict):
                if "x_norm_patchtokens" in out:
                    t = out["x_norm_patchtokens"]
                    H, W = out.get("H"), out.get("W")
                    if H is None or W is None: H, W = self._hw(x)
                    return t, H, W
                if "tokens" in out and isinstance(out["tokens"], torch.Tensor):
                    t = out["tokens"]; H, W = self._hw(x)
                    if t.shape[1] == H*W + 1: t = t[:, 1:, :]
                    return t, H, W
            if isinstance(out, torch.Tensor):
                H, W = self._hw(x)
                t = out
                if t.dim() == 3 and t.shape[1] == H*W + 1: t = t[:, 1:, :]
                return t, H, W
        except Exception:
            pass
        t = self.dino(x); H, W = self._hw(x)
        if isinstance(t, (list, tuple)): t = t[-1]
        if t.dim() == 3 and t.shape[1] == H*W + 1: t = t[:, 1:, :]
        return t, H, W

    def forward(self, x):
        toks, H, W = self._patch_tokens(x)
        B, N, C = toks.shape
        fmap = toks.transpose(1, 2).contiguous().view(B, C, H, W)
        return {"0": fmap}

def build_detector(dino_model, num_classes, image_size):
    backbone = DinoV3Backbone(dino_model, 16)
    anchors = AnchorGenerator(sizes=((16,32,64,128,256),), aspect_ratios=((0.5,1.0,2.0),))
    return FasterRCNN(backbone=backbone, num_classes=num_classes,
                      rpn_anchor_generator=anchors, min_size=image_size, max_size=image_size)

# ----------------------------
# Alignment (Gram anchoring + optional cosine)
# ----------------------------
class FeatProjector(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1), nn.GELU(),
            nn.Conv2d(in_dim, out_dim, 1)
        )
    def forward(self, f):  # [B,C,H,W] -> [B, D, H, W] (L2-normalized along channel)
        z = self.net(f)
        return F.normalize(z, dim=1)

@torch.no_grad()
def frozen_dino_patch_map(dino_model, x: torch.Tensor):
    out = None
    try:
        out = dino_model.forward_features(x)
        if isinstance(out, dict) and "x_norm_patchtokens" in out:
            t = out["x_norm_patchtokens"]
        else:
            t = out if isinstance(out, torch.Tensor) else None
    except Exception:
        t = None
    if t is None:
        gil = getattr(dino_model, "get_intermediate_layers", None)
        if gil is not None:
            t = gil(x, 1)
            if isinstance(t, (list, tuple)): t = t[0]
    if t is None:
        t = dino_model(x)
        if isinstance(t, (list, tuple)): t = t[-1]
    B, N, C = t.shape
    H = int(round(N**0.5)); W = max(1, N // max(1, H))
    if t.shape[1] == H*W + 1: t = t[:, 1:, :]
    f = t[:, :H*W, :].transpose(1, 2).reshape(B, C, H, W)
    return F.normalize(f, dim=1)

def gram_matrix(f: torch.Tensor) -> torch.Tensor:
    """
    f: [B, D, H, W] (assume L2-normalized along D already)
    returns G: [B, D, D] with per-sample channel Gram matrices normalized by spatial size.
    """
    B, D, H, W = f.shape
    x = f.view(B, D, H * W)                          # [B, D, HW]
    G = torch.matmul(x, x.transpose(1, 2)) / (H * W) # [B, D, D]
    return G

def gram_anchoring_loss(Fs: torch.Tensor, Ft: torch.Tensor, reduction="mean") -> torch.Tensor:
    """
    Fs, Ft: [B, D, H, W] (L2-normalized feature maps)
    Builds Gram matrices and matches them (MSE on normalized Grams).
    """
    Gs = gram_matrix(Fs)
    Gt = gram_matrix(Ft)
    # Optional channel-wise normalization to equalize scale across samples
    Gs = F.normalize(Gs.reshape(Gs.shape[0], -1), dim=1).view_as(Gs)
    Gt = F.normalize(Gt.reshape(Gt.shape[0], -1), dim=1).view_as(Gt)
    loss = F.mse_loss(Gs, Gt, reduction=reduction)
    return loss

def cosine_patch_loss(Fs, Ft):
    return (1.0 - (Fs * Ft).sum(dim=1)).mean()

# ----------------------------
# EMA teacher utilities
# ----------------------------
def update_ema(student, teacher, decay):
    with torch.no_grad():
        for ps, pt in zip(student.parameters(), teacher.parameters()):
            pt.data.mul_(decay).add_(ps.data, alpha=1.0 - decay)

def detach_pseudo_from_preds(preds, score_thr=0.7):
    out = []
    for p in preds:
        keep = p["scores"] >= score_thr
        out.append({"boxes": p["boxes"][keep].detach(), "labels": p["labels"][keep].detach()})
    return out

# ----------------------------
# Augmentations
# ----------------------------
def build_augs(size):
    weak = transforms.Compose([
        transforms.Resize((size, size), antialias=True),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ])
    strong = transforms.Compose([
        transforms.Resize((size, size), antialias=True),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(0.2),
        transforms.ToTensor(),
    ])
    return weak, strong

# ----------------------------
# Validation (mAP@50)
# ----------------------------
@torch.no_grad()
def validate(model, data_loader, device, epoch=0):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", iou_thresholds=[0.5], class_metrics=False)
    for images, targets in tqdm(data_loader, desc=f"[Val ep{epoch+1}]"):
        images  = [img.to(device) for img in images]
        targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
        preds   = model(images)
        preds   = [{k: v.detach().cpu() for k, v in p.items()} for p in preds]
        tgts    = [{k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
        metric.update(preds, tgts)
    res = metric.compute()
    return float(res.get("map_50", res.get("map", torch.tensor(0.0))))

# ----------------------------
# Target evaluation (class-agnostic + per-class @50)
# ----------------------------
@torch.no_grad()
def evaluate_region(model, root: str, split: str, device,
                    batch_size=8, num_workers=8, image_size=800,
                    title="", results_csv=None):
    ds = SourceDetDataset(root, split, image_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=True, collate_fn=collate_fn)

    model.eval()
    m_class = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=True,  iou_thresholds=[0.5])
    m_agn   = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=False, iou_thresholds=[0.5])

    for images, targets in tqdm(dl, desc=f"[Eval {title or (Path(root).name + '/' + split)}]"):
        images = [i.to(device) for i in images]
        preds  = model(images)
        preds_cpu = [{k: v.to('cpu') for k, v in p.items()} for p in preds]
        tgts_cpu  = [{k: v.to('cpu') if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

        m_class.update(preds_cpu, tgts_cpu)
        preds_agn = [{'boxes': p['boxes'], 'scores': p['scores'],
                      'labels': torch.ones_like(p['labels'])} for p in preds_cpu]
        tgts_agn  = [{'boxes': t['boxes'],
                      'labels': torch.ones_like(t['labels'])} for t in tgts_cpu]
        m_agn.update(preds_agn, tgts_agn)

    res_class = m_class.compute()
    res_agn   = m_agn.compute()

    ca_map50 = float(res_agn.get('map_50', res_agn.get('map', torch.tensor(0.0)))) * 100.0
    classes  = res_class.get('classes', torch.tensor([])).tolist() if 'classes' in res_class else []
    ap_list  = res_class.get('map_per_class', torch.tensor([])).tolist() if 'map_per_class' in res_class else []

    per_cls = {}
    for c, ap in zip(classes, ap_list):
        try:
            apf = float(ap)
            if np.isfinite(apf) and apf >= 0.0:
                per_cls[int(c)] = apf * 100.0
        except Exception:
            pass

    mc_map50 = (sum(per_cls.values()) / max(1, len(per_cls))) if per_cls else 0.0
    def g(k): return float(per_cls.get(k, 0.0))

    print("\n" + "=" * 84)
    print(f" Region: {title or (Path(root).name + ' — ' + split)}")
    print("=" * 84)
    print(f"{'CA mAP@50':<12}{'MC mAP@50':<12}{'CFCBK@50':<12}{'FCBK@50':<12}{'Zigzag@50':<12}")
    print("-" * 84)
    print(f"{ca_map50:<12.2f}{mc_map50:<12.2f}{g(1):<12.2f}{g(2):<12.2f}{g(3):<12.2f}")
    print("=" * 84 + "\n")

    if results_csv is not None:
        is_new = not os.path.exists(results_csv)
        with open(results_csv, "a", newline="") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(["Region", "Split", "CA_mAP50", "MC_mAP50",
                            "CFCBK_mAP50", "FCBK_mAP50", "Zigzag_mAP50"])
            w.writerow([title or Path(root).name, split,
                        f"{ca_map50:.2f}", f"{mc_map50:.2f}",
                        f"{g(1):.2f}", f"{g(2):.2f}", f"{g(3):.2f}"])
    return ca_map50, mc_map50, per_cls

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser("Phase-2 Train/Val/Eval with DINOv3 Faster R-CNN (Gram)")
    # Data (train/val)
    ap.add_argument("--src_root", required=True)
    ap.add_argument("--src_split", default="train")
    ap.add_argument("--tgt_roots", nargs="+", required=True)
    ap.add_argument("--tgt_split", default="train")
    ap.add_argument("--val_root", required=True)
    ap.add_argument("--val_split", default="val")
    ap.add_argument("--offline_pl_dirname", default="pseudo_fused_labels")
    # DINOv3
    ap.add_argument("--dinov3_dir", required=True)
    ap.add_argument("--dino_model", default="dinov3_vitl16")
    ap.add_argument("--dino_weights", required=True)
    # Model & train
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--image_size", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--backbone_lr", type=float, default=1e-5)
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.04)
    # Gates
    ap.add_argument("--ninitPL", type=int, default=20000)
    ap.add_argument("--ninitSim", type=int, default=5000)
    # Loss weights
    ap.add_argument("--lambda_unsup", type=float, default=1.0)
    ap.add_argument("--lambda_sim", type=float, default=0.5)
    # Alignment kind
    ap.add_argument("--sim_type", choices=["gram", "cosine"], default="gram")
    # EMA
    ap.add_argument("--use_ema_teacher", action="store_true")
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--ema_start_iter", type=int, default=20000)
    ap.add_argument("--pl_score_thr", type=float, default=0.70)
    # Eval on target
    ap.add_argument("--eval_tgt_roots", nargs="+", default=[])
    ap.add_argument("--eval_tgt_split", default="test")
    ap.add_argument("--eval_csv", default="phase2_target_eval.csv")
    # Misc
    ap.add_argument("--save_dir", default="phase2_runs")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

# ----------------------------
# Main (train -> val -> eval[target])
# ----------------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Student
    dino_student = torch.hub.load(repo_or_dir=args.dinov3_dir,
                                  model=args.dino_model,
                                  source="local",
                                  weights=args.dino_weights,
                                  skip_validation=True)
    student = build_detector(dino_student, args.num_classes, args.image_size).to(device)

    # Frozen alignment encoder
    dino_align = torch.hub.load(repo_or_dir=args.dinov3_dir,
                                model=args.dino_model,
                                source="local",
                                weights=args.dino_weights,
                                skip_validation=True)
    for p in dino_align.parameters(): p.requires_grad = False
    dino_align.eval().to(device)

    # Alignment projectors
    projector_student = FeatProjector(in_dim=student.backbone.out_channels, out_dim=256).to(device)
    teacher_in_dim = getattr(dino_align, "embed_dim", None) or getattr(dino_align, "num_features", 1024)
    projector_teacher = FeatProjector(in_dim=int(teacher_in_dim), out_dim=256).to(device)
    for p in projector_teacher.parameters(): p.requires_grad = False

    # EMA teacher
    teacher = None
    if args.use_ema_teacher:
        teacher = copy.deepcopy(student).to(device).eval()
        for p in teacher.parameters(): p.requires_grad = False

    # Dataloaders
    src_dl = DataLoader(SourceDetDataset(args.src_root, args.src_split, args.image_size),
                        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        collate_fn=collate_fn, drop_last=True, pin_memory=True)
    val_dl = DataLoader(SourceDetDataset(args.val_root, args.val_split, args.image_size),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        collate_fn=collate_fn, pin_memory=True)
    tgt_dls = []
    for root in args.tgt_roots:
        offline_dir = Path(root) / args.tgt_split / args.offline_pl_dirname
        if not offline_dir.exists():
            alt = Path(root) / args.offline_pl_dirname
            offline_dir = alt if alt.exists() else None
        tgt_dls.append(DataLoader(
            TargetDetDataset(root, args.tgt_split, args.image_size, str(offline_dir) if offline_dir else None),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
            collate_fn=collate_fn, drop_last=True, pin_memory=True
        ))

    # Optimizer
    back_params = [p for n, p in student.named_parameters() if p.requires_grad and n.startswith("backbone.")]
    head_params = [p for n, p in student.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW(
        [{"params": back_params, "lr": args.backbone_lr},
         {"params": head_params, "lr": args.head_lr},
         {"params": projector_student.parameters(), "lr": args.head_lr}],
        weight_decay=args.weight_decay
    )

    weak_aug, strong_aug = build_augs(args.image_size)
    total_iters = 0
    best_map = -1.0
    ckpt_stem = f"phase2_student_{args.sim_type}"
    ckpt_best = Path(args.save_dir) / f"{ckpt_stem}_best.pth"
    ckpt_last = Path(args.save_dir) / f"{ckpt_stem}_last.pth"
    # ckpt_path = Path(args.save_dir) / "best_phase2_student.pth"

    # ---------------- Train + Val ----------------
    for epoch in range(args.epochs):
        student.train(); projector_student.train()
        tgt_iterators = [iter(dl) for dl in tgt_dls]
        pbar = tqdm(src_dl, desc=f"[Train ep{epoch+1}]")
        for imgs_s, tgts_s in pbar:
            total_iters += 1

            # Weak/Strong views
            try:
                xS_w = [weak_aug(Image.open(t["path"]).convert("RGB")).to(device) for t in tgts_s]
                xS_s = [strong_aug(Image.open(t["path"]).convert("RGB")).to(device) for t in tgts_s]
            except Exception:
                xS_w = [i.to(device) for i in imgs_s]; xS_s = xS_w

            # Round-robin target
            dl = tgt_dls[total_iters % len(tgt_dls)]
            try:
                imgs_t, tgts_t = next(tgt_iterators[total_iters % len(tgt_dls)])
            except StopIteration:
                tgt_iterators[total_iters % len(tgt_dls)] = iter(dl)
                imgs_t, tgts_t = next(tgt_iterators[total_iters % len(tgt_dls)])

            try:
                xT_w = [weak_aug(Image.open(t["path"]).convert("RGB")).to(device) for t in tgts_t]
                xT_s = [strong_aug(Image.open(t["path"]).convert("RGB")).to(device) for t in tgts_t]
            except Exception:
                xT_w = [i.to(device) for i in imgs_t]; xT_s = xT_w

            tgts_s = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in tgts_s]
            pseudo_offline = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in tgts_t]

            # 1) Supervised det on source
            loss_Sw = student(xS_w, tgts_s)
            loss_Ss = student(xS_s, tgts_s)
            L_det_S = (sum(loss_Sw.values()) + sum(loss_Ss.values())) * 0.5

            # 2) Unsupervised det on target (offline → EMA)
            if total_iters >= args.ninitPL:
                if args.use_ema_teacher and total_iters >= args.ema_start_iter:
                    with torch.no_grad():
                        teacher.eval()
                        preds = teacher(xT_w)
                        pseudo_T = detach_pseudo_from_preds(preds, score_thr=args.pl_score_thr)
                else:
                    pseudo_T = pseudo_offline
                L_det_T = sum(student(xT_s, pseudo_T).values())
            else:
                L_det_T = torch.tensor(0.0, device=device)

            # 3) Alignment loss (source always; target after ninitSim)
            Fs = student.backbone(torch.stack(xS_w))["0"]
            Zs = projector_student(Fs)
            Fs_ref = frozen_dino_patch_map(dino_align, torch.stack(xS_w))
            Zs_ref = projector_teacher(Fs_ref)

            if args.sim_type == "gram":
                L_sim_S = gram_anchoring_loss(Zs, Zs_ref)
            else:
                L_sim_S = cosine_patch_loss(Zs, Zs_ref)

            if total_iters >= args.ninitSim:
                Ft = student.backbone(torch.stack(xT_w))["0"]
                Zt = projector_student(Ft)
                Ft_ref = frozen_dino_patch_map(dino_align, torch.stack(xT_w))
                Zt_ref = projector_teacher(Ft_ref)

                if args.sim_type == "gram":
                    L_sim_T = gram_anchoring_loss(Zt, Zt_ref)
                else:
                    L_sim_T = cosine_patch_loss(Zt, Zt_ref)
            else:
                L_sim_T = torch.tensor(0.0, device=device)

            L_sim = L_sim_S + L_sim_T
            L_total = L_det_S + args.lambda_unsup * L_det_T + args.lambda_sim * L_sim

            optimizer.zero_grad(set_to_none=True)
            L_total.backward()
            nn.utils.clip_grad_norm_(list(student.parameters()) + list(projector_student.parameters()), 1.0)
            optimizer.step()

            if args.use_ema_teacher:
                update_ema(student, teacher, args.ema_decay)

            pbar.set_postfix(it=total_iters, L=float(L_total),
                             Ls=float(L_det_S), Lt=float(L_det_T), Lsim=float(L_sim))

        # Validation
        val_map = validate(student.eval(), val_dl, device, epoch)
        print(f"[E{epoch+1:02d}] val mAP@50 = {val_map:.4f}")
        if val_map > best_map:
            best_map = val_map
            to_save = {
                "student": student.state_dict(),
                "projector_student": projector_student.state_dict(),
                "best_map50": best_map,
                "sim_type": args.sim_type,
            }
            if args.use_ema_teacher and teacher is not None:
                to_save["ema_teacher"] = teacher.state_dict()
            torch.save(to_save, ckpt_best)
            print(f"[CKPT] saved -> {ckpt_best} (mAP50={best_map:.4f})")
    # at the END of training, always save a "last" snapshot too:
        to_save_last = {
            "student": student.state_dict(),
            "projector_student": projector_student.state_dict(),
            "best_map50": best_map,
            "sim_type": args.sim_type,
        }
        if args.use_ema_teacher and teacher is not None:
            to_save_last["ema_teacher"] = teacher.state_dict()
        torch.save(to_save_last, ckpt_last)
        print(f"[CKPT] saved last -> {ckpt_last}")
        print("✅ Training finished.")

    # ---------------- Final Evaluation on Target Regions ----------------
    if args.eval_tgt_roots:
        print("\n=== Final Evaluation on Target Regions ===")
        state = torch.load(ckpt_best, map_location="cpu")
        student.load_state_dict(state["student"], strict=False)
        student.to(device).eval()
        for tgt_root in args.eval_tgt_roots:
            evaluate_region(
                model=student,
                root=tgt_root,
                split=args.eval_tgt_split,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                image_size=args.image_size,
                title=f"{Path(tgt_root).name} — {args.eval_tgt_split}",
                results_csv=str(Path(args.save_dir) / args.eval_csv),
            )

if __name__ == "__main__":
    main()