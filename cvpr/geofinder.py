#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchmetrics.detection import MeanAveragePrecision

# =========================
# Configuration
# =========================
DINOV3_GITHUB_LOCATION = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/dinov3"
DINOV3_LOCATION = os.getenv("DINOV3_LOCATION") or DINOV3_GITHUB_LOCATION
DINO_MODEL_NAME = "dinov3_vitl16"
DINO_WEIGHTS = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/dinov3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
assert os.path.exists(DINO_WEIGHTS), f"Missing DINO weights file: {DINO_WEIGHTS}"

# UP_ROOT  = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/uttar_pradesh"
# BD_ROOT  = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/bangladesh"
# PKP_ROOT = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/pak_punjab"

XVIEW_ROOT="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/xview/processed/split"
PSUDO_ROOT="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/psudo_data"
DOTA_ROOT="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/dota/processed"
TEST_ROOT="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/xview/processed/split"

IMAGE_SIZE    = 512
BATCH_SIZE    = 32
NUM_WORKERS   = 8
NUM_EPOCHS    = 25
BACKBONE_LR   = 1e-5
HEAD_LR       = 1e-4
WEIGHT_DECAY  = 0.04
NUM_CLASSES   = 4  # background plus 3 kiln classes
mode="freeze"  # "freeze" or "unfreeze"
note="satellite-base-model-base-of-dota-xview-data-512px"
OUT_DIR       = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/checkpoints"
os.makedirs(OUT_DIR, exist_ok=True)
# BEST_CKPT     = os.path.join(OUT_DIR, f"best_val_map50_dinov3_{mode}_dino_{note}.pth")
BEST_CKPT= "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/checkpoints/best_val_map50_dinov3_freeze_dino_satellite-base-model-base-of-dota-xview-data-512px.pth"
RESULTS_CSV   = os.path.join(OUT_DIR, f"region_eval_final_{mode}_dino-{note}.csv")

CLIP_NORM     = 1.0
PRINT_EVERY   = 20

# =========================
# Knobs
# =========================
USE_FINETUNE_CKPT = False
FINETUNE_CKPT = f"/home/kirtangangani/satellite_data/dino_vitl16_aligned_epoch10.pth"
TRAIN_BACKBONE = False if mode == "freeze" else True

# =========================
# Dataset
# =========================
# class BrickKilnDataset(Dataset):
#     """
#     Layout: <root>/<split>/{images,labels}
#     YOLO OBB row: <cls> x1 y1 x2 y2 x3 y3 x4 y4 in range [0,1]
#     Converted to axis aligned XYXY for Faster R CNN
#     """
#     IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

#     def __init__(self, root: str, split: str, input_size: int = 224):
#         self.root = Path(root)
#         self.split = split
#         cand = self.root if (self.root / "images").is_dir() else (self.root / split)
#         self.img_dir = cand / "images"
#         self.label_dir = cand / "labels"
#         if not self.img_dir.is_dir():
#             raise FileNotFoundError(f"Missing images directory: {self.img_dir}")
#         if not self.label_dir.is_dir():
#             raise FileNotFoundError(f"Missing labels directory: {self.label_dir}")

#         self.input_size = int(input_size)
#         self.transform = transforms.Compose([
#             transforms.Resize((self.input_size, self.input_size),
#                               interpolation=transforms.InterpolationMode.BILINEAR,
#                               antialias=True),
#             transforms.ToTensor(),
#         ])

#         self.img_files: List[str] = sorted(
#             [f for f in os.listdir(self.img_dir) if Path(f).suffix.lower() in self.IMG_EXTS]
#         )

#     def __len__(self):
#         return len(self.img_files)

#     def __getitem__(self, idx: int):
#         img_name = self.img_files[idx]
#         img_path = self.img_dir / img_name
#         label_path = self.label_dir / f"{Path(img_name).stem}.txt"

#         try:
#             img = Image.open(img_path).convert("RGB")
#         except (UnidentifiedImageError, OSError):
#             img = Image.fromarray(np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8))
#         img_tensor = self.transform(img)
#         _, Ht, Wt = img_tensor.shape

#         boxes, labels = [], []
#         if label_path.exists():
#             with open(label_path, "r") as f:
#                 for line in f:
#                     parts = line.strip().split()
#                     if len(parts) != 9:
#                         continue
#                     try:
#                         cls_id = int(float(parts[0])) + 1  # shift to 1..3
#                         obb = np.array([float(p) for p in parts[1:]], dtype=np.float32)
#                         xs = obb[0::2] * Wt
#                         ys = obb[1::2] * Ht
#                         xmin, ymin = float(np.min(xs)), float(np.min(ys))
#                         xmax, ymax = float(np.max(xs)), float(np.max(ys))
#                         if xmax > xmin and ymax > ymin:
#                             boxes.append([xmin, ymin, xmax, ymax])
#                             labels.append(cls_id)
#                     except ValueError:
#                         continue

#         target = {
#             "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
#             "labels": torch.as_tensor(labels, dtype=torch.int64),
#             "image_id": torch.tensor([idx]),
#         }
#         return img_tensor, target
# =========================
# Dataset (supports YOLO_AA and YOLO_OBB)
# =========================
class BrickKilnDataset(Dataset):
    """
    Layout: <root>/<split>/{images,labels}
    Label formats (per line):
      - YOLO_AA : <cls> cx cy w h        (normalized)
      - YOLO_OBB: <cls> x1 y1 ... x4 y4  (normalized)
    Outputs: target["boxes"] -> Tensor[N,4] in XYXY absolute pixels
             target["labels"] -> int64 in {1..C}
    """
    IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

    def __init__(self, root: str, split: str, input_size: int = 224,
                 force_label_fmt: str | None = None):
        """
        force_label_fmt: None|"aa"|"obb"
          - None: auto-detect per file (5 tokens -> aa, 9 tokens -> obb)
          - "aa": treat all as YOLO_AA
          - "obb": treat all as YOLO_OBB
        """
        self.root = Path(root)
        self.split = split
        cand = self.root if (self.root / "images").is_dir() else (self.root / split)
        self.img_dir = cand / "images"
        self.label_dir = cand / "labels"
        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"Missing images directory: {self.img_dir}")
        if not self.label_dir.is_dir():
            raise FileNotFoundError(f"Missing labels directory: {self.label_dir}")

        self.input_size = int(input_size)
        self.force_label_fmt = force_label_fmt  # "aa"/"obb"/None

        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size),
                              interpolation=transforms.InterpolationMode.BILINEAR,
                              antialias=True),
            transforms.ToTensor(),
        ])

        self.img_files: List[str] = sorted(
            [f for f in os.listdir(self.img_dir) if Path(f).suffix.lower() in self.IMG_EXTS]
        )

    def __len__(self):
        return len(self.img_files)

    @staticmethod
    def _detect_label_fmt(first_nonempty_line: str) -> str:
        # returns "aa" or "obb"
        parts = first_nonempty_line.strip().split()
        return "aa" if len(parts) == 5 else "obb"  # minimal, robust for your two cases

    @staticmethod
    def _xywhn_to_xyxy_abs(cx, cy, w, h, W, H):
        x1 = (cx - w/2.0) * W
        y1 = (cy - h/2.0) * H
        x2 = (cx + w/2.0) * W
        y2 = (cy + h/2.0) * H
        return x1, y1, x2, y2

    def __getitem__(self, idx: int):
        img_name = self.img_files[idx]
        img_path = self.img_dir / img_name
        label_path = self.label_dir / f"{Path(img_name).stem}.txt"

        # image
        try:
            img = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            img = Image.fromarray(np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8))
        x = self.transform(img)
        _, Ht, Wt = x.shape  # transformed size

        boxes, labels = [], []
        if label_path.exists():
            # read once; decide format
            with open(label_path, "r") as f:
                lines = [ln for ln in (l.strip() for l in f) if ln]

            fmt = self.force_label_fmt
            if fmt is None and lines:
                fmt = self._detect_label_fmt(lines[0])

            for line in lines:
                parts = line.split()
                try:
                    cls_id_raw = int(float(parts[0]))
                except ValueError:
                    continue

                if fmt == "aa":  # YOLO_AA: <cls> cx cy w h (normalized)
                    if len(parts) < 5:
                        continue
                    try:
                        cx, cy, w, h = map(float, parts[1:5])
                    except ValueError:
                        continue
                    x1, y1, x2, y2 = self._xywhn_to_xyxy_abs(cx, cy, w, h, Wt, Ht)

                else:  # "obb": YOLO_OBB: <cls> x1 y1 x2 y2 x3 y3 x4 y4 (normalized)
                    if len(parts) < 9:
                        continue
                    try:
                        obb = np.array([float(p) for p in parts[1:9]], dtype=np.float32)
                    except ValueError:
                        continue
                    xs = obb[0::2] * Wt
                    ys = obb[1::2] * Ht
                    x1, y1 = float(np.min(xs)), float(np.min(ys))
                    x2, y2 = float(np.max(xs)), float(np.max(ys))

                # clip to image bounds
                x1 = max(0.0, min(x1, Wt - 1))
                y1 = max(0.0, min(y1, Ht - 1))
                x2 = max(0.0, min(x2, Wt - 1))
                y2 = max(0.0, min(y2, Ht - 1))

                # filter degenerate
                if x2 <= x1 or y2 <= y1:
                    continue

                # shift labels to 1..C for Faster R-CNN (0 is background)
                cls_id = cls_id_raw + 1
                boxes.append([x1, y1, x2, y2])
                labels.append(cls_id)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return x, target

def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)

# =========================
# DINOv3 Backbone Wrapper
# =========================
class DinoV3BackboneWrapper(nn.Module):
    """Return dict with key "0" mapping to Tensor[B, C, H16, W16], set out_channels=C."""
    def __init__(self, dino_model: nn.Module, patch_stride: int = 16):
        super().__init__()
        self.dino = dino_model
        self.patch_stride = patch_stride
        C = getattr(dino_model, "embed_dim", None) or getattr(dino_model, "num_features", None)
        if C is None:
            with torch.no_grad():
                x = torch.zeros(1, 3, 32, 32)
                tokens, Ht, Wt = self._get_patch_tokens(x)
                C = tokens.shape[-1] if tokens is not None else 1024
        self.out_channels = C

    @torch.no_grad()
    def _maybe_h_w(self, x):
        _, _, H, W = x.shape
        return math.ceil(H / self.patch_stride), math.ceil(W / self.patch_stride)

    def _get_patch_tokens(self, x):
        try:
            out = self.dino.forward_features(x)
            if isinstance(out, dict):
                if "x_norm_patchtokens" in out and out["x_norm_patchtokens"] is not None:
                    tokens = out["x_norm_patchtokens"]
                    Ht = out.get("H") or self._maybe_h_w(x)[0]
                    Wt = out.get("W") or self._maybe_h_w(x)[1]
                    return tokens, Ht, Wt
                if "tokens" in out and out["tokens"] is not None:
                    t = out["tokens"]
                    Ht, Wt = self._maybe_h_w(x)
                    if t.shape[1] == (Ht * Wt + 1):
                        t = t[:, 1:, :]
                    return t, Ht, Wt
            if isinstance(out, torch.Tensor):
                t = out
                Ht, Wt = self._maybe_h_w(x)
                N = Ht * Wt
                if t.shape[1] == N + 1:
                    t = t[:, 1:, :]
                elif t.shape[1] != N:
                    N = t.shape[1]
                    Wt = int(round(math.sqrt(N)))
                    Ht = N // Wt
                return t, Ht, Wt
        except Exception:
            pass

        if hasattr(self.dino, "get_intermediate_layers"):
            t = self.dino.get_intermediate_layers(x, n=1, return_class_token=False)[0]
            Ht, Wt = self._maybe_h_w(x)
            return t, Ht, Wt

        t = self.dino(x)
        Ht, Wt = self._maybe_h_w(x)
        if t.dim() == 3 and t.shape[1] == (Ht * Wt + 1):
            t = t[:, 1:, :]
        return t, Ht, Wt

    def forward(self, x: torch.Tensor):
        tokens, Ht, Wt = self._get_patch_tokens(x)
        if tokens is None or tokens.numel() == 0:
            raise RuntimeError("DINO backbone returned no tokens")
        B, N, C = tokens.shape
        feat = tokens.transpose(1, 2).contiguous().view(B, C, Ht, Wt)
        return {"0": feat}


def create_model(dino_model: nn.Module, num_classes: int, image_size: int = 800) -> FasterRCNN:
    backbone = DinoV3BackboneWrapper(dino_model, patch_stride=16)
    anchor_generator = AnchorGenerator(
        sizes=((4,8,16, 32, 64, 128),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        min_size=image_size,
        max_size=image_size,
        image_mean=[0.430, 0.411, 0.296],
        image_std=[0.213, 0.156, 0.143],
    )
    return model

# =========================
# Optimizer and Scheduler
# =========================
def split_backbone_head_params(model: FasterRCNN):
    bb, head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone") or n.startswith("transform"):
            bb.append(p)
        else:
            head.append(p)
    return bb, head

def build_optimizer(model: FasterRCNN):
    bb_params, head_params = split_backbone_head_params(model)
    param_groups = []
    if bb_params:  # only add if non-empty
        param_groups.append({"params": bb_params, "lr": BACKBONE_LR})
    if head_params:
        param_groups.append({"params": head_params, "lr": HEAD_LR})
    return torch.optim.AdamW(
        param_groups,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

def build_scheduler(optimizer, steps_per_epoch: int):
    warmup = max(steps_per_epoch, 1)
    T = max(NUM_EPOCHS * steps_per_epoch, warmup + 1)
    def lr_lambda(step):
        if step < warmup:
            return float(step + 1) / float(warmup)
        t = (step - warmup) / max(T - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# =========================
# Validation
# =========================
@torch.no_grad()
def validate(model, data_loader, device, epoch=0):
    model.eval()
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=[0.5],
        class_metrics=False
    )
    for batch in tqdm(data_loader, desc=f"Inference epoch {epoch+1}"):
        images, targets = batch
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        try:
            preds = model(images)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            continue
        preds   = [{k: v.detach().cpu() for k, v in p.items()} for p in preds]
        targets = [{k: v.detach().cpu() for k, v in t.items()} for t in targets]
        metric.update(preds, targets)
    res   = metric.compute()
    map50 = float(res.get("map_50", res.get("map", torch.tensor(0.0))))
    print(f"mAP@50 first one is for val 2 for test data: {map50:.4f}")
    return map50

# =========================
# Training
# =========================
def train_one_epoch(model, loader, device, optimizer, epoch=0, clip_norm=None, print_every=PRINT_EVERY, scheduler=None):
    model.train()
    running = 0.0
    step = 0
    pbar = tqdm(loader, desc=f"Train epoch {epoch+1}")
    for images, targets in pbar:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        try:
            losses: dict = model(images, targets)
            loss = sum(v for v in losses.values())
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            loss_val = float(loss.detach().cpu())
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            continue

        running += loss_val
        step += 1
        if step % print_every == 0:
            pbar.set_postfix(loss=f"{loss_val:.4f}")
    denom = max(step, 1)
    return running / denom

# =========================
# Evaluation helpers
# =========================
@torch.no_grad()
def evaluate_region(model, root: str, split: str, device,
                    batch_size=16, num_workers=8, image_size=800,
                    title="", results_csv=None):
    ds = BrickKilnDataset(root=root, split=split, input_size=image_size)
    if len(ds) == 0:
        print(f"[WARN] Empty dataset for {root} {split}")
        return 0.0, 0.0, {}

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True,
                    collate_fn=collate_fn)

    model.eval()
    metric_class = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True,  iou_thresholds=[0.5])
    metric_agn   = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=False, iou_thresholds=[0.5])

    for images, targets in tqdm(dl, desc=f"Test [{title or split}]"):
        images = [i.to(device) for i in images]
        try:
            preds = model(images)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            continue
        preds_cpu = [{k: v.to("cpu") for k, v in p.items()} for p in preds]
        tgts_cpu  = [{k: v.to("cpu") for k, v in t.items()} for t in targets]
        metric_class.update(preds_cpu, tgts_cpu)

        preds_agn = [{"boxes": p["boxes"], "scores": p["scores"],
                      "labels": torch.ones_like(p["labels"])} for p in preds_cpu]
        tgts_agn  = [{"boxes": t["boxes"],
                      "labels": torch.ones_like(t["labels"])} for t in tgts_cpu]
        metric_agn.update(preds_agn, tgts_agn)

    res_class = metric_class.compute()
    res_agn   = metric_agn.compute()
    ca_map50 = float(res_agn.get("map_50", res_agn.get("map", torch.tensor(0.0)))) * 100.0

    classes = res_class.get("classes", torch.tensor([])).tolist() if "classes" in res_class else []
    ap_list = res_class.get("map_per_class", torch.tensor([])).tolist() if "map_per_class" in res_class else []
    valid_pairs = []
    for c, ap in zip(classes, ap_list):
        try:
            apf = float(ap)
        except Exception:
            continue
        if apf >= 0.0 and np.isfinite(apf):
            valid_pairs.append((int(c), apf))
    per_cls = {c: ap * 100.0 for c, ap in valid_pairs}
    mc_map50 = (sum(ap for _, ap in valid_pairs) / len(valid_pairs) * 100.0) if valid_pairs else 0.0

    def g(k): return float(per_cls.get(k, 0.0))
    print("\n" + "=" * 84)
    print(f" Region: {title or (Path(root).name + ' — ' + split)}")
    print("=" * 84)
    print(f"{'CA mAP@50':<12}{'MC mAP@50':<12}{'C_0@50':<12}{'C_1@50':<12}{'C_2@50':<12}")
    print("-" * 84)
    print(f"{ca_map50:<12.2f}{mc_map50:<12.2f}{g(1):<12.2f}{g(2):<12.2f}{g(3):<12.2f}")
    print("=" * 84 + "\n")

    if results_csv is not None:
        is_new = not os.path.exists(results_csv)
        with open(results_csv, "a", newline="") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(["Region", "Split", "CA_mAP50", "MC_mAP50",
                            "C_0_mAP50", "C_1_mAP50", "C_2_mAP50"])
            w.writerow([title or Path(root).name, split,
                        f"{ca_map50:.2f}", f"{mc_map50:.2f}",
                        f"{g(1):.2f}", f"{g(2):.2f}", f"{g(3):.2f}"])
    return ca_map50, mc_map50, per_cls

# =========================
# Backbone helpers
# =========================
def load_into_dino_backbone(dino_model: nn.Module, ckpt_path: str):
    if not os.path.exists(ckpt_path):
        print(f"[WARN] Finetune checkpoint not found: {ckpt_path}")
        return
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    clean = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("backbone.") or nk.startswith("encoder."):
            nk = nk.split(".", 1)[1]
        clean[nk] = v
    res = dino_model.load_state_dict(clean, strict=False)
    missing = len(getattr(res, "missing_keys", []))
    unexpected = len(getattr(res, "unexpected_keys", []))
    print(f"[CKPT] Loaded into DINO backbone from {ckpt_path} | missing={missing} unexpected={unexpected}")

def set_backbone_trainable(detector: FasterRCNN, train: bool):
    for p in detector.backbone.parameters():
        p.requires_grad = train

def print_sanity(detector: FasterRCNN, device: torch.device, image_size: int):
    total = sum(p.numel() for p in detector.parameters())
    trainable = sum(p.numel() for p in detector.parameters() if p.requires_grad)
    bb_trainable = sum(p.numel() for p in detector.backbone.parameters() if p.requires_grad)
    print(f"[SANITY] params total={total:,} trainable={trainable:,} backbone_trainable={bb_trainable:,}")
    detector.eval()
    with torch.no_grad():
        x = torch.zeros(3, image_size, image_size, device=device)
        out = detector([x])[0]
        print(f"[SANITY] pred shapes boxes={tuple(out['boxes'].shape)} scores={tuple(out['scores'].shape)} labels={tuple(out['labels'].shape)}")
def load_dino_weights(model: nn.Module, ckpt_path: str) -> bool:
    if not os.path.exists(ckpt_path):
        print(f"[WARN] DINO weights file not found: {ckpt_path}")
        return False

    sd = torch.load(ckpt_path, map_location="cpu")

    # Normalize to a bare state_dict
    if isinstance(sd, dict):
        if "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        elif "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]

    if not isinstance(sd, dict):
        print(f"[ERR] Unexpected checkpoint type: {type(sd)}")
        return False

    # Strip common prefixes
    cleaned = {}
    for k, v in sd.items():
        k = k.removeprefix("module.")
        if k.startswith(("backbone.", "encoder.")):
            k = k.split(".", 1)[1]
        cleaned[k] = v

    # Stats before load
    model_keys = set(model.state_dict().keys())
    ckpt_keys  = set(cleaned.keys())
    intersect  = model_keys & ckpt_keys

    res = model.load_state_dict(cleaned, strict=False)
    missing = list(getattr(res, "missing_keys", []))
    unexpected = list(getattr(res, "unexpected_keys", []))

    print(f"[CKPT] intersect={len(intersect)}  missing={len(missing)}  unexpected={len(unexpected)}")

    # Heuristic success: at least, say, 80% of model params matched
    match_ratio = len(intersect) / max(1, len(model_keys))
    ok = match_ratio >= 0.8

    if not ok:
        print("[WARN] Low match ratio; weights may not correspond to this architecture.")
    if missing:
        print(f"[MISS] e.g. {missing[:8]}")
    if unexpected:
        print(f"[UNEX] e.g. {unexpected[:8]}")

    return ok
# =========================
# Main
# =========================
def main():
    print(f"DINOv3 location set to {DINOV3_LOCATION}")
    # robust hub load with fallback to manual state dict
    
    dino_model = torch.hub.load(
            repo_or_dir=DINOV3_LOCATION,
            model=DINO_MODEL_NAME,
            source="local",
            weights=DINO_WEIGHTS,
            skip_validation=True,
        )

    # if os.path.exists(DINO_WEIGHTS):
    #     sd = torch.load(DINO_WEIGHTS, map_location="cpu")
    #     dino_model.load_state_dict(sd)
    # else:
    #     print(f"[WARN] DINO weights file not found: {DINO_WEIGHTS}")
    
    ok = load_dino_weights(dino_model, DINO_WEIGHTS)
    if not ok:
        print("[FAIL] DINO weights not properly loaded; check checkpoint path/format and model name.")

    if USE_FINETUNE_CKPT:
        load_into_dino_backbone(dino_model, FINETUNE_CKPT)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = create_model(dino_model, num_classes=NUM_CLASSES, image_size=IMAGE_SIZE).to(device)
    print("model footprint:", model)
    set_backbone_trainable(model, TRAIN_BACKBONE)
    print(f"[CFG] TRAIN_BACKBONE={TRAIN_BACKBONE}  USE_FINETUNE_CKPT={USE_FINETUNE_CKPT}")
    print_sanity(model, device, IMAGE_SIZE)
    ds_train = BrickKilnDataset(root=PSUDO_ROOT, split="thresh_0.8", input_size=IMAGE_SIZE)
    ds_val   = BrickKilnDataset(root=DOTA_ROOT, split="val",   input_size=IMAGE_SIZE)
    ds_test  = BrickKilnDataset(root=XVIEW_ROOT, split="test",  input_size=IMAGE_SIZE)
    if len(ds_train) == 0:
        raise RuntimeError("Empty training dataset.")
    if len(ds_val) == 0:
        raise RuntimeError("Empty validation dataset.")
    if len(ds_test) == 0:
        raise RuntimeError("Empty test dataset.")
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    dl_val   = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)

    # optimizer = build_optimizer(model)
    # scheduler = build_scheduler(optimizer, steps_per_epoch=max(len(dl_train), 1))

    # best_map = -1.0
    # for epoch in range(NUM_EPOCHS):
    #     train_loss = train_one_epoch(model, dl_train, device, optimizer,
    #                                  epoch=epoch, clip_norm=CLIP_NORM, print_every=PRINT_EVERY, scheduler=scheduler)
    #     print(f"Epoch {epoch+1}/{NUM_EPOCHS} train loss {train_loss:.4f}")
    #     map50 = validate(model, dl_val, device, epoch=epoch)
    #     map_test = validate(model, dl_test, device, epoch=epoch)
    #     # torch.save(model.state_dict(), EPOCH_CKPT.format(epoch+1))

    #     if map50 > best_map:
    #         best_map = map50
    #         torch.save(model.state_dict(), BEST_CKPT)
    #         print(f"Saved best checkpoint with mAP@50 {best_map:.4f} to {BEST_CKPT}")

    if os.path.exists(BEST_CKPT):
        state = torch.load(BEST_CKPT, map_location="cpu")
        model.load_state_dict(state, strict=False)
        model.to(device).eval()

    evaluate_region(
        model, root=DOTA_ROOT, split="test", device=device,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, image_size=IMAGE_SIZE,
        title="DOTA — IN-REGION (test)", results_csv=RESULTS_CSV,
    )

    evaluate_region(
        model, root=XVIEW_ROOT, split="test", device=device,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, image_size=IMAGE_SIZE,
        title="xView — OOR (test)", results_csv=RESULTS_CSV,
    )

    # evaluate_region(
    #     model, root=BD_ROOT, split="test", device=device,
    #     batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, image_size=IMAGE_SIZE,
    #     title="Bangladesh — OOR (test)", results_csv=RESULTS_CSV,
    # )

if __name__ == "__main__":
    main()
    
    
    # 