#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
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

UP_ROOT  = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/uttar_pradesh"
BD_ROOT  = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/bangladesh"
PKP_ROOT = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/pak_punjab"

IMAGE_SIZE    = 800
BATCH_SIZE    = 8
NUM_WORKERS   = 8
NUM_EPOCHS    = 10
BACKBONE_LR   = 1e-5
HEAD_LR       = 1e-4
WEIGHT_DECAY  = 0.04
NUM_CLASSES   = 4  # background + 3 classes (labels 1..3)

BEST_CKPT     = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/checkpoints/best_pak_punjab_val_map50_dinov3.pth"
RESULTS_CSV   = "pak_punjab_region_eval_final.csv"

# =========================
# Dataset
# =========================
class BrickKilnDataset(Dataset):
    """
    Folder layout: <root>/<split>/{images,labels}
    YOLO-OBB line: <cls> x1 y1 x2 y2 x3 y3 x4 y4  (all in [0,1])
    Converted to axis-aligned XYXY for Faster R-CNN.
    """
    IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

    def __init__(self, root: str, split: str, input_size: int = 224):
        self.root = Path(root)
        self.split = split
        cand = self.root if (self.root / "images").is_dir() else (self.root / split)
        self.img_dir = cand / "images"
        self.label_dir = cand / "labels"
        assert self.img_dir.is_dir(), f"Missing images directory: {self.img_dir}"
        assert self.label_dir.is_dir(), f"Missing labels directory: {self.label_dir}"

        self.input_size = int(input_size)
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size),
                              interpolation=transforms.InterpolationMode.BILINEAR,
                              antialias=True),
            transforms.ToTensor(),
        ])

        # Include ALL images (even those with no GT) so FPs are penalized.
        self.img_files: List[str] = sorted(
            [f for f in os.listdir(self.img_dir) if Path(f).suffix.lower() in self.IMG_EXTS]
        )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx: int):
        img_name = self.img_files[idx]
        img_path = self.img_dir / img_name
        label_path = self.label_dir / f"{Path(img_name).stem}.txt"

        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        _, Ht, Wt = img_tensor.shape

        boxes, labels = [], []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 9:
                        continue
                    cls_id = int(float(parts[0])) + 1  # shift to 1..3 (0 is background)
                    obb = np.array([float(p) for p in parts[1:]], dtype=np.float32)
                    xs = obb[0::2] * Wt
                    ys = obb[1::2] * Ht
                    xmin, ymin = float(np.min(xs)), float(np.min(ys))
                    xmax, ymax = float(np.max(xs)), float(np.max(ys))
                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(cls_id)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return img_tensor, target


def collate_fn(batch):
    # Keep images even with zero GT boxes (important for correct FP accounting).
    images, targets = list(zip(*batch))
    return list(images), list(targets)

# =========================
# DINOv3 Backbone Wrapper
# =========================
class DinoV3BackboneWrapper(nn.Module):
    """Return {'0': Tensor[B, C, H/16, W/16]} with out_channels=C."""
    def __init__(self, dino_model: nn.Module, patch_stride: int = 16):
        super().__init__()
        self.dino = dino_model
        self.patch_stride = patch_stride
        C = getattr(dino_model, "embed_dim", None) or getattr(dino_model, "num_features", None)
        if C is None:
            with torch.no_grad():
                x = torch.zeros(1, 3, 32, 32)
                tokens, Ht, Wt = self._get_patch_tokens(x)
                C = tokens.shape[-1]
        self.out_channels = C

    @torch.no_grad()
    def _maybe_h_w(self, x):
        _, _, H, W = x.shape
        return math.ceil(H / self.patch_stride), math.ceil(W / self.patch_stride)

    def _get_patch_tokens(self, x):
        try:
            out = self.dino.forward_features(x)
            if isinstance(out, dict):
                if "x_norm_patchtokens" in out:
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
        B, N, C = tokens.shape
        feat = tokens.transpose(1, 2).contiguous().view(B, C, Ht, Wt)
        return {"0": feat}

def create_model(dino_model: nn.Module, num_classes: int, image_size: int = 800) -> FasterRCNN:
    backbone = DinoV3BackboneWrapper(dino_model, patch_stride=16)
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

# ---------------------------
# Validation during training (mAP@50)
# ---------------------------
@torch.no_grad()
def validate(model, data_loader, device, epoch=0):
    """
    Computes mAP@50 aggregated over classes (no classwise breakdown).
    """
    model.eval()
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=[0.5],  # strict @50
        class_metrics=False
    )

    for batch in tqdm(data_loader, desc=f"Validation epoch {epoch+1}"):
        if batch is None:
            continue
        images, targets = batch
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        preds   = model(images)

        preds   = [{k: v.detach().cpu() for k, v in p.items()} for p in preds]
        targets = [{k: v.detach().cpu() for k, v in t.items()} for t in targets]
        metric.update(preds, targets)

    res   = metric.compute()
    map50 = float(res.get("map_50", res.get("map", torch.tensor(0.0)) ))  # be explicit
    print(f"Validation Results - mAP@50: {map50:.4f}")
    # Keep signature (map_all, map_50); both are @50 here
    return map50, map50

# --------------------------------------
# Final evaluation (IN-REGION / OOR)
# --------------------------------------
@torch.no_grad()
def evaluate_region(model, root: str, split: str, device,
                    batch_size=16, num_workers=8, image_size=800,
                    title="", results_csv=None):
    """
    Computes:
      - CA mAP@50 (class-agnostic; collapse labels to 1)
      - MC mAP@50 (macro over classes WITH GT only)
      - Per-class AP@50 only for classes WITH GT (no -1.0 surprises).
    """
    ds = BrickKilnDataset(root=root, split=split, input_size=image_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True,
                    collate_fn=collate_fn)

    model.eval()
    metric_class = MeanAveragePrecision(
        box_format='xyxy', iou_type='bbox', class_metrics=True,  iou_thresholds=[0.5]
    )
    metric_agn   = MeanAveragePrecision(
        box_format='xyxy', iou_type='bbox', class_metrics=False, iou_thresholds=[0.5]
    )

    for batch in tqdm(dl, desc=f"Test [{title or split}]"):
        if batch is None:
            continue
        images, targets = batch
        images = [i.to(device) for i in images]
        preds  = model(images)

        preds_cpu = [{k: v.to('cpu') for k, v in p.items()} for p in preds]
        tgts_cpu  = [{k: v.to('cpu') for k, v in t.items()} for t in targets]

        # class-wise
        metric_class.update(preds_cpu, tgts_cpu)

        # class-agnostic (collapse labels to 1)
        preds_agn = [{'boxes': p['boxes'], 'scores': p['scores'],
                      'labels': torch.ones_like(p['labels'])} for p in preds_cpu]
        tgts_agn  = [{'boxes': t['boxes'],
                      'labels': torch.ones_like(t['labels'])} for t in tgts_cpu]
        metric_agn.update(preds_agn, tgts_agn)

    # ---- Compute ----
    res_class = metric_class.compute()
    res_agn   = metric_agn.compute()

    # CA mAP@50 (explicit)
    ca_map50 = float(res_agn.get('map_50', res_agn.get('map', torch.tensor(0.0)))) * 100.0

    # Per-class AP@50 list and classes
    classes = res_class.get('classes', torch.tensor([])).tolist() if 'classes' in res_class else []
    ap_list = res_class.get('map_per_class', torch.tensor([])).tolist() if 'map_per_class' in res_class else []

    # Filter out undefined classes (torchmetrics uses -1.0 or NaN when no GT)
    valid_pairs = []
    for c, ap in zip(classes, ap_list):
        try:
            apf = float(ap)
        except Exception:
            continue
        if apf >= 0.0 and np.isfinite(apf):
            valid_pairs.append((int(c), apf))

    per_cls = {c: ap * 100.0 for c, ap in valid_pairs}
    if len(valid_pairs) > 0:
        mc_map50 = sum(ap for _, ap in valid_pairs) / len(valid_pairs) * 100.0
    else:
        mc_map50 = 0.0

    # Pretty print (class ids 1,2,3 mapped to CFCBK/FCBK/Zigzag)
    def g(k):  # safe getter in %
        return float(per_cls.get(k, 0.0))

    print("\n" + "=" * 84)
    print(f" Region: {title or (Path(root).name + ' — ' + split)}")
    print("=" * 84)
    print(f"{'CA mAP@50':<12}{'MC mAP@50':<12}{'CFCBK@50':<12}{'FCBK@50':<12}{'Zigzag@50':<12}")
    print("-" * 84)
    print(f"{ca_map50:<12.2f}{mc_map50:<12.2f}{g(1):<12.2f}{g(2):<12.2f}{g(3):<12.2f}")
    print("=" * 84 + "\n")

    # Optional: write one line per region to CSV
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

# =========================
# Main
# =========================
def main():
    print(f"DINOv3 location set to {DINOV3_LOCATION}")
    dino_model = torch.hub.load(
        repo_or_dir=DINOV3_LOCATION,
        model=DINO_MODEL_NAME,
        source="local",
        weights=DINO_WEIGHTS,
        skip_validation=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(dino_model, num_classes=NUM_CLASSES, image_size=IMAGE_SIZE).to(device)

    # If you want to train here, uncomment your training block.
    # For now we load the best ckpt and run evaluations:
    state = torch.load(BEST_CKPT, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    # IN-REGION (PKP)
    evaluate_region(
        model,
        root=PKP_ROOT,
        split="test",
        device=device,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE,
        title="pak_punjab — IN-REGION (test)",
        results_csv=RESULTS_CSV,
    )

    # OOR: UP
    evaluate_region(
        model,
        root=UP_ROOT,
        split="test",
        device=device,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE,
        title="Uttar_pradesh — OOR (test)",
        results_csv=RESULTS_CSV,
    )

    # OOR: BD
    evaluate_region(
        model,
        root=BD_ROOT,
        split="test",
        device=device,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE,
        title="Bangladesh — OOR (test)",
        results_csv=RESULTS_CSV,
    )

if __name__ == "__main__":
    main()