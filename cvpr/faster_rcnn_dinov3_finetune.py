#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import math
from pathlib import Path
from typing import List, Tuple, Iterable, Dict

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
FINETUNED_DINO_WEIGHTS = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/cvpr/dino_vitl16_metadata_contrastive_lr_epoch4.pth"

UP_ROOT  = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/uttar_pradesh"
BD_ROOT  = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/bangladesh"
PKP_ROOT = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/pak_punjab"

IMAGE_SIZE    = 800
BATCH_SIZE    = 16
NUM_WORKERS   = 16
NUM_EPOCHS    = 10
BACKBONE_LR   = 1e-5
HEAD_LR       = 1e-4
WEIGHT_DECAY  = 0.04
NUM_CLASSES   = 4  # background + 3 classes (1..3)

BEST_CKPT     = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/cvpr/best_pak_punjab_val_map50_dinov3-cl-meta.pth"
RESULTS_CSV   = "pak_punjab_region_eval_final-cl-meta.csv"

TRAIN_IN_REGION = True
USE_AMP = True
CLIP_NORM = 1.0

# =========================
# Dataset
# =========================
class BrickKilnDataset(Dataset):
    """
    Layout: <root>/<split>/{images,labels}
    YOLO OBB line: <cls> x1 y1 x2 y2 x3 y3 x4 y4 in [0,1]
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
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 9:
                        continue
                    cls_id = int(float(parts[0])) + 1  # shift to 1..3
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
    images, targets = list(zip(*batch))
    return list(images), list(targets)

# =========================
# DINOv3 backbone
# =========================
class DinoV3BackboneWrapper(nn.Module):
    """Return dict {'0': Tensor[B, C, H/16, W/16]} with out_channels=C."""
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

# =========================
# Optimizer and scheduler
# =========================
def split_backbone_head_params(model: FasterRCNN) -> Tuple[Iterable[nn.Parameter], Iterable[nn.Parameter]]:
    backbone_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone") or n.startswith("transform"):
            backbone_params.append(p)
        else:
            head_params.append(p)
    return backbone_params, head_params


def build_optimizer(model: FasterRCNN):
    bb_params, head_params = split_backbone_head_params(model)
    optim = torch.optim.AdamW(
        [
            {"params": bb_params, "lr": BACKBONE_LR},
            {"params": head_params, "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    return optim


def build_scheduler(optimizer):
    # cosine decay with linear warmup of one epoch
    def _make(steps_per_epoch: int):
        warmup = max(steps_per_epoch, 1)
        T = max(NUM_EPOCHS * steps_per_epoch, warmup + 1)

        def lr_lambda(step):
            if step < warmup:
                return float(step + 1) / float(warmup)
            t = (step - warmup) / max(T - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * t))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return _make

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
    for images, targets in tqdm(data_loader, desc=f"Validation epoch {epoch+1}"):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        preds   = model(images)
        preds   = [{k: v.detach().cpu() for k, v in p.items()} for p in preds]
        targets = [{k: v.detach().cpu() for k, v in t.items()} for t in targets]
        metric.update(preds, targets)
    res   = metric.compute()
    map50 = float(res.get("map_50", res.get("map", torch.tensor(0.0))))
    print(f"Validation Results mAP50 {map50:.4f}")
    return map50, map50

# =========================
# Training
# =========================
def train_one_epoch(model, loader, device, optimizer, scaler=None, epoch=0, clip_norm=None):
    model.train()
    running = 0.0
    pbar = tqdm(loader, desc=f"Train epoch {epoch+1}")
    for images, targets in pbar:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)

        if scaler is None:
            losses: Dict[str, torch.Tensor] = model(images, targets)
            loss = sum(v for v in losses.values())
            loss.backward()
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            loss_val = float(loss.detach().cpu())
        else:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                losses: Dict[str, torch.Tensor] = model(images, targets)
                loss = sum(v for v in losses.values())
            scaler.scale(loss).backward()
            if clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
            loss_val = float(loss.detach().cpu())

        running += loss_val
        pbar.set_postfix(loss=f"{loss_val:.4f}")
    return running / max(len(loader), 1)

# --------------------------------------
# Final evaluation
# --------------------------------------
CLASS_NAMES = {1: "CFCBK", 2: "FCBK", 3: "Zigzag"}

@torch.no_grad()
def evaluate_region(model, root: str, split: str, device,
                    batch_size=16, num_workers=8, image_size=800,
                    title="", results_csv=None, min_gt_warn: int = 5):
    ds = BrickKilnDataset(root=root, split=split, input_size=image_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True,
                    collate_fn=collate_fn)

    model.eval()
    metric_class = MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox", class_metrics=True, iou_thresholds=[0.5]
    )
    metric_agn = MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox", class_metrics=False, iou_thresholds=[0.5]
    )

    gt_support = {c: 0 for c in range(1, NUM_CLASSES)}

    for images, targets in tqdm(dl, desc=f"Test [{title or split}]"):
        images = [i.to(device) for i in images]
        preds  = model(images)

        preds_cpu = [{k: v.to("cpu") for k, v in p.items()} for p in preds]
        tgts_cpu  = [{k: v.to("cpu") for k, v in t.items()} for t in targets]

        for t in tgts_cpu:
            if t["labels"].numel() > 0:
                for lbl in t["labels"].tolist():
                    if 1 <= lbl < NUM_CLASSES:
                        gt_support[lbl] = gt_support.get(lbl, 0) + 1

        metric_class.update(preds_cpu, tgts_cpu)

        preds_agn = [
            {"boxes": p["boxes"], "scores": p["scores"], "labels": torch.ones_like(p["labels"])}
            for p in preds_cpu
        ]
        tgts_agn = [
            {"boxes": t["boxes"], "labels": torch.ones_like(t["labels"])}
            for t in tgts_cpu
        ]
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
        if np.isfinite(apf) and apf >= 0.0:
            valid_pairs.append((int(c), apf))

    per_cls = {c: ap * 100.0 for c, ap in valid_pairs}
    if len(valid_pairs) > 0:
        mc_map50 = sum(ap for _, ap in valid_pairs) / len(valid_pairs) * 100.0
    else:
        mc_map50 = 0.0

    zero_gt = [c for c in range(1, NUM_CLASSES) if gt_support.get(c, 0) == 0]
    low_sup = [c for c in range(1, NUM_CLASSES) if 0 < gt_support.get(c, 0) < min_gt_warn]

    def cname(c):
        return f"{CLASS_NAMES.get(c, f'c{c}')}"
    def g(k):
        return float(per_cls.get(k, 0.0))

    print("\n" + "=" * 96)
    print(f" Region: {title or (Path(root).name + ' — ' + split)}")
    print("=" * 96)
    print(f"{'CA mAP@50':<12}{'MC mAP@50':<12}")
    print("-" * 96)
    print(f"{ca_map50:<12.2f}{mc_map50:<12.2f}")

    print("\nGT support per class and AP@50:")
    for c in range(1, NUM_CLASSES):
        sup = gt_support.get(c, 0)
        apv = f"{g(c):.2f}" if c in per_cls else "—"
        print(f"  {cname(c):<8}  GT={sup:<5}  AP@50={apv}")

    if zero_gt:
        z = ", ".join([f"{cname(c)}(GT=0)" for c in zero_gt])
        print(f"\nExcluded from MC mAP@50 due to zero GT: {z}")
    if low_sup:
        l = ", ".join([f"{cname(c)}(GT={gt_support[c]})" for c in low_sup])
        print(f"Low-support classes (GT < {min_gt_warn}): {l}")

    print("=" * 96 + "\n")

    if results_csv is not None:
        is_new = not os.path.exists(results_csv)
        with open(results_csv, "a", newline="") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow([
                    "Region", "Split", "CA_mAP50", "MC_mAP50",
                    "CFCBK_mAP50", "FCBK_mAP50", "Zigzag_mAP50",
                    "GT_CFCBK", "GT_FCBK", "GT_Zigzag",
                    "Zero_GT", "Low_Support_Thresh"
                ])
            w.writerow([
                title or Path(root).name, split,
                f"{ca_map50:.2f}", f"{mc_map50:.2f}",
                f"{g(1):.2f}", f"{g(2):.2f}", f"{g(3):.2f}",
                gt_support.get(1, 0), gt_support.get(2, 0), gt_support.get(3, 0),
                "|".join(str(c) for c in zero_gt) if zero_gt else "",
                min_gt_warn
            ])

    return ca_map50, mc_map50, per_cls, gt_support, zero_gt, low_sup

# =========================
# Main
# =========================
def main():
    print(f"DINOv3 location {DINOV3_LOCATION}")

    # load model skeleton from local repo
    dino_model = torch.hub.load(
        repo_or_dir=DINOV3_LOCATION,
        model=DINO_MODEL_NAME,
        source="local",
        skip_validation=True,
        trust_repo=True,
    )

    # load base pretrain weights
    base = torch.load(DINO_WEIGHTS, map_location="cpu")
    base_state = base.get("state_dict", base.get("model", base))
    base_state = {k.replace("module.", ""): v for k, v in base_state.items()}
    miss, unexp = dino_model.load_state_dict(base_state, strict=False)
    print(f"[base load] missing={len(miss)} unexpected={len(unexp)}")

    # load finetuned overwrite
    ft = torch.load(FINETUNED_DINO_WEIGHTS, map_location="cpu")
    ft_state = ft.get("state_dict", ft.get("model", ft))
    ft_state = {k.replace("module.", ""): v for k, v in ft_state.items()}
    miss, unexp = dino_model.load_state_dict(ft_state, strict=False)
    print(f"[finetuned load] missing={len(miss)} unexpected={len(unexp)}")

    # quick sanity
    with torch.no_grad():
        x = torch.randn(1, 3, 800, 800)
        try:
            y = dino_model.get_intermediate_layers(x, n=1, return_class_token=False)
            _ = y[0].shape
            print("[backbone] dummy forward OK")
        except Exception as e:
            print("[backbone] dummy forward FAILED:", e)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(dino_model, num_classes=NUM_CLASSES, image_size=IMAGE_SIZE).to(device)

    if TRAIN_IN_REGION:
        ds_train = BrickKilnDataset(root=PKP_ROOT, split="train", input_size=IMAGE_SIZE)
        ds_val   = BrickKilnDataset(root=PKP_ROOT, split="val",   input_size=IMAGE_SIZE)
        dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
        dl_val   = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)

        optimizer = build_optimizer(model)
        make_sched = build_scheduler(optimizer)
        scheduler = make_sched(steps_per_epoch=max(len(dl_train), 1))

        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

        best_map = -1.0
        for epoch in range(NUM_EPOCHS):
            train_loss = train_one_epoch(model, dl_train, device, optimizer,
                                         scaler=scaler if USE_AMP else None,
                                         epoch=epoch, clip_norm=CLIP_NORM)
            scheduler.step()
            print(f"Epoch {epoch+1} train loss {train_loss:.4f}")

            _, map50 = validate(model, dl_val, device, epoch=epoch)
            if map50 > best_map:
                best_map = map50
                os.makedirs(os.path.dirname(BEST_CKPT), exist_ok=True)
                torch.save(model.state_dict(), BEST_CKPT)
                print(f"Saved best checkpoint with mAP50 {best_map:.4f} to {BEST_CKPT}")

        if os.path.exists(BEST_CKPT):
            state = torch.load(BEST_CKPT, map_location="cpu")
            model.load_state_dict(state, strict=False)
            model.to(device).eval()
    else:
        state = torch.load(BEST_CKPT, map_location="cpu")
        model.load_state_dict(state, strict=False)
        model.to(device).eval()

    evaluate_region(
        model,
        root=PKP_ROOT,
        split="test",
        device=device,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE,
        title="pak_punjab IN REGION test",
        results_csv=RESULTS_CSV,
    )

    evaluate_region(
        model,
        root=UP_ROOT,
        split="test",
        device=device,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE,
        title="Uttar Pradesh OOR test",
        results_csv=RESULTS_CSV,
    )

    evaluate_region(
        model,
        root=BD_ROOT,
        split="test",
        device=device,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE,
        title="Bangladesh OOR test",
        results_csv=RESULTS_CSV,
    )

if __name__ == "__main__":
    main()