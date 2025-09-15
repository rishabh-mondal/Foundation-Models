#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Faster R-CNN on AlphaEarth AEF (64 bands) or RGB tiles.

Variants:
  1) head_only       : simple 1x1 adapter head (C->out)
  2) thin_cnn        : small CNN over inputs
  3) resnet18_ae     : ResNet-18 (conv1 adapted to in_ch) -> C3 features
  4) resnet50_ae     : ResNet-50 + FPN (conv1 adapted if in_ch!=3)
  5) resnet50_ae_imnet: as above, ImageNet weights (except conv1 if in_ch!=3)

Notes:
  - AEF assumed in [-1, 1] (no normalization).
  - RGB default normalization: ImageNet mean/std (toggle with --rgb_norm none|imnet).
  - Sanity: --smoke_infer, --overfit_k
  - Evaluation: class-agnostic mAP@50 + per-class AP@50.
"""

import os, argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from PIL import Image

try:
    import rasterio as rio
except Exception:
    rio = None
try:
    import tifffile as tiff
except Exception:
    tiff = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from torchvision.models import resnet18, resnet50
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchmetrics.detection import MeanAveragePrecision

# -------------------------
# Config (edit paths)
# -------------------------
REGION_ROOTS: Dict[str, str] = {
    "uttar_pradesh": "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/uttar_pradesh",
    "bangladesh":    "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/bangladesh",
    "pak_punjab":    "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/pak_punjab",
}

IMG_SIZE     = 800
NUM_CLASSES  = 4  # background + 3 kiln classes (labels 1..3)

# -------------------------
# I/O helpers
# -------------------------
def load_tif_64(path: Path) -> np.ndarray:
    """Load AEF GeoTIFF as float32 [C,H,W] with C==64."""
    arr = None
    if rio is not None:
        try:
            with rio.open(path) as ds:
                arr = ds.read().astype(np.float32)   # [C,H,W]
        except Exception:
            arr = None
    if arr is None and tiff is not None:
        arr = tiff.imread(str(path)).astype(np.float32)  # [H,W,C] or [C,H,W]
        if arr.ndim == 3 and arr.shape[-1] == 64 and arr.shape[0] != 64:
            arr = np.moveaxis(arr, -1, 0)
    if arr is None or arr.ndim != 3 or arr.shape[0] != 64:
        raise RuntimeError(f"AEF GeoTIFF must be 64xHxW, got {None if arr is None else arr.shape}: {path}")
    return arr  # [64,H,W]

def load_rgb(path: Path) -> np.ndarray:
    """Load RGB image as float32 [3,H,W] in [0,1]."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0      # [H,W,3]
    arr = np.moveaxis(arr, -1, 0)                         # [3,H,W]
    return arr

def resize_bchw(x: torch.Tensor, size: int) -> torch.Tensor:
    """Bilinear resize [C,H,W] -> [C,size,size]."""
    return F.interpolate(x.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)

# -------------------------
# Dataset (AEF or RGB)
# -------------------------
class DetDataset(Dataset):
    """
    Detection dataset for AEF (64ch) or RGB (3ch).
      - AEF:   <root>/<split>/embeddings/*.tif
      - RGB:   <root>/<split>/images/*.(jpg|jpeg|png|tif)
      - Labels:<root>/<split>/labels/*.txt (YOLO-OBB → XYXY)
    """
    def __init__(self, root: str, split: str, image_size: int,
                 modality: str = "aef", nodata_value: Optional[float]=None):
        assert modality in ("aef", "rgb")
        self.root = Path(root)
        self.split = split
        self.size = int(image_size)
        self.modality = modality
        self.nodata = nodata_value

        if modality == "aef":
            self.img_dir = self.root / split / "embeddings"
            exts = ["*.tif"]
        else:
            self.img_dir = self.root / split / "images"
            exts = ["*.jpg", "*.jpeg", "*.png", "*.tif"]

        self.lab_dir = self.root / split / "labels"
        assert self.img_dir.is_dir(), f"Missing {self.img_dir}"
        assert self.lab_dir.is_dir(), f"Missing {self.lab_dir}"

        files = []
        for pat in exts:
            files.extend(self.img_dir.glob(pat))
        self.files = sorted(files)
        assert len(self.files) > 0, f"No images in {self.img_dir}"

    def __len__(self): return len(self.files)

    @staticmethod
    def _yolo_obb_to_xyxy(txt_path: Path, W: int, H: int):
        boxes, labels = [], []
        if not txt_path.exists(): return boxes, labels
        with open(txt_path, "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) != 9: continue
                cls_id = int(float(p[0])) + 1
                obb = np.array([float(x) for x in p[1:]], dtype=np.float32)
                xs = obb[0::2] * W; ys = obb[1::2] * H
                xmin, ymin = float(xs.min()), float(ys.min())
                xmax, ymax = float(xs.max()), float(ys.max())
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin,ymin,xmax,ymax])
                    labels.append(cls_id)
        return boxes, labels

    def __getitem__(self, idx: int):
        path = self.files[idx]
        stem = path.stem
        lbl_path = self.lab_dir / f"{stem}.txt"

        if self.modality == "aef":
            arr = load_tif_64(path)                      # [64,H,W]
            arr = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
            if self.nodata is not None:
                arr[arr == self.nodata] = 0.0
            x = torch.from_numpy(arr)
            x = resize_bchw(x, self.size).clamp_(-1.0, 1.0)
        else:
            x = torch.from_numpy(load_rgb(path))         # [3,H,W] in [0,1]
            x = resize_bchw(x, self.size).clamp_(0.0, 1.0)

        _, Ht, Wt = x.shape
        boxes, labels = self._yolo_obb_to_xyxy(lbl_path, Wt, Ht)
        target = {
            "boxes":  torch.tensor(boxes, dtype=torch.float32).reshape(-1,4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return x, target

def collate_fn(batch):
    imgs, tgts = zip(*batch)
    return list(imgs), list(tgts)

# -------------------------
# Backbones (AEF/RGB → features)
# -------------------------
class IdentityAdapter(nn.Module):
    def __init__(self, in_ch=64, out_ch=64):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.out_channels = out_ch
    def forward(self, x): return {"0": self.conv(x)}

class ThinCNN(nn.Module):
    def __init__(self, in_ch=64, mid=128, out=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(True),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),   nn.BatchNorm2d(mid), nn.ReLU(True),
            nn.Conv2d(mid, out, 3, padding=1, bias=False),   nn.BatchNorm2d(out), nn.ReLU(True),
        )
        self.out_channels = out
    def forward(self, x): return {"0": self.net(x)}

class ResNet18_Mod(nn.Module):
    """ResNet18 modified to accept in_ch, returns C3 features."""
    def __init__(self, in_ch=64):
        super().__init__()
        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.body = nn.Sequential(
            m.conv1, m.bn1, nn.ReLU(inplace=True),
            m.maxpool,
            m.layer1,  # C=64
            m.layer2,  # C=128
            m.layer3,  # C=256
        )
        self.out_channels = 256
    def forward(self, x): return {"0": self.body(x)}

class ResNet50_AE_FPN(nn.Module):
    """ResNet-50 + FPN. Adapts conv1 if in_ch != 3. No pretrained weights."""
    def __init__(self, in_ch=64, out_channels=256, trainable_layers=5):
        super().__init__()
        m = resnet50(weights=None)
        if in_ch != 3:
            m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Freeze early layers if desired (0..5)
        layers_to_freeze = max(0, min(5, 5 - trainable_layers))
        if layers_to_freeze > 0:
            freeze = [m.conv1, m.bn1, m.layer1, m.layer2, m.layer3][:layers_to_freeze]
            for layer in freeze:
                for p in layer.parameters():
                    p.requires_grad = False

        return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        in_channels_list = [256, 512, 1024, 2048]
        self.backbone = BackboneWithFPN(
            m, return_layers, in_channels_list, out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = out_channels
        self.is_fpn = True
    def forward(self, x): return self.backbone(x)

class ResNet50_AE_FPN_ImNet(nn.Module):
    """ResNet-50 + FPN with ImageNet weights; conv1 adapted if in_ch != 3."""
    def __init__(self, in_ch=64, out_channels=256, trainable_layers=5):
        super().__init__()
        m = resnet50(weights=ResNet50_Weights.DEFAULT)
        if in_ch != 3:
            old = m.conv1  # [64,3,7,7]
            new = nn.Conv2d(in_ch, old.out_channels, old.kernel_size,
                            stride=old.stride, padding=old.padding, bias=False)
            with torch.no_grad():
                w = old.weight.mean(dim=1, keepdim=True)   # [64,1,7,7]
                new.weight.copy_(w.repeat(1, in_ch, 1, 1))
            m.conv1 = new

        layers_to_freeze = max(0, min(5, 5 - trainable_layers))
        if layers_to_freeze > 0:
            freeze = [m.conv1, m.bn1, m.layer1, m.layer2, m.layer3][:layers_to_freeze]
            for layer in freeze:
                for p in layer.parameters():
                    p.requires_grad = False

        return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        in_channels_list = [256, 512, 1024, 2048]
        self.backbone = BackboneWithFPN(
            m, return_layers, in_channels_list, out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = out_channels
        self.is_fpn = True
    def forward(self, x): return self.backbone(x)

# -------------------------
# Anchors (kiln-friendly)
# -------------------------
def make_kiln_anchors(backbone, image_size: int):
    """
    AnchorGenerator tuned for brick kilns.
    One size per FPN level; wider ARs to catch elongated shapes.
    """
    AR_STANDARD = (0.25, 0.5, 1.0, 2.0, 4.0)
    AR_WIDE     = (0.2, 0.33, 0.5, 1.0, 2.0, 3.0, 5.0)

    use_fpn = getattr(backbone, "is_fpn", False)

    if image_size <= 256:
        if use_fpn:
            sizes = ((4,), (8,), (16,), (32,), (64,))
            aspect_ratios = (AR_WIDE,) * len(sizes)
        else:
            sizes = ((4, 6, 8, 12, 16, 24, 32, 48, 64),)
            aspect_ratios = (AR_WIDE,)
    else:
        if use_fpn:
            sizes = ((16,), (32,), (64,), (128,), (256,))  # add (512,) if needed
            aspect_ratios = (AR_STANDARD,) * len(sizes)
        else:
            sizes = ((12, 16, 24, 32, 48, 64, 96, 128, 192, 256),)
            aspect_ratios = (AR_STANDARD,)

    return AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)

# -------------------------
# Build detector
# -------------------------
def build_model(variant: str, num_classes: int, image_size: int,
                modality: str = "aef", rgb_norm: str = "imnet") -> FasterRCNN:
    assert modality in ("aef", "rgb")
    assert rgb_norm in ("none", "imnet")
    in_ch = 3 if modality == "rgb" else 64

    # backbones
    if variant == "head_only":
        backbone = IdentityAdapter(in_ch=in_ch, out_ch=128)
    elif variant == "thin_cnn":
        backbone = ThinCNN(in_ch=in_ch, mid=128, out=256)
    elif variant == "resnet18_ae":
        backbone = ResNet18_Mod(in_ch=in_ch)
    elif variant == "resnet50_ae":
        backbone = ResNet50_AE_FPN(in_ch=in_ch, out_channels=256, trainable_layers=5)
    elif variant == "resnet50_ae_imnet":
        backbone = ResNet50_AE_FPN_ImNet(in_ch=in_ch, out_channels=256, trainable_layers=5)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # anchors
    anchor_generator = make_kiln_anchors(backbone, image_size)

    # image normalization
    if modality == "aef":
        image_mean = [0.0] * in_ch
        image_std  = [1.0] * in_ch
    else:  # rgb
        if rgb_norm == "imnet":
            image_mean = [0.485, 0.456, 0.406]
            image_std  = [0.229, 0.224, 0.225]
        else:  # none
            image_mean = [0.0, 0.0, 0.0]
            image_std  = [1.0, 1.0, 1.0]

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        min_size=image_size,
        max_size=image_size,
        image_mean=image_mean,
        image_std=image_std,
    )
    return model

# -------------------------
# Train / Validate
# -------------------------
def train_one_epoch(model, opt, loader, device, epoch):
    model.train()
    total, steps = 0.0, 0
    pbar = tqdm(loader, desc=f"Train ep{epoch+1}")
    for imgs, tgts in pbar:
        imgs  = [x.to(device) for x in imgs]
        tgts  = [{k:v.to(device) for k,v in t.items()} for t in tgts]
        loss_dict = model(imgs, tgts)
        loss = sum(loss_dict.values())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += float(loss); steps += 1
        pbar.set_postfix(loss=f"{float(loss):.4f}")
    return total / max(1, steps)

@torch.no_grad()
def validate(model, loader, device, epoch=0):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", iou_thresholds=[0.5], class_metrics=False)
    for imgs, tgts in tqdm(loader, desc=f"Val ep{epoch+1}"):
        imgs = [x.to(device) for x in imgs]
        preds = model(imgs)
        preds = [{k:v.detach().cpu() for k,v in p.items()} for p in preds]
        tgts  = [{k:v.detach().cpu() for k,v in t.items()} for t in tgts]
        metric.update(preds, tgts)
    res = metric.compute()
    return float(res.get("map_50", torch.tensor(0.0)))

# -------------------------
# Final Evaluation (ID/OOD)
# -------------------------
@torch.no_grad()
def evaluate_region(model, root: str, split: str, device, image_size: int,
                    modality: str, batch_size=8, num_workers=8, title=None, nodata_value=None):
    ds = DetDataset(root, split, image_size, modality=modality, nodata_value=nodata_value)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=True, collate_fn=collate_fn)

    metric_c = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=True,  iou_thresholds=[0.5])
    metric_a = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=False, iou_thresholds=[0.5])

    for imgs, tgts in tqdm(dl, desc=f"Test [{title or split}]"):
        imgs = [x.to(device) for x in imgs]
        preds = model(imgs)
        preds_cpu = [{k:v.to('cpu') for k,v in p.items()} for p in preds]
        tgts_cpu  = [{k:v.to('cpu') for k,v in t.items()} for t in tgts]

        metric_c.update(preds_cpu, tgts_cpu)

        preds_agn = [{'boxes': p['boxes'], 'scores': p['scores'],
                      'labels': torch.ones_like(p['labels'])} for p in preds_cpu]
        tgts_agn  = [{'boxes': t['boxes'],
                      'labels': torch.ones_like(t['labels'])} for t in tgts_cpu]
        metric_a.update(preds_agn, tgts_agn)

    rc = metric_c.compute(); ra = metric_a.compute()
    ca50 = float(ra.get('map_50', torch.tensor(0.0))) * 100.0
    classes = rc.get('classes', torch.tensor([])).tolist() if 'classes' in rc else []
    ap_pc   = rc.get('map_per_class', torch.tensor([])).tolist() if 'map_per_class' in rc else []
    per_cls = {int(c): float(ap)*100.0 for c, ap in zip(classes, ap_pc)
               if np.isfinite(float(ap)) and float(ap) >= 0}
    mc50 = (sum(per_cls.values()) / max(1,len(per_cls))) if per_cls else 0.0

    def g(k): return float(per_cls.get(k, 0.0))
    print("\n" + "="*84)
    print(f" Region: {title or (Path(root).name + ' — ' + split)}")
    print("="*84)
    print(f"{'CA mAP@50':<12}{'MC mAP@50':<12}{'CFCBK@50':<12}{'FCBK@50':<12}{'Zigzag@50':<12}")
    print("-"*84)
    print(f"{ca50:<12.2f}{mc50:<12.2f}{g(1):<12.2f}{g(2):<12.2f}{g(3):<12.2f}")
    print("="*84 + "\n")
    return ca50, mc50, per_cls

# -------------------------
# Smoke / Overfit
# -------------------------
@torch.no_grad()
def smoke_infer(model, ds: Dataset, device, batches=2, bs=2):
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0, collate_fn=collate_fn)
    it = iter(dl)
    for _ in range(batches):
        imgs, tgts = next(it)
        imgs = [x.to(device) for x in imgs]
        _ = model(imgs)
    print("[SMOKE] inference passed.")

def overfit_small(model, ds: Dataset, device, k=8, epochs=5, lr=1e-4):
    idx = torch.randperm(len(ds))[:k].tolist()
    small = Subset(ds, idx)
    dl = DataLoader(small, batch_size=min(2,k), shuffle=True, num_workers=0, collate_fn=collate_fn)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    print(f"[OVERFIT] {k} samples for {epochs} epochs")
    for ep in range(epochs):
        model.train()
        losses = []
        for imgs, tgts in dl:
            imgs = [x.to(device) for x in imgs]
            tgts = [{k:v.to(device) for k,v in t.items()} for t in tgts]
            ld = model(imgs, tgts); loss = sum(ld.values())
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
        print(f"[OVERFIT] ep{ep+1}: loss={np.mean(losses):.4f}")
    print("[OVERFIT] done.")

# -------------------------
# CLI / Main
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser("Faster R-CNN on AEF (64ch) or RGB tiles")
    ap.add_argument("--variant",
                    choices=["head_only","thin_cnn","resnet18_ae","resnet50_ae","resnet50_ae_imnet"],
                    default="thin_cnn")
    ap.add_argument("--modality", choices=["aef","rgb"], default="aef",
                    help="Use 64-ch AEF embeddings or RGB images.")
    ap.add_argument("--rgb_norm", choices=["none","imnet"], default="imnet",
                    help="RGB normalization: none or ImageNet mean/std.")

    ap.add_argument("--train_region", default="uttar_pradesh", choices=list(REGION_ROOTS.keys()))
    ap.add_argument("--in_region",    default="uttar_pradesh", choices=list(REGION_ROOTS.keys()))
    ap.add_argument("--oor_regions",  nargs="*", default=["pak_punjab","bangladesh"])
    ap.add_argument("--train_split",  default="train")
    ap.add_argument("--val_split",    default="val")
    ap.add_argument("--test_split",   default="test")
    ap.add_argument("--image_size",   type=int, default=IMG_SIZE)
    ap.add_argument("--batch_size",   type=int, default=8)
    ap.add_argument("--num_workers",  type=int, default=8)
    ap.add_argument("--epochs",       type=int, default=6)
    ap.add_argument("--backbone_lr",  type=float, default=1e-4)
    ap.add_argument("--head_lr",      type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--save_dir",     default="aef_rgb_runs")
    ap.add_argument("--ckpt",         default="")
    ap.add_argument("--nodata_value", type=float, default=None)
    # sanity
    ap.add_argument("--smoke_infer", action="store_true")
    ap.add_argument("--overfit_k", type=int, default=0)
    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # datasets
    train_ds = DetDataset(REGION_ROOTS[args.train_region], args.train_split,
                          args.image_size, modality=args.modality, nodata_value=args.nodata_value)
    val_ds   = DetDataset(REGION_ROOTS[args.train_region], args.val_split,
                          args.image_size, modality=args.modality, nodata_value=args.nodata_value)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    # model
    model = build_model(args.variant, NUM_CLASSES, args.image_size,
                        modality=args.modality, rgb_norm=args.rgb_norm).to(device)

    # optional sanity passes
    if args.smoke_infer:
        smoke_infer(model, val_ds, device, batches=2, bs=min(2, args.batch_size))

    if args.overfit_k > 0:
        overfit_small(model, train_ds, device, k=args.overfit_k, epochs=5, lr=1e-3)

    # optim (different LRs for backbone/head)
    back_params, head_params = [], []
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        if n.startswith("backbone."):
            back_params.append(p)
        else:
            head_params.append(p)
    opt = torch.optim.AdamW(
        [{"params": back_params, "lr": args.backbone_lr},
         {"params": head_params, "lr": args.head_lr}],
        weight_decay=args.weight_decay,
    )

    # train loop
    best = -1.0
    ckpt_path = Path(args.save_dir) / f"best_{args.variant}_{args.modality}_{args.train_region}.pth"
    if args.ckpt and Path(args.ckpt).exists():
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=False)
        print(f"[INIT] loaded weights from {args.ckpt}")

    for ep in range(args.epochs):
        tl = train_one_epoch(model, opt, train_dl, device, ep)
        mv = validate(model, val_dl, device, ep)
        print(f"[E{ep+1:02d}] train_loss={tl:.4f}  val_mAP50={mv:.4f}")
        if mv > best:
            best = mv
            torch.save(model.state_dict(), ckpt_path)
            print(f"[CKPT] saved -> {ckpt_path}")

    # eval best
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(device).eval()

    print("\n[ID] In-Region evaluation")
    evaluate_region(model, REGION_ROOTS[args.in_region], args.test_split,
                    device, args.image_size, modality=args.modality,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                    title=f"{args.in_region} (IN-REGION)", nodata_value=args.nodata_value)

    for r in args.oor_regions:
        print(f"\n[OOD] Out-of-Region: {r}")
        evaluate_region(model, REGION_ROOTS[r], args.test_split,
                        device, args.image_size, modality=args.modality,
                        batch_size=args.batch_size, num_workers=args.num_workers,
                        title=f"{r} (OOD)", nodata_value=args.nodata_value)

if __name__ == "__main__":
    main()



# CUDA_VISIBLE_DEVICES=2 nohup python -u aef_rgb_frcnn.py \
#   --variant resnet50_ae_imnet --modality aef --rgb_norm imnet \
#   --train_region uttar_pradesh --in_region uttar_pradesh --oor_regions pak_punjab bangladesh \
#   --image_size 800 --batch_size 8 --epochs 6 \
#   --save_dir runs/aef_resnet50_imnet_s800_e6_b8 \
#   > logs/aef_resnet50_imnet_s800_e6_b8.log 2>&1 &    


# CUDA_VISIBLE_DEVICES=3 nohup python -u aef_rgb_frcnn.py \
#   --variant thin_cnn --modality rgb --rgb_norm imnet \
#   --train_region uttar_pradesh --in_region uttar_pradesh --oor_regions pak_punjab bangladesh \
#   --image_size 800 --batch_size 8 --epochs 10 \
#   --save_dir runs/rgb_thincnn_s128_e10_b16 \
#   > logs/rgb_thincnn_s128_e10_b16.log 2>&1 &


# CUDA_VISIBLE_DEVICES=1 nohup python -u aef_rgb_frcnn.py \
#   --variant resnet50_ae_imnet --modality rgb --rgb_norm imnet \
#   --train_region uttar_pradesh --in_region uttar_pradesh --oor_regions pak_punjab bangladesh \
#   --image_size 800 --batch_size 8 --epochs 6 \
#   --save_dir runs/rgb_resnet50_imnet_s800_e6_b8 \
#   > logs/rgb_resnet50_imnet_s800_e6_b8.log 2>&1 &