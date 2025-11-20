#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RGB Pseudo-Label Generation with standard DINOv3 + Faster R-CNN pipeline.

- Loads DINOv3 backbone (local repo/weights) + your trained FasterRCNN checkpoint.
- Runs on target regions' train split RGB tiles.
- Saves:
    <root>/<split>/pseudo_rgb_json/*.json     (bbox, score, cls)
    <root>/<split>/pseudo_rgb_labels/*.txt    (YOLO: cls x_c y_c w h, normalized to IMAGE_SIZE)

EDIT the paths below if needed.
"""

import os, math, json
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import nms
from PIL import Image

# =========================
# Configuration (EDIT)
# =========================
DINOV3_GITHUB_LOCATION = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/dinov3"
DINOV3_LOCATION = os.getenv("DINOV3_LOCATION") or DINOV3_GITHUB_LOCATION
DINO_MODEL_NAME = "dinov3_vitl16"
DINO_WEIGHTS = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/dinov3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"

REGION_ROOTS: Dict[str, str] = {
    "bangladesh": "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/bangladesh",
    "pak_punjab": "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/pak_punjab",
}
SPLIT = "train"  # pseudo-label the training split of targets

# Teacher checkpoint (trained on source region, RGB)
RGB_TEACHER_CKPT = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/checkpoints/best_up_val_map50_dinov3.pth"

IMAGE_SIZE    = 800
BATCH_SIZE    = 8
NUM_WORKERS   = 8
NUM_CLASSES   = 4   # background + 3 classes (labels 1..3 in detector outputs)

SCORE_THRESH  = 0.25   # filter low-confidence preds
NMS_IOU       = 0.50

# =========================
# Unlabeled RGB dataset
# =========================
class UnlabeledRGBDataset(Dataset):
    IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    def __init__(self, root: str, split: str, image_size: int = 800):
        base = Path(root) / split / "images"
        assert base.is_dir(), f"Images folder not found: {base}"
        self.paths = sorted([p for p in base.iterdir() if p.suffix.lower() in self.IMG_EXTS])
        assert self.paths, f"No images in {base}"
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),  # (0..1) range; detector will normalize internally
        ])
        self.image_size = image_size

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        x = self.tf(img)                      # [3,H,W] with H=W=image_size
        meta = {
            "stem": p.stem,
            "orig_size": img.size,            # (W,H) if you need it later
            "proc_size": (self.image_size, self.image_size),
        }
        # Dummy target with image_id for downstream naming
        tgt = {"image_id": torch.tensor([idx])}
        return x, tgt, meta

def collate_fn(batch):
    imgs, tgts, metas = zip(*batch)
    return list(imgs), list(tgts), list(metas)

# =========================
# Standard DINOv3 backbone wrapper (as in your pipeline)
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
                t, Ht, Wt = self._get_patch_tokens(torch.zeros(1,3,32,32))
                C = t.shape[-1]
        self.out_channels = C

    @torch.no_grad()
    def _maybe_hw(self, x):
        _, _, H, W = x.shape
        return math.ceil(H/self.patch_stride), math.ceil(W/self.patch_stride)

    def _get_patch_tokens(self, x):
        out = None
        try:
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
        except Exception:
            pass
        if hasattr(self.dino, "get_intermediate_layers"):
            t = self.dino.get_intermediate_layers(x, n=1, return_class_token=False)[0]
            Ht, Wt = self._maybe_hw(x)
            return t, Ht, Wt
        t = self.dino(x); Ht, Wt = self._maybe_hw(x)
        if t.dim()==3 and t.shape[1]==(Ht*Wt+1): t = t[:,1:,:]
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
    print(model)
    return model

# =========================
# Save helpers
# =========================
def write_json(path: Path, boxes, scores, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            [{"bbox": [float(x) for x in b],
              "score": float(s),
              "cls": int(c)} for b, s, c in zip(boxes, scores, labels)],
            f
        )

def write_yolo_txt(path: Path, boxes, labels, W: int, H: int, save_empty: bool = True):
    """
    Write YOLO-format .txt file.
    If save_empty=False, skip writing when there are no boxes.
    """
    if len(boxes) == 0:
        if save_empty:
            path.write_text("")   # create an empty file
        return

    with open(path, "w") as f:
        for b, c in zip(boxes, labels):
            xmin, ymin, xmax, ymax = b
            xc = (xmin + xmax) / 2 / W
            yc = (ymin + ymax) / 2 / H
            w  = (xmax - xmin) / W
            h  = (ymax - ymin) / H
            f.write(f"{c-1} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

# =========================
# Inference
# =========================
@torch.no_grad()
def run_teacher(model: FasterRCNN, dl: DataLoader, json_dir: Path, txt_dir: Path, device, image_size: int):
    json_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    for images, targets, metas in tqdm(dl, desc=f"[RGB PSEUDO] {json_dir.parent.name}"):
        images = [im.to(device) for im in images]
        preds  = model(images)

        for i, p in enumerate(preds):
            boxes  = p["boxes"].detach().cpu()
            scores = p["scores"].detach().cpu()
            labels = p["labels"].detach().cpu()

            # filter low scores
            keep = (scores >= SCORE_THRESH).nonzero(as_tuple=False).flatten()
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # per-class NMS
            if boxes.numel() > 0:
                fb, fs, fl = [], [], []
                for cls_id in labels.unique(sorted=True).tolist():
                    idx = (labels == cls_id).nonzero(as_tuple=False).flatten()
                    if idx.numel() == 0: continue
                    keep_idx = nms(boxes[idx], scores[idx], NMS_IOU)
                    fb.append(boxes[idx][keep_idx]); fs.append(scores[idx][keep_idx]); fl.append(labels[idx][keep_idx])
                if fb:
                    boxes  = torch.cat(fb, 0)
                    scores = torch.cat(fs, 0)
                    labels = torch.cat(fl, 0)

            stem = metas[i]["stem"]
            # Save JSON
            write_json(json_dir / f"{stem}.json",
                       boxes.tolist(), scores.tolist(), labels.tolist())
            # Save YOLO txt normalized to training resize
            write_yolo_txt(txt_dir / f"{stem}.txt",
                           boxes.tolist(), labels.tolist(),
                           W=image_size, H=image_size)

# =========================
# Main
# =========================
def main():
    print(f"[INFO] Using DINOv3 from: {DINOV3_LOCATION}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build DINOv3 encoder
    dino = torch.hub.load(
        repo_or_dir=DINOV3_LOCATION,
        model=DINO_MODEL_NAME,
        source="local",
        weights=DINO_WEIGHTS,
        skip_validation=True,
    )
    # Build detector and load teacher weights
    model = create_model(dino, num_classes=NUM_CLASSES, image_size=IMAGE_SIZE).to(device)
    state = torch.load(RGB_TEACHER_CKPT, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    # Targets
    for region in ["bangladesh", "pak_punjab"]:
        root = REGION_ROOTS[region]
        ds = UnlabeledRGBDataset(root, split=SPLIT, image_size=IMAGE_SIZE)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)

        out_base = Path(root) / SPLIT
        json_dir = out_base / "pseudo_rgb_json"
        txt_dir  = out_base / "pseudo_rgb_labels"

        print(f"==> Generating pseudo labels (RGB) for {region}/{SPLIT}")
        run_teacher(model, dl, json_dir, txt_dir, device, IMAGE_SIZE)

    print("🎉 Done: pseudo labels saved in pseudo_rgb_json/ and pseudo_rgb_labels/ under each target region.")

if __name__ == "__main__":
    main()