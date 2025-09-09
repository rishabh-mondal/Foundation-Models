#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeoContrast Phase 1 (switchable AEF backends) + split inference from CSV name:
- aef_mode="mean": per-image [64] .npy vectors (bandwise mean over AEF GeoTIFF)
- aef_mode="map":  load 64xHxW GeoTIFF and encode via a small CNN + MLP adapter
- CLIP-style image <-> AEF contrast (+ optional SimCLR image<->image)
- Optional AEF-aware hard-negative penalty
- Saves ONLY the DINO encoder weights for Phase-B fine-tuning

CSV format (train/val):
  Minimal for mean-mode:
    region,filename,aef_npy
  For map-mode (preferred):
    region,filename,aef_tif
  If map-mode and aef_tif is absent, we try to auto-resolve the TIF by stem.

Region roots must contain split subfolders (train/val/test):
  <REGION_ROOTS[region]>/<split>/{images,labels,embeddings}

Run (example, mean mode on an all_train.csv):
  python geocontrast_phase1_switchable.py \
    --train_csv /path/to/all_train.csv \
    --val_csv   /path/to/all_val.csv \
    --ckpt_out  /path/to/checkpoints/geocontrast_dinov3_vitl16.pth \
    --aef_mode mean
"""

import os, math, csv, random, argparse
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# -------------------
# Paths (EDIT THESE IF NEEDED)
# -------------------
DINOV3_GITHUB_LOCATION = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/dinov3"
DINOV3_LOCATION = os.getenv("DINOV3_LOCATION") or DINOV3_GITHUB_LOCATION
DINO_MODEL_NAME = "dinov3_vitl16"
DINO_WEIGHTS    = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/dinov3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"

# Region roots — must point to the directory that contains train/val/test for each region
REGION_ROOTS: Dict[str, str] = {
    "uttar_pradesh": "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/uttar_pradesh",
    "bangladesh":    "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/bangladesh",
    "pak_punjab":    "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/pak_punjab",
}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SPLIT_KEYS = ("train", "val", "test")

# -------------------
# Utils
# -------------------
def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_img_transform(image_size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
        transforms.RandomGrayscale(0.2),
        transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
    ])

def infer_split_from_csv(csv_path: str) -> Optional[str]:
    """Return 'train' | 'val' | 'test' if present in CSV filename, else None."""
    name = Path(csv_path).name.lower()
    for s in SPLIT_KEYS:
        if s in name:
            return s
    return None

def _candidate_image_paths(base: Path, filename: str, split_hint: Optional[str]):
    # prefer hinted split
    if split_hint:
        yield base / split_hint / "images" / filename
    # try all splits
    for split in SPLIT_KEYS:
        yield base / split / "images" / filename
    # legacy layouts (rare)
    yield base / "images" / filename
    yield base / filename

def resolve_image_path(region: str, filename: str, split_hint: Optional[str]) -> Path:
    p = Path(filename)
    if p.is_absolute() and p.exists():
        return p
    base = Path(REGION_ROOTS[region])
    for cand in _candidate_image_paths(base, filename, split_hint):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Image not found: {filename} (searched under {base}/<split>/images)")

def _candidate_aef_tif_paths(base: Path, stem: str, split_hint: Optional[str]):
    # prefer hinted split
    if split_hint:
        yield base / split_hint / "embeddings" / f"{stem}.tif"
    # try all splits
    for split in SPLIT_KEYS:
        yield base / split / "embeddings" / f"{stem}.tif"
    # legacy
    yield base / "embeddings" / f"{stem}.tif"
    yield base / f"{stem}.tif"

def resolve_aef_tif_path(region: str, filename: str, csv_tif: Optional[str], split_hint: Optional[str]) -> Path:
    # CSV provided full path?
    if csv_tif:
        p = Path(csv_tif)
        if p.exists():
            return p
    # Infer by stem under region root
    stem = Path(filename).stem
    base = Path(REGION_ROOTS[region])
    for cand in _candidate_aef_tif_paths(base, stem, split_hint):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"AEF GeoTIFF not found for stem='{stem}' (searched under {base}/<split>/embeddings)")

def load_aef_tif_as_tensor(tif_path: Path) -> torch.Tensor:
    """
    Load 64xHxW float32 tensor in [C,H,W]. (No normalization; encoder handles it.)
    """
    arr = None
    if rio is not None:
        try:
            with rio.open(tif_path) as ds:
                arr = ds.read().astype(np.float32)  # [C,H,W]
        except Exception:
            arr = None
    if arr is None and tiff is not None:
        arr = tiff.imread(str(tif_path)).astype(np.float32)  # [H,W,C] or [C,H,W]
        if arr.ndim == 3 and arr.shape[0] != 64 and arr.shape[-1] == 64:
            arr = np.moveaxis(arr, -1, 0)
    if arr is None or arr.ndim != 3:
        raise RuntimeError(f"Bad AEF TIF (3D required): {tif_path}")
    return torch.from_numpy(arr)  # [C,H,W]

# -------------------
# Dataset
# -------------------
# -------------------
# Dataset
# -------------------
class PerImageAEFDataset(Dataset):
    """
    Returns (two augmented image views) + AEF target (vector OR map):
      mean mode: x1, x2, v[64], region, filename
      map  mode: x1, x2, M[64,S,S], region, filename  (S = aef_map_size)

    CSV columns accepted:
      - region, filename, aef_npy  (for mean)
      - region, filename, aef_tif  (for map; optional but preferred)
    """
    def __init__(
        self,
        csv_path: str,
        aef_mode: str,
        image_size: int = 224,
        aef_map_size: int = 224,   # NEW: force a common H=W=S for map mode
    ):
        assert aef_mode in ("mean", "map")
        self.csv_path = str(csv_path)
        self.split_hint = infer_split_from_csv(self.csv_path)  # 'train'|'val'|'test' or None
        self.aef_mode = aef_mode
        self.aef_map_size = int(aef_map_size)

        self.rows = []
        with open(csv_path, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                self.rows.append({
                    "region": row["region"],
                    "filename": row["filename"],
                    "aef_npy": row.get("aef_npy", ""),
                    "aef_tif": row.get("aef_tif", ""),
                })
        assert len(self.rows) > 0, f"No rows in {csv_path}"
        self.tf = build_img_transform(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]

        # --- RGB image (two augmented views) ---
        img_path = resolve_image_path(r["region"], r["filename"], self.split_hint)
        img = Image.open(img_path).convert("RGB")
        x1 = self.tf(img)
        x2 = self.tf(img)

        if self.aef_mode == "mean":
            # --- AEF vector [64] ---
            npy_path = r["aef_npy"]
            if not npy_path:
                raise RuntimeError("CSV must provide aef_npy for mean mode (or switch to --aef_mode map).")
            v = np.load(npy_path).astype(np.float32).reshape(-1)
            if v.shape[0] != 64:
                raise RuntimeError(f"AEF vector must be [64]; got {v.shape} for {npy_path}")
            v = torch.from_numpy(v)
            v = v / (v.norm(p=2) + 1e-6)  # L2 normalize for stability
            return x1, x2, v, r["region"], r["filename"]

        else:
            # --- AEF map [64,H,W] -> resize to [64,S,S] for batching ---
            tif_path = resolve_aef_tif_path(r["region"], r["filename"], r["aef_tif"], self.split_hint)
            M = load_aef_tif_as_tensor(tif_path)  # float32 [64,H,W]
            S = self.aef_map_size
            if S and (M.shape[-2] != S or M.shape[-1] != S):
                # bilinear expects [N,C,H,W]
                M = F.interpolate(
                    M.unsqueeze(0), size=(S, S),
                    mode="bilinear", align_corners=False
                ).squeeze(0)  # [64,S,S]
            return x1, x2, M, r["region"], r["filename"]
        


# -------------------
# Encoders
# -------------------
class ImageEncoder(nn.Module):
    def __init__(self, dino, proj_dim=256):
        super().__init__()
        self.dino = dino
        C = getattr(dino, "embed_dim", None) or getattr(dino, "num_features", None) or 1024
        self.proj = nn.Sequential(
            nn.Linear(C, C), nn.GELU(), nn.Linear(C, proj_dim)
        )

    @torch.no_grad()
    def _tokens(self, x):
        out = self.dino.forward_features(x)
        if isinstance(out, dict) and "x_norm_patchtokens" in out:
            return out["x_norm_patchtokens"]
        if isinstance(out, dict) and out.get("tokens") is not None:
            return out["tokens"]
        if hasattr(self.dino, "get_intermediate_layers"):
            return self.dino.get_intermediate_layers(x, 1, False)[0]
        return out

    def forward(self, x):
        t = self._tokens(x)             # [B, N(+1), C]
        if t.dim() == 3 and t.shape[1] > 1:
            z = t[:, 1:, :].mean(1)     # mean over patches (drop cls if present)
        else:
            z = t.mean(1)
        z = F.normalize(self.proj(z), dim=-1)
        return z

class AEFVectorEncoder(nn.Module):
    """Mean strategy: project [64] -> proj_dim, then L2 normalize."""
    def __init__(self, in_dim=64, proj_dim=256, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(True), nn.Linear(hidden, proj_dim)
        )
    def forward(self, v):
        return F.normalize(self.net(v), dim=-1)

class AEFMapEncoder(nn.Module):
    """
    CNN adapter for full AEF map (64xHxW) -> proj_dim.
    Light stack + global average pooling + MLP.
    """
    def __init__(self, in_ch=64, proj_dim=256, mid_ch=128, hidden=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.InstanceNorm2d(in_ch, affine=False, eps=1e-6),
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(mid_ch, hidden), nn.ReLU(True), nn.Linear(hidden, proj_dim)
        )

    def forward(self, M):            # M: [B,64,H,W]
        h = self.encoder(M)          # [B,mid,H,W]
        g = self.pool(h).squeeze(-1).squeeze(-1)  # [B,mid]
        z = self.proj(g)             # [B,proj_dim]
        return F.normalize(z, dim=-1)

# -------------------
# GeoContrast model & losses
# -------------------
class GeoContrast(nn.Module):
    def __init__(self, img_enc, aef_enc, init_temp=0.07):
        super().__init__()
        self.img_enc = img_enc
        self.aef_enc = aef_enc
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temp)))

    def logits_image_aef(self, zi, za):
        s = self.logit_scale.exp().clamp(1e-3, 100.0)
        return s * (zi @ za.t())  # [B,B]

def clip_loss(logits):
    B = logits.size(0)
    y = torch.arange(B, device=logits.device)
    return 0.5*(F.cross_entropy(logits, y) + F.cross_entropy(logits.t(), y))

def simclr_loss(z1, z2, temperature=0.2):
    sim = (z1 @ z2.t()) / temperature
    B = z1.size(0)
    y = torch.arange(B, device=z1.device)
    return 0.5*(F.cross_entropy(sim, y) + F.cross_entropy(sim.t(), y))

def aef_hardneg_penalty(logits_img_aef, za, alpha=0.2):
    if alpha <= 0:
        return torch.zeros((), device=logits_img_aef.device)
    with torch.no_grad():
        S = (za @ za.t()).clamp(0, 1)               # cosine sim
        W = S - torch.diag_embed(torch.diag(S))     # zero diagonal
        W = W / (W.sum() + 1e-6)
    P = logits_img_aef.softmax(dim=1)               # p(aef_j | img_i)
    return alpha * (P * W).sum()

# -------------------
# Build / Save
# -------------------
def build_dino():
    return torch.hub.load(
        repo_or_dir=DINOV3_LOCATION,
        model=DINO_MODEL_NAME,
        source="local",
        weights=DINO_WEIGHTS,
        skip_validation=True,
    )

def save_dino_encoder(model: GeoContrast, out_path: str):
    sd = {f"dino.{k}": v.cpu() for k, v in model.img_enc.dino.state_dict().items()}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, out_path)
    print(f"[OK] Saved DINO encoder -> {out_path}")

# -------------------
# Eval (optional): retrieval@1
# -------------------
@torch.no_grad()
def eval_retrieval_top1(model: GeoContrast, loader: DataLoader, device: torch.device, aef_mode: str):
    model.eval()
    correct = 0
    total = 0
    for x1, _, aef, _, _ in tqdm(loader, desc="Eval (retrieval@1)"):
        x1 = x1.to(device, non_blocking=True)
        zi = model.img_enc(x1)
        if aef_mode == "mean":
            v  = aef.to(device, non_blocking=True)      # [B,64]
            za = model.aef_enc(v)
        else:
            M  = aef.to(device, non_blocking=True)      # [B,64,H,W]
            za = model.aef_enc(M)
        logits = model.logits_image_aef(zi, za)         # [B,B]
        pred = logits.argmax(dim=1)
        y = torch.arange(logits.size(0), device=logits.device)
        correct += int((pred == y).sum().item())
        total   += logits.size(0)
    acc = 100.0 * correct / max(1, total)
    print(f"[Retrieval@1] image->AEF top1 = {acc:.2f}%")
    return acc

# -------------------
# Train
# -------------------
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # data
    ds_train = PerImageAEFDataset(args.train_csv, aef_mode=args.aef_mode, image_size=args.image_size,aef_map_size=args.aef_map_size,)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)

    dl_val = None
    if args.val_csv and Path(args.val_csv).exists():
        ds_val = PerImageAEFDataset(args.val_csv, aef_mode=args.aef_mode, image_size=args.image_size,aef_map_size=args.aef_map_size,)
        dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # model
    dino    = build_dino()
    img_enc = ImageEncoder(dino, proj_dim=args.proj_dim).to(device)
    if args.aef_mode == "mean":
        aef_enc = AEFVectorEncoder(in_dim=64, proj_dim=args.proj_dim, hidden=args.hidden_dim).to(device)
    else:
        aef_enc = AEFMapEncoder(in_ch=64, proj_dim=args.proj_dim, mid_ch=args.cnn_mid_ch, hidden=args.hidden_dim).to(device)
    model   = GeoContrast(img_enc, aef_enc, init_temp=args.init_temp).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda:3", enabled=not args.no_amp)

    best_val = -1.0
    for ep in range(1, args.epochs+1):
        model.train()
        loss_sum = 0.0
        pbar = tqdm(dl_train, desc=f"Train ep{ep}/{args.epochs}")
        for x1, x2, aef, _, _ in pbar:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=not args.no_amp):
                zi1 = model.img_enc(x1)

                if args.aef_mode == "mean":
                    v = aef.to(device, non_blocking=True)          # [B,64]
                    za = model.aef_enc(v)                          # [B,D]
                else:
                    M = aef.to(device, non_blocking=True)          # [B,64,H,W]
                    za = model.aef_enc(M)                          # [B,D]

                logits = model.logits_image_aef(zi1, za)
                loss = clip_loss(logits)

                if args.img_img_lambda > 0:
                    zi2 = model.img_enc(x2)
                    loss = loss + args.img_img_lambda * simclr_loss(zi1, zi2, temperature=args.simclr_temp)

                if args.hardneg_alpha > 0:
                    loss = loss + aef_hardneg_penalty(logits, za, alpha=args.hardneg_alpha)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            loss_sum += float(loss)
            pbar.set_postfix(loss=f"{float(loss):.4f}",
                             T=f"{float(model.logit_scale.exp()):.3f}")

        avg = loss_sum / max(1, len(dl_train))
        print(f"[GeoContrast] epoch {ep:03d} avg_loss={avg:.4f}  T={float(model.logit_scale.exp()):.3f}")

        metric = -avg
        if dl_val is not None:
            acc = eval_retrieval_top1(model, dl_val, device, args.aef_mode)
            metric = acc
        if metric > best_val:
            best_val = metric
            save_dino_encoder(model, args.ckpt_out)

    # final save
    save_dino_encoder(model, args.ckpt_out)

# -------------------
# CLI
# -------------------
def parse_args():
    ap = argparse.ArgumentParser("GeoContrast Phase 1 (mean vs map AEF backends) with split inference")
    ap.add_argument("--train_csv", default="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/all_train.csv", help="CSV with region,filename,(aef_npy|aef_tif)")
    ap.add_argument("--val_csv",   default="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/all_val.csv", help="Optional CSV for retrieval@1 eval")
    ap.add_argument("--ckpt_out",  default="geocontrast_dinov3_vitl16_map_224.pth")
    ap.add_argument("--aef_map_size", type=int, default=224,
                help="Resize AEF GeoTIFFs to SxS before batching (map mode)")

    ap.add_argument("--aef_mode", choices=["mean","map"], default="map",
                    help="mean: use .npy [64]; map: load 64xHxW GeoTIFF via CNN")
    ap.add_argument("--image_size",   type=int, default=224)
    ap.add_argument("--batch_size",   type=int, default=256)
    ap.add_argument("--num_workers",  type=int, default=8)
    ap.add_argument("--epochs",       type=int, default=50)
    ap.add_argument("--lr",           type=float, default=1e-4)
    ap.add_argument("--wd",           type=float, default=0.05)
    ap.add_argument("--proj_dim",     type=int, default=256)
    ap.add_argument("--hidden_dim",   type=int, default=512)
    ap.add_argument("--cnn_mid_ch",   type=int, default=128, help="mid channels in AEFMapEncoder")
    ap.add_argument("--init_temp",    type=float, default=0.07)
    ap.add_argument("--img_img_lambda", type=float, default=0.5, help="0 to disable SimCLR branch")
    ap.add_argument("--simclr_temp",  type=float, default=0.2)
    ap.add_argument("--hardneg_alpha", type=float, default=0.2, help="0 to disable hard negative penalty")
    ap.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    ap.add_argument("--seed", type=int, default=1337)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)

# nohup python -u geocontrast_phase1_switchable.py  > geocontrast_phase1_map_224.log 2>&1 &  

# nohup python -u geocontrast_phase1_switchable.py \
#   --train_csv /home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/all_train.csv \
#   --val_csv   /home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/all_val.csv \
#   --aef_mode   map \
#   --ckpt_out  /home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/checkpoints/dinov3_geocontrast_map_224_all.pth \
#   > geocontrast_phase1_map_224.log 2>&1 &    