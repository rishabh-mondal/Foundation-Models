#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeoContrast Phase 1 (switchable AEF backends) + split inference from CSV name:
- aef_mode="mean": per-image [64] .npy vectors (bandwise mean over AEF GeoTIFF)
- aef_mode="map":  load 64xHxW GeoTIFF and encode via a small CNN + MLP adapter
- aef_mode="xattn": shift-aware image<->AEF cross-attention over token grids
- CLIP-style image <-> AEF contrast (+ optional SimCLR image<->image)
- Optional AEF-aware hard-negative penalty
- Saves ONLY the DINO encoder weights for Phase-B fine-tuning

CSV format (train/val/test/all):
  For mean-mode:
    region,filename,aef_npy
  For map/xattn (preferred):
    region,filename,aef_tif   # if blank, we auto-resolve by stem under region root
"""

import os, math, csv, random, argparse
from pathlib import Path
from typing import Dict, Optional, List, Tuple
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
    if split_hint:
        yield base / split_hint / "images" / filename
    for split in SPLIT_KEYS:
        yield base / split / "images" / filename
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
    if split_hint:
        yield base / split_hint / "embeddings" / f"{stem}.tif"
    for split in SPLIT_KEYS:
        yield base / split / "embeddings" / f"{stem}.tif"
    yield base / "embeddings" / f"{stem}.tif"
    yield base / f"{stem}.tif"

def resolve_aef_tif_path(region: str, filename: str, csv_tif: Optional[str], split_hint: Optional[str]) -> Path:
    if csv_tif:
        p = Path(csv_tif)
        if p.exists():
            return p
    stem = Path(filename).stem
    base = Path(REGION_ROOTS[region])
    for cand in _candidate_aef_tif_paths(base, stem, split_hint):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"AEF GeoTIFF not found for stem='{stem}' (searched under {base}/<split>/embeddings)")

def load_aef_tif_as_tensor(tif_path: Path) -> torch.Tensor:
    """Load 64xHxW float32 tensor in [C,H,W]. (No normalization; encoder handles it.)"""
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

def _maybe_roll_map(M: torch.Tensor, k: int, p: float) -> torch.Tensor:
    """Randomly roll the AEF map by ±k patches (training-time augmentation)."""
    if k <= 0 or random.random() > p:
        return M
    S = M.shape[-1]
    max_pix = max(1, (S // 14) * k)  # approx; real shift handled at token level
    dy = random.randint(-max_pix, max_pix)
    dx = random.randint(-max_pix, max_pix)
    return M.roll(shifts=(dy, dx), dims=(-2, -1))

# --- grid inference for token sequences (handles non-square) ---
def _infer_hw(n: int) -> Tuple[int, int]:
    g = int(round(n ** 0.5))
    if g * g == n:
        return g, g
    gh = int(math.floor(n ** 0.5))
    gw = max(1, (n + gh - 1) // max(1, gh))
    if gh * gw != n:
        # fallback: make it 1 x n
        return 1, n
    return gh, gw

# -------------------
# Dataset
# -------------------
class PerImageAEFDataset(Dataset):
    """
    Returns (two augmented image views) + AEF target (vector OR map):
      mean mode: x1, x2, v[64], region, filename
      map  mode: x1, x2, M[64,S,S], region, filename  (S = aef_map_size)
      xattn mode: same as map (AEF map), fusion happens in the model
    """
    def __init__(
        self,
        csv_path: str,
        aef_mode: str,
        image_size: int = 224,
        aef_map_size: int = 224,
        aef_jitter_k: int = 0,
        aef_jitter_p: float = 0.0,
    ):
        assert aef_mode in ("mean", "map", "xattn")
        self.csv_path = str(csv_path)
        self.split_hint = infer_split_from_csv(self.csv_path)
        self.aef_mode = aef_mode
        self.aef_map_size = int(aef_map_size)
        self.aef_jitter_k = int(aef_jitter_k)
        self.aef_jitter_p = float(aef_jitter_p)

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
        img_path = resolve_image_path(r["region"], r["filename"], self.split_hint)
        img = Image.open(img_path).convert("RGB")
        x1 = self.tf(img)
        x2 = self.tf(img)

        if self.aef_mode == "mean":
            npy_path = r["aef_npy"]
            if not npy_path:
                raise RuntimeError("CSV must provide aef_npy for mean mode (or switch to --aef_mode map/xattn).")
            v = np.load(npy_path).astype(np.float32).reshape(-1)
            if v.shape[0] != 64:
                raise RuntimeError(f"AEF vector must be [64]; got {v.shape} for {npy_path}")
            v = torch.from_numpy(v)
            v = v / (v.norm(p=2) + 1e-6)
            return x1, x2, v, r["region"], r["filename"]

        # map or xattn: load AEF GeoTIFF and resize to [64,S,S]
        tif_path = resolve_aef_tif_path(r["region"], r["filename"], r["aef_tif"], self.split_hint)
        M = load_aef_tif_as_tensor(tif_path)  # [64,H,W]
        S = self.aef_map_size
        if S and (M.shape[-2] != S or M.shape[-1] != S):
            M = F.interpolate(M.unsqueeze(0), size=(S, S), mode="bilinear", align_corners=False).squeeze(0)
        M = _maybe_roll_map(M, self.aef_jitter_k, self.aef_jitter_p)
        return x1, x2, M, r["region"], r["filename"]

# -------------------
# Encoders
# -------------------
class ImageEncoder(nn.Module):
    def __init__(self, dino, proj_dim=256):
        super().__init__()
        self.dino = dino
        C = getattr(dino, "embed_dim", None) or getattr(dino, "num_features", None) or 1024
        self.proj = nn.Sequential(nn.Linear(C, C), nn.GELU(), nn.Linear(C, proj_dim))

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
    def __init__(self, in_dim=64, proj_dim=256, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(True), nn.Linear(hidden, proj_dim))
    def forward(self, v):
        return F.normalize(self.net(v), dim=-1)

class AEFMapEncoder(nn.Module):
    def __init__(self, in_ch=64, proj_dim=256, mid_ch=128, hidden=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.InstanceNorm2d(in_ch, affine=False, eps=1e-6),
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Linear(mid_ch, hidden), nn.ReLU(True), nn.Linear(hidden, proj_dim))

    def forward(self, M):  # [B,64,H,W]
        h = self.encoder(M)
        g = self.pool(h).squeeze(-1).squeeze(-1)
        z = self.proj(g)
        return F.normalize(z, dim=-1)

# -------------------
# Cross-attention option (xattn)
# -------------------
class ImageEncoderWithTokens(ImageEncoder):
    """Expose patch tokens (pre-pooled) for cross-attention."""
    def forward_tokens(self, x):
        t = self._tokens(x)  # [B, N(+1), C]
        if t.dim() == 3 and t.shape[1] > 1:
            t = t[:, 1:, :]  # drop cls if present
        return t  # [B, N, C]

class AEFMapTokenizer(nn.Module):
    """AEF map (64xSxS) -> tokens [B, Na, C] aligned to a grid."""
    def __init__(self, in_ch=64, out_dim=1024, grid=14, mid=192):
        super().__init__()
        self.grid = grid
        self.backbone = nn.Sequential(
            nn.InstanceNorm2d(in_ch, affine=False, eps=1e-6),
            nn.Conv2d(in_ch, mid, 3, padding=1, bias=False), nn.BatchNorm2d(mid), nn.GELU(),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),   nn.BatchNorm2d(mid), nn.GELU(),
            nn.Conv2d(mid, out_dim, 1, bias=False)
        )
        self.pool = nn.AdaptiveAvgPool2d((grid, grid))  # align to ViT target grid

    def forward(self, M):  # [B,64,S,S]
        Fm = self.pool(self.backbone(M))                # [B, C, g, g]
        B, C, g, _ = Fm.shape
        tokens = Fm.permute(0, 2, 3, 1).reshape(B, g*g, C)  # [B, Na, C]
        return tokens, g

class ShiftAwareCrossAttention(nn.Module):
    """
    Shift-invariant cross-attn: enumerate small spatial shifts (rolls) of grid tokens,
    compute Q·K,V attention per shift, then max-pool across shifts. Supports rectangular grids.
    """
    def __init__(self, dim, num_heads=4, max_shift=2, qkv_bias=True, proj_out_dim=256):
        super().__init__()
        self.dim = dim
        self.h = num_heads
        self.max_shift = max_shift
        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, proj_out_dim)

    def _attend(self, Q, K, V):
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        A = attn.softmax(dim=-1)
        return A @ V

    def forward(self, img_tokens, aef_tokens, grid_hw: Optional[Tuple[int,int]] = None):
        """
        img_tokens: [B, Ni, C]   (queries)
        aef_tokens: [B, Nk, C]   (keys/values) laid out on a grid (gh, gw)
        grid_hw: optional (gh, gw). If None, inferred from Nk.
        returns: fused embedding [B, proj_out_dim]
        """
        B, Ni, C = img_tokens.shape
        Nk = aef_tokens.shape[1]

        if grid_hw is None:
            gh, gw = _infer_hw(Nk)
        else:
            gh, gw = grid_hw
        assert gh * gw == Nk, f"aef_tokens={Nk}, grid_hw=({gh},{gw}) mismatch"

        # project to Q,K,V and reshape to heads
        Q = self.q(img_tokens).view(B, Ni, self.h, C // self.h).permute(0, 2, 1, 3)  # [B,H,Ni,Dh]
        K0 = self.k(aef_tokens).view(B, Nk, self.h, C // self.h).permute(0, 2, 1, 3) # [B,H,Nk,Dh]
        V0 = self.v(aef_tokens).view(B, Nk, self.h, C // self.h).permute(0, 2, 1, 3)

        # gridify K,V as (gh, gw)
        Kgrid = K0.permute(0,1,3,2).reshape(B, self.h, C//self.h, gh, gw)  # [B,H,Dh,gh,gw]
        Vgrid = V0.permute(0,1,3,2).reshape(B, self.h, C//self.h, gh, gw)

        outs = []
        shifts = range(-self.max_shift, self.max_shift+1)
        for dy in shifts:
            for dx in shifts:
                if dy == 0 and dx == 0:
                    K = K0; V = V0
                else:
                    Kroll = Kgrid.roll(shifts=(dy, dx), dims=(-2, -1)).reshape(B, self.h, C//self.h, gh*gw)
                    Vroll = Vgrid.roll(shifts=(dy, dx), dims=(-2, -1)).reshape(B, self.h, C//self.h, gh*gw)
                    K = Kroll.permute(0,1,3,2)  # [B,H,Nk,Dh]
                    V = Vroll.permute(0,1,3,2)  # [B,H,Nk,Dh]
                O = self._attend(Q, K, V)     # [B,H,Ni,Dh]
                outs.append(O)

        Omax = torch.stack(outs, dim=0).max(dim=0).values      # [B,H,Ni,Dh]
        Omax = Omax.permute(0,2,1,3).reshape(B, Ni, C)         # [B,Ni,C]
        pooled = Omax.mean(dim=1)                               # [B,C]
        return F.normalize(self.proj(pooled), dim=-1)           # [B,proj_out_dim]

class GeoContrastX(nn.Module):
    """GeoContrast with shift-aware image<->AEF cross-attention pooling (aef_mode='xattn')."""
    def __init__(self, dino, proj_dim=256, aef_grid=14, heads=4, max_shift=2):
        super().__init__()
        C = getattr(dino, "embed_dim", None) or getattr(dino, "num_features", None) or 1024
        self.img_enc = ImageEncoderWithTokens(dino, proj_dim=proj_dim)
        self.aef_tok = AEFMapTokenizer(in_ch=64, out_dim=C, grid=aef_grid)
        self.i2a = ShiftAwareCrossAttention(dim=C, num_heads=heads, max_shift=max_shift, proj_out_dim=proj_dim)
        self.a2i = ShiftAwareCrossAttention(dim=C, num_heads=heads, max_shift=max_shift, proj_out_dim=proj_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0/0.07)))

    def fused_embeddings(self, x_img, M_aef):
        ti = self.img_enc.forward_tokens(x_img)     # [B,Ni,C]
        ta, _ = self.aef_tok(M_aef)                 # [B,Nk,C]

        # infer grids independently (handles 13x15 vs 14x14, etc.)
        gi_h, gi_w = _infer_hw(ti.shape[1])
        ga_h, ga_w = _infer_hw(ta.shape[1])

        zi = self.i2a(ti, ta, (ga_h, ga_w))         # image queries over AEF grid
        za = self.a2i(ta, ti, (gi_h, gi_w))         # AEF queries over image grid
        return zi, za

    def logits_image_aef(self, zi, za):
        s = self.logit_scale.exp().clamp(1e-3, 100.0)
        return s * (zi @ za.t())

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
        S = (za @ za.t()).clamp(0, 1)
        W = S - torch.diag_embed(torch.diag(S))
        W = W / (W.sum() + 1e-6)
    P = logits_img_aef.softmax(dim=1)
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

def save_dino_encoder(model, out_path: str):
    sd = {f"dino.{k}": v.cpu() for k, v in model.img_enc.dino.state_dict().items()}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, out_path)
    print(f"[OK] Saved DINO encoder -> {out_path}")

# -------------------
# Eval (optional): retrieval@1
# -------------------
@torch.no_grad()
def eval_retrieval_top1(model, loader: DataLoader, device: torch.device, aef_mode: str):
    model.eval()
    correct = 0
    total = 0
    for x1, _, aef, _, _ in tqdm(loader, desc="Eval (retrieval@1)"):
        x1 = x1.to(device, non_blocking=True)

        if aef_mode == "mean":
            zi = model.img_enc(x1)
            v  = aef.to(device, non_blocking=True)      # [B,64]
            za = model.aef_enc(v)
        elif aef_mode == "map":
            zi = model.img_enc(x1)
            M  = aef.to(device, non_blocking=True)      # [B,64,H,W]
            za = model.aef_enc(M)
        else:  # "xattn"
            M  = aef.to(device, non_blocking=True)
            zi, za = model.fused_embeddings(x1, M)

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
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # data
    ds_train = PerImageAEFDataset(
        args.train_csv, aef_mode=args.aef_mode, image_size=args.image_size,
        aef_map_size=args.aef_map_size, aef_jitter_k=args.aef_jitter_k, aef_jitter_p=args.aef_jitter_p
    )
    print(f"[Data] Train rows: {len(ds_train)} from {args.train_csv}")
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)

    dl_val = None
    if args.val_csv and Path(args.val_csv).exists():
        ds_val = PerImageAEFDataset(
            args.val_csv, aef_mode=args.aef_mode, image_size=args.image_size,
            aef_map_size=args.aef_map_size, aef_jitter_k=0, aef_jitter_p=0.0
        )
        print(f"[Data] Val rows: {len(ds_val)} from {args.val_csv}")
        dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # model
    dino = build_dino()
    if args.aef_mode == "xattn":
        grid = args.image_size // args.vit_patch  # target grid for tokenizer
        model = GeoContrastX(
            dino=dino,
            proj_dim=args.proj_dim,
            aef_grid=grid,
            heads=args.xattn_heads,
            max_shift=args.xattn_shift
        ).to(device)
    else:
        img_enc = ImageEncoder(dino, proj_dim=args.proj_dim).to(device)
        if args.aef_mode == "mean":
            aef_enc = AEFVectorEncoder(in_dim=64, proj_dim=args.proj_dim, hidden=args.hidden_dim).to(device)
        else:  # "map"
            aef_enc = AEFMapEncoder(in_ch=64, proj_dim=args.proj_dim, mid_ch=args.cnn_mid_ch, hidden=args.hidden_dim).to(device)
        model   = GeoContrast(img_enc, aef_enc, init_temp=args.init_temp).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda', enabled=not args.no_amp)

    best_val = -1.0
    for ep in range(1, args.epochs+1):
        model.train()
        loss_sum = 0.0
        pbar = tqdm(dl_train, desc=f"Train ep{ep}/{args.epochs}")
        for x1, x2, aef, _, _ in pbar:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=not args.no_amp):
                if args.aef_mode == "xattn":
                    M = aef.to(device, non_blocking=True)           # [B,64,H,W]
                    zi, za = model.fused_embeddings(x1, M)          # fused, shift-robust
                    logits = model.logits_image_aef(zi, za)
                    loss = clip_loss(logits)
                    if args.img_img_lambda > 0:
                        zi2 = model.img_enc(x2)                     # pooled image path
                        loss = loss + args.img_img_lambda * simclr_loss(zi, zi2, temperature=args.simclr_temp)
                else:
                    zi1 = model.img_enc(x1)
                    if args.aef_mode == "mean":
                        v = aef.to(device, non_blocking=True)       # [B,64]
                        za = model.aef_enc(v)
                    else:  # "map"
                        M = aef.to(device, non_blocking=True)       # [B,64,H,W]
                        za = model.aef_enc(M)
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
            pbar.set_postfix(loss=f"{float(loss):.4f}", T=f"{float(model.logit_scale.exp()):.3f}")

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
    ap = argparse.ArgumentParser("GeoContrast Phase 1 (mean/map/xattn AEF backends) with split inference")
    ap.add_argument("--train_csv", default="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/all_images.csv", help="CSV with region,filename,(aef_npy|aef_tif)")
    ap.add_argument("--val_csv",   default="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/all_val.csv", help="Optional CSV for retrieval@1 eval")
    ap.add_argument("--ckpt_out",  default="dinov3_geocontrast_mean_all_splits_800.pth")

    ap.add_argument("--aef_mode", choices=["mean","map","xattn"], default="xattn",
                    help="mean: .npy[64]; map: CNN over 64xHxW; xattn: shift-aware cross-attention")
    ap.add_argument("--aef_map_size", type=int, default=800, help="Resize AEF GeoTIFFs to SxS before batching")
    ap.add_argument("--image_size",   type=int, default=512, help="Input image size (after augment)")
    ap.add_argument("--vit_patch",    type=int, default=16,  help="ViT patch size (e.g., 16 for ViT/16)")
    ap.add_argument("--xattn_heads",  type=int, default=4,   help="Cross-attention heads for xattn mode")
    ap.add_argument("--xattn_shift",  type=int, default=2,   help="Max patch shift K for xattn (±K)")

    ap.add_argument("--aef_jitter_k", type=int, default=0,   help="Random roll up to ±k tokens (approx) on AEF map")
    ap.add_argument("--aef_jitter_p", type=float, default=0.0, help="Prob of applying AEF jitter per sample")

    ap.add_argument("--batch_size",   type=int, default=32)
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
    ap.add_argument("--device", type=str, default="cuda:0", help="e.g., cuda:3 or cpu")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)

# CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
# nohup python -u geocontrast_phase1_xattn_all_splits_pretrain.py \
#   --aef_mode xattn \
#   --vit_patch 16 --xattn_heads 4 --xattn_shift 2 \
#   --aef_jitter_p 0.25 --aef_jitter_k 2 \
#   --batch_size 8 --num_workers 4 \
#   --no_amp \
#   --device cuda:0 \
#   > geocontrast_phase1_xattn_all_splits_dbg.log 2>&1 &



# CUDA_VISIBLE_DEVICES=1 \
# nohup python -u geocontrast_phase1_xattn_all_splits_pretrain.py \
#   --aef_mode mean \
#   --device cuda:0 \
#   > geocontrast_phase1_mean_800_512_dbg.log 2>&1 &



# CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 \
# nohup python -u geocontrast_phase1_xattn_all_splits_pretrain.py \
#   --aef_mode xattn \
#   --image_size 800 --vit_patch 16 --xattn_heads 4 --xattn_shift 2 \
#   --aef_jitter_p 0.25 --aef_jitter_k 2 \
#   --batch_size 8 --num_workers 4 \
#   --no_amp \
#   --device cuda:0 \
#   > geocontrast_phase1_xattn_all_splits_dbg.log 2>&1 &