import os
import json
import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode as IM
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

# ----------------------------
# Dataset (multi-task targets)
# ----------------------------
class LandcoverCSV(Dataset):
    """
    CSV must have: image_path, season_no (1..4), lat, lon, landcover_id (0..7)
    Returns: image tensor, targets dict
    """
    def __init__(
        self,
        csv_path: str,
        img_size: int = 224,
        lat_mean: Optional[float] = None,
        lat_std: Optional[float] = None,
        lon_mean: Optional[float] = None,
        lon_std: Optional[float] = None,
    ):
        self.df = pd.read_csv(csv_path)
        req = {"image_path", "season_no", "lat", "lon", "landcover_id"}
        miss = req - set(self.df.columns)
        if miss:
            raise ValueError(f"Missing columns in {csv_path}: {miss}")

        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=IM.BILINEAR),
            transforms.ToTensor(),  # 0..1
        ])

        # use provided stats (from train) or compute
        lat_vals = self.df["lat"].astype(float).values
        lon_vals = self.df["lon"].astype(float).values
        self.lat_mean = float(np.mean(lat_vals)) if lat_mean is None else float(lat_mean)
        self.lat_std  = float(np.std(lat_vals) + 1e-8) if lat_std is None else float(lat_std)
        self.lon_mean = float(np.mean(lon_vals)) if lon_mean is None else float(lon_mean)
        self.lon_std  = float(np.std(lon_vals) + 1e-8) if lon_std is None else float(lon_std)

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        img = Image.open(r.image_path).convert("RGB")
        x = self.tf(img)  # C,H,W

        y_landcover = int(r.landcover_id)
        y_season = int(r.season_no) - 1  # 0..3

        latz = (float(r.lat) - self.lat_mean) / self.lat_std
        lonz = (float(r.lon) - self.lon_mean) / self.lon_std
        y_latlon_norm = torch.tensor([latz, lonz], dtype=torch.float32)

        y_latlon_raw = torch.tensor([float(r.lat), float(r.lon)], dtype=torch.float32)

        targets = {
            "landcover": y_landcover,
            "season": y_season,
            "latlon_norm": y_latlon_norm,
            "latlon_raw": y_latlon_raw,
            "image_path": r.image_path,
        }
        return x, targets

# ----------------------------
# Import Galileo encoder
# ----------------------------
def import_galileo(models_dir: str):
    import sys
    models_dir = str(Path(models_dir))
    if not Path(models_dir).exists():
        raise FileNotFoundError(f"--models_dir not found: {models_dir}")
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)
    from galileo import (
        Encoder as GalileoEncoder, SPACE_TIME_BANDS, SPACE_TIME_BANDS_GROUPS_IDX,
        SPACE_BANDS, SPACE_BAND_GROUPS_IDX, TIME_BANDS, TIME_BAND_GROUPS_IDX,
        STATIC_BANDS, STATIC_BAND_GROUPS_IDX
    )
    return {
        "GalileoEncoder": GalileoEncoder,
        "SPACE_TIME_BANDS": SPACE_TIME_BANDS,
        "SPACE_TIME_BANDS_GROUPS_IDX": SPACE_TIME_BANDS_GROUPS_IDX,
        "SPACE_BANDS": SPACE_BANDS,
        "SPACE_BAND_GROUPS_IDX": SPACE_BAND_GROUPS_IDX,
        "TIME_BANDS": TIME_BANDS,
        "TIME_BAND_GROUPS_IDX": TIME_BAND_GROUPS_IDX,
        "STATIC_BANDS": STATIC_BANDS,
        "STATIC_BAND_GROUPS_IDX": STATIC_BAND_GROUPS_IDX,
    }

# ----------------------------
# Galileo backbone -> feature map
# ----------------------------
class GalileoBackboneWrapper(nn.Module):
    def __init__(self, gal_ctx: Dict, pretrained_path: str, patch_size: int = 8):
        super().__init__()
        GalileoEncoder = gal_ctx["GalileoEncoder"]
        SPACE_TIME_BANDS = gal_ctx["SPACE_TIME_BANDS"]
        SPACE_TIME_BANDS_GROUPS_IDX = gal_ctx["SPACE_TIME_BANDS_GROUPS_IDX"]

        self.SPACE_TIME_BANDS = SPACE_TIME_BANDS
        self.SPACE_TIME_BANDS_GROUPS_IDX = SPACE_TIME_BANDS_GROUPS_IDX

        ckpt_dir = Path(pretrained_path)
        cfg = ckpt_dir / "config.json"
        if not cfg.exists():
            raise FileNotFoundError(f"config.json not found in --galileo_ckpt: {ckpt_dir}")

        self.encoder = GalileoEncoder.load_from_folder(ckpt_dir, device='cpu')
        self.out_channels = self.encoder.embedding_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)

        self.rgb_idx = list(self.SPACE_TIME_BANDS_GROUPS_IDX.keys()).index('S2_RGB')
        self.rgb_bands = [self.SPACE_TIME_BANDS.index(b) for b in ["B2", "B3", "B4"]]

        # cache band dicts used later
        self.SPACE_BANDS = gal_ctx["SPACE_BANDS"]
        self.SPACE_BAND_GROUPS_IDX = gal_ctx["SPACE_BAND_GROUPS_IDX"]
        self.TIME_BANDS = gal_ctx["TIME_BANDS"]
        self.TIME_BAND_GROUPS_IDX = gal_ctx["TIME_BAND_GROUPS_IDX"]
        self.STATIC_BANDS = gal_ctx["STATIC_BANDS"]
        self.STATIC_BAND_GROUPS_IDX = gal_ctx["STATIC_BAND_GROUPS_IDX"]

    @torch.no_grad()
    def _prep_tokens(self, x: torch.Tensor):
        b, _, h, w = x.shape
        dev, dt = x.device, x.dtype

        s_t_x = torch.zeros(b, h, w, 1, len(self.SPACE_TIME_BANDS), device=dev, dtype=dt)
        s_t_x[..., self.rgb_bands] = x.permute(0, 2, 3, 1).unsqueeze(-2)

        s_t_m = torch.ones(b, h, w, 1, len(self.SPACE_TIME_BANDS_GROUPS_IDX), device=dev, dtype=torch.long)
        s_t_m[..., self.rgb_idx] = 0

        sp_x = torch.zeros(b, h, w, len(self.SPACE_BANDS), device=dev, dtype=dt)
        t_x  = torch.zeros(b, 1, len(self.TIME_BANDS), device=dev, dtype=dt)
        st_x = torch.zeros(b, len(self.STATIC_BANDS), device=dev, dtype=dt)

        sp_m = torch.ones(b, h, w, len(self.SPACE_BAND_GROUPS_IDX), device=dev, dtype=torch.long)
        t_m  = torch.ones(b, 1, len(self.TIME_BAND_GROUPS_IDX), device=dev, dtype=torch.long)
        st_m = torch.ones(b, len(self.STATIC_BAND_GROUPS_IDX), device=dev, dtype=torch.long)

        months = torch.ones(b, 1, device=dev, dtype=torch.long) * 6
        return s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = self._prep_tokens(x)
        s_t_out, *_ = self.encoder(
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months,
            patch_size=self.patch_size, add_layernorm_on_exit=True
        )
        tokens = s_t_out[:, :, :, 0, self.rgb_idx, :]   # B,H,W,C
        fmap = tokens.permute(0, 3, 1, 2).contiguous()  # B,C,H,W
        return self.proj(fmap)

# ----------------------------
# Multi-task classifier: landcover, season, lat/lon regression
# ----------------------------
class GalileoMultiTask(nn.Module):
    def __init__(self, gal_ctx: Dict, galileo_ckpt: str,
                 num_landcover: int = 8, num_seasons: int = 4,
                 patch_size: int = 8, pool: str = "gem", freeze_backbone: bool = False):
        super().__init__()
        self.backbone = GalileoBackboneWrapper(gal_ctx, galileo_ckpt, patch_size)
        c = self.backbone.out_channels

        if pool not in {"avg", "max", "gem"}:
            raise ValueError("pool must be one of {avg, max, gem}")
        self.pool_type = pool
        if pool == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool == "max":
            self.pool = nn.AdaptiveMaxPool2d(1)
        else:
            self.p = nn.Parameter(torch.ones(1) * 3.0)  # GeM exponent

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.landcover_head = nn.Sequential(
            nn.Linear(c, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_landcover),
        )
        self.season_head = nn.Sequential(
            nn.Linear(c, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_seasons),
        )
        self.latlon_head = nn.Sequential(
            nn.Linear(c, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def gem_pool(self, x: torch.Tensor, eps: float = 1e-6):
        p = F.relu(self.p) + 1e-3
        x = x.clamp(min=eps).pow(p.unsqueeze(-1).unsqueeze(-1))
        x = x.mean(dim=(2, 3)).pow(1.0 / p)
        return x

    def forward(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        fmap = self.backbone(img)
        if self.pool_type in {"avg", "max"}:
            feat = self.pool(fmap).flatten(1)
        else:
            feat = self.gem_pool(fmap)
        return {
            "landcover_logits": self.landcover_head(feat),
            "season_logits": self.season_head(feat),
            "latlon_norm": self.latlon_head(feat),
        }

# ----------------------------
# Evaluation
# ----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             num_landcover: int, num_seasons: int,
             lat_mean: float, lat_std: float, lon_mean: float, lon_std: float):
    model.eval()
    n_lc_correct = n_se_correct = n = 0
    preds_rows = []

    cm_lc = torch.zeros(num_landcover, num_landcover, dtype=torch.long, device=device)
    cm_se = torch.zeros(num_seasons, num_seasons, dtype=torch.long, device=device)

    mae_lat = mae_lon = 0.0

    for x, tgt in loader:
        x = x.to(device)
        y_lc = tgt["landcover"].to(device)
        y_se = tgt["season"].to(device)
        y_latlon_norm = tgt["latlon_norm"].to(device)
        y_latlon_raw = tgt["latlon_raw"]  # CPU

        out = model(x)
        lc_logits = out["landcover_logits"]
        se_logits = out["season_logits"]
        latlon_pred_norm = out["latlon_norm"]

        lc_pred = lc_logits.argmax(1)
        se_pred = se_logits.argmax(1)

        n_lc_correct += (lc_pred == y_lc).sum().item()
        n_se_correct += (se_pred == y_se).sum().item()
        n += y_lc.numel()

        for t, p in zip(y_lc, lc_pred):
            cm_lc[t, p] += 1
        for t, p in zip(y_se, se_pred):
            cm_se[t, p] += 1

        lat_pred = (latlon_pred_norm[:, 0].cpu().numpy() * lat_std) + lat_mean
        lon_pred = (latlon_pred_norm[:, 1].cpu().numpy() * lon_std) + lon_mean

        lat_true = y_latlon_raw[:, 0].numpy()
        lon_true = y_latlon_raw[:, 1].numpy()

        mae_lat += np.abs(lat_pred - lat_true).sum()
        mae_lon += np.abs(lon_pred - lon_true).sum()

        for i in range(len(lat_true)):
            preds_rows.append({
                "image_path": tgt["image_path"][i],
                "landcover_true": int(y_lc[i].cpu().item()),
                "landcover_pred": int(lc_pred[i].cpu().item()),
                "season_true": int(y_se[i].cpu().item()),
                "season_pred": int(se_pred[i].cpu().item()),
                "lat_true": float(lat_true[i]),
                "lat_pred": float(lat_pred[i]),
                "lon_true": float(lon_true[i]),
                "lon_pred": float(lon_pred[i]),
            })

    lc_acc = n_lc_correct / max(n, 1)
    se_acc = n_se_correct / max(n, 1)
    mae_lat /= max(n, 1)
    mae_lon /= max(n, 1)

    return {
        "lc_acc": lc_acc,
        "se_acc": se_acc,
        "mae_lat": mae_lat,
        "mae_lon": mae_lon,
        "cm_lc": cm_lc.cpu().numpy(),
        "cm_se": cm_se.cpu().numpy(),
        "rows": preds_rows
    }

# ----------------------------
# Train
# ----------------------------
def train_one_epoch(model, loader, opt, device, crit_lc, crit_se, crit_ll,
                    w_lc: float, w_se: float, w_ll: float):
    model.train()
    total_loss = 0.0
    for x, tgt in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y_lc = tgt["landcover"].to(device)
        y_se = tgt["season"].to(device)
        y_ll = tgt["latlon_norm"].to(device)

        out = model(x)
        loss_lc = crit_lc(out["landcover_logits"], y_lc)
        loss_se = crit_se(out["season_logits"], y_se)
        loss_ll = crit_ll(out["latlon_norm"], y_ll)

        loss = w_lc * loss_lc + w_se * loss_se + w_ll * loss_ll

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--test_csv", type=str, required=True)
    ap.add_argument("--models_dir", type=str, required=True,
                    help="Path to folder containing galileo.py (…/models)")
    ap.add_argument("--galileo_ckpt", type=str, required=True,
                    help="Folder containing Galileo config.json + weights")
    ap.add_argument("--out_dir", type=str, default="runs_multitask")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--pool", type=str, default="gem", choices=["avg", "max", "gem"])
    ap.add_argument("--patch_size", type=int, default=8)
    ap.add_argument("--num_landcover", type=int, default=8)
    ap.add_argument("--num_seasons", type=int, default=4)
    ap.add_argument("--w_landcover", type=float, default=1.0)
    ap.add_argument("--w_season", type=float, default=0.5)
    ap.add_argument("--w_latlon", type=float, default=0.25)
    ap.add_argument("--latlon_loss", type=str, default="smoothl1", choices=["l1", "smoothl1", "mse"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # import galileo symbols
    gal_ctx = import_galileo(args.models_dir)

    # datasets (test uses train stats for normalization)
    train_ds = LandcoverCSV(args.train_csv, img_size=args.img_size)
    test_ds  = LandcoverCSV(
        args.test_csv, img_size=args.img_size,
        lat_mean=train_ds.lat_mean, lat_std=train_ds.lat_std,
        lon_mean=train_ds.lon_mean, lon_std=train_ds.lon_std
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = GalileoMultiTask(
        gal_ctx=gal_ctx,
        galileo_ckpt=args.galileo_ckpt,
        num_landcover=args.num_landcover,
        num_seasons=args.num_seasons,
        patch_size=args.patch_size,
        pool=args.pool,
        freeze_backbone=args.freeze_backbone
    ).to(device)

    # landcover class weights (ensure length == num_landcover)
    counts = train_ds.df["landcover_id"].value_counts().sort_index()
    counts = counts.reindex(range(args.num_landcover), fill_value=0)
    inv = counts.sum() / (counts + 1e-6)
    inv[counts == 0] = 0.0  # no gradient for truly absent classes
    nz = inv[inv > 0]
    if len(nz) > 0:
        inv = inv / nz.mean()
    lc_w = torch.tensor(inv.values, dtype=torch.float32, device=device)

    crit_lc = nn.CrossEntropyLoss(weight=lc_w)
    crit_se = nn.CrossEntropyLoss()

    if args.latlon_loss == "l1":
        crit_ll = nn.L1Loss()
    elif args.latlon_loss == "mse":
        crit_ll = nn.MSELoss()
    else:
        crit_ll = nn.SmoothL1Loss(beta=0.5)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model, train_loader, opt, device, crit_lc, crit_se, crit_ll,
            args.w_landcover, args.w_season, args.w_latlon
        )

        eval_out = evaluate(
            model, test_loader, device,
            num_landcover=args.num_landcover, num_seasons=args.num_seasons,
            lat_mean=test_ds.lat_mean, lat_std=test_ds.lat_std,
            lon_mean=test_ds.lon_mean, lon_std=test_ds.lon_std
        )

        composite = 0.5 * (eval_out["lc_acc"] + eval_out["se_acc"]) - 0.5 * ((eval_out["mae_lat"] + eval_out["mae_lon"]) / 2.0)

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "loss": loss,
                "metrics": eval_out,
                "args": vars(args),
            },
            os.path.join(args.out_dir, f"checkpoint_{epoch:03d}.pt")
        )
        if composite > best_score:
            best_score = composite
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))

        print(
            f"epoch {epoch}  loss {loss:.4f}  "
            f"lc_acc {eval_out['lc_acc']:.4f}  se_acc {eval_out['se_acc']:.4f}  "
            f"mae_lat {eval_out['mae_lat']:.5f}  mae_lon {eval_out['mae_lon']:.5f}  "
            f"best {best_score:.4f}"
        )

    final_eval = evaluate(
        model, test_loader, device,
        num_landcover=args.num_landcover, num_seasons=args.num_seasons,
        lat_mean=test_ds.lat_mean, lat_std=test_ds.lat_std,
        lon_mean=test_ds.lon_mean, lon_std=test_ds.lon_std
    )
    pd.DataFrame(final_eval["cm_lc"].astype(int)).to_csv(os.path.join(args.out_dir, "cm_landcover.csv"), index=False)
    pd.DataFrame(final_eval["cm_se"].astype(int)).to_csv(os.path.join(args.out_dir, "cm_season.csv"), index=False)
    pd.DataFrame(final_eval["rows"]).to_csv(os.path.join(args.out_dir, "test_predictions.csv"), index=False)

    id_to_class = {
        0: "Tree cover",
        1: "Shrubland",
        2: "Grassland",
        3: "Cropland",
        4: "Built-up",
        5: "Bare or sparse vegetation",
        6: "Permanent water bodies",
        7: "Herbaceous wetland",
    }
    with open(os.path.join(args.out_dir, "id_to_class.json"), "w") as f:
        json.dump(id_to_class, f, indent=2)

if __name__ == "__main__":
    """
    Example:
    python gallio_classification.py \
      --train_csv classified_dataset/train_labels.csv \
      --test_csv  classified_dataset/test_labels.csv  \
      --models_dir /home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/iclr_2026/models \
      --galileo_ckpt /home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/iclr_2026/nano \
      --epochs 50 --batch_size 32 --img_size 800 --pool gem \
      --w_landcover 1.0 --w_season 0.5 --w_latlon 0.25 --latlon_loss smoothl1
    """
    main()
