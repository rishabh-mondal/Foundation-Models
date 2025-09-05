#!/usr/bin/env python3
"""
Download AlphaEarth Foundations (64-band) patch embeddings per split.

Creates per-split directories:
  final_data/<state>/<split>/embeddings/<lat>_<lon>.tif

Inputs:
  - final_data/patch_index_from_latlon_names.geojson
    with columns: state, split, patch_id, geometry (Polygon, EPSG:4326)
"""

from pathlib import Path
import time
import math
import io
import json
import requests

import ee
import geopandas as gpd
import rasterio
from rasterio.io import MemoryFile

# =======================
# CONFIG
# =======================
BASE = Path("/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data")
PATCH_INDEX = BASE / "patch_index_from_latlon_names.geojson"  # produced earlier
YEAR_START, YEAR_END = "2024-01-01", "2025-01-01"
SCALE_M = 10
BANDS_COLLECTION_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"  # AlphaEarth Foundations annual collection
MAX_RETRIES = 3
SLEEP_BETWEEN = 0.5  # seconds, between requests

# =======================
# EE init
# =======================
def init_ee():
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

# =======================
# Helpers
# =======================
def utm_epsg_from_lon(lat: float, lon: float) -> str:
    """
    Return EPSG code string for UTM zone of given lon/lat (Northern hemisphere => 326xx).
    """
    zone = int(math.floor((lon + 180) / 6) + 1)
    if lat >= 0:
        return f"EPSG:{32600 + zone}"  # northern
    else:
        return f"EPSG:{32700 + zone}"  # southern (not used here, but safe)

def polygon_to_ee_region(poly):
    """
    Convert shapely Polygon (EPSG:4326) to EE region format (list of [lon, lat]).
    """
    coords = list(poly.exterior.coords)
    # EE expects [ [lon, lat], ...]
    return [[float(x), float(y)] for (x, y) in coords]

def build_image():
    return (ee.ImageCollection(BANDS_COLLECTION_ID)
            .filterDate(YEAR_START, YEAR_END)
            .mosaic()
            .toFloat())

def download_patch(img, region_coords, crs_epsg, scale, timeout_s=600):
    """
    Request a multi-band GeoTIFF for a region from EE and return raw bytes.
    Retries with exponential backoff.
    """
    params = {
        "region": json.dumps(region_coords),
        "scale": scale,
        "crs": crs_epsg,
        "filePerBand": False,
        "format": "GEO_TIFF"
    }
    # get URL
    url = img.getDownloadURL(params)
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=timeout_s)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_exc = e
            wait = min(2 ** attempt, 10)
            print(f"  ⚠️  download failed (attempt {attempt}/{MAX_RETRIES}): {e}; retrying in {wait}s…")
            time.sleep(wait)
    raise RuntimeError(f"Failed to download after {MAX_RETRIES} attempts: {last_exc}")

def ensure_gdf():
    if not PATCH_INDEX.exists():
        raise FileNotFoundError(f"Patch index not found: {PATCH_INDEX}\n"
                                f"Run the earlier script to create this GeoJSON.")
    gdf = gpd.read_file(PATCH_INDEX)
    # Ensure CRS is WGS84
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    # Basic sanity
    needed = {"state", "split", "patch_id", "geometry"}
    missing = needed - set(gdf.columns)
    if missing:
        raise ValueError(f"Patch index missing columns: {missing}")
    # Drop missing geometry rows
    gdf = gdf.dropna(subset=["geometry"]).reset_index(drop=True)
    return gdf

def save_tif_bytes(raw_bytes, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(raw_bytes)

# =======================
# Main
# =======================
def main():
    init_ee()
    gdf = ensure_gdf()
    img = build_image()

    total = len(gdf)
    print(f"Found {total} patches with geometry. Starting downloads…")

    done = 0
    skipped = 0
    failed = 0

    # Process grouped by state/split for nice directory layout
    for (state, split), sub in gdf.groupby(["state", "split"], sort=True):
        out_dir = BASE / state / split / "embeddings"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n== {state} / {split} -> {len(sub)} patches ==")

        for row in sub.itertuples(index=False):
            patch_id = row.patch_id  # '<lat>_<lon>'
            out_path = out_dir / f"{patch_id}.tif"
            if out_path.exists():
                skipped += 1
                continue

            # Derive center for UTM selection (from filename)
            try:
                lat_str, lon_str = patch_id.split("_")
                lat = float(lat_str)
                lon = float(lon_str)
            except Exception:
                # fallback to geometry centroid
                lat = float(row.geometry.centroid.y)
                lon = float(row.geometry.centroid.x)

            crs_epsg = utm_epsg_from_lon(lat, lon)
            region_coords = polygon_to_ee_region(row.geometry)

            try:
                raw = download_patch(img, region_coords, crs_epsg, SCALE_M)
                save_tif_bytes(raw, out_path)
                # quick validate
                with rasterio.open(out_path) as src:
                    if src.count != 64:
                        print(f"  ⚠️  {out_path.name}: expected 64 bands, got {src.count}")
                done += 1
            except Exception as e:
                failed += 1
                print(f"  ❌  {patch_id}: {e}")

            time.sleep(SLEEP_BETWEEN)

        print(f"  -> saved: {done}, skipped: {skipped}, failed: {failed}")

    print(f"\n✅ Finished. Saved: {done} | Skipped (exists): {skipped} | Failed: {failed}")

if __name__ == "__main__":
    main()