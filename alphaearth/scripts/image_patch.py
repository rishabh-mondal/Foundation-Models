#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point

# =======================
# CONFIG
# =======================
BASE = Path("/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data")
PATCH_META_DIR = Path("/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/processed_data/sentinel/patch_metadata")

META_FILES = {
    "uttar_pradesh": PATCH_META_DIR / "uttar_pradesh_patch_centers.geojson",
    "pak_punjab":    PATCH_META_DIR / "pak_punjab_metadata.geojson",
    "bangladesh":    PATCH_META_DIR / "bangladesh_metadata.geojson",
}

SPLITS = ["train", "val", "test"]
IMG_EXT = ".png"  # your patch images

# =======================
# Helpers
# =======================
def parse_patch_id(stem: str):
    """
    Parse 'lat_lon' from filename stem like '25.2879_80.4283' -> (lat, lon)
    """
    lat_str, lon_str = stem.split("_")
    return round(float(lat_str), 4), round(float(lon_str), 4)

def load_all_metadata() -> gpd.GeoDataFrame:
    """
    Read all state metadata, ensure WGS84, and add rounding cols:
    lat4, lon4 that match your filename precision (.4f).
    """
    frames = []
    for state, path in META_FILES.items():
        gdf = gpd.read_file(path)
        # Ensure CRS
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)

        # We expect lat_center/lon_center in the metadata you produced
        if not {"lat_center", "lon_center"}.issubset(gdf.columns):
            raise ValueError(f"{path} must have 'lat_center' and 'lon_center' columns.")

        gdf["lat4"] = gdf["lat_center"].round(4)
        gdf["lon4"] = gdf["lon_center"].round(4)
        gdf["state"] = state
        frames.append(gdf[["state", "lat_center", "lon_center", "lat4", "lon4", "geometry"]])

    all_meta = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(all_meta, geometry="geometry", crs="EPSG:4326")

def build_index(meta_gdf: gpd.GeoDataFrame):
    """
    Build a dictionary {(state, lat4, lon4) -> geometry}
    """
    idx = {}
    for row in meta_gdf.itertuples(index=False):
        idx[(row.state, row.lat4, row.lon4)] = row.geometry
    return idx

def image_iter(base: Path):
    """
    Yield tuples (state, split, img_path, patch_id, lat4, lon4)
    for every .png image under final_data/<state>/<split>/images
    """
    for state_dir in sorted([d for d in base.iterdir() if d.is_dir()]):
        state = state_dir.name  # 'uttar_pradesh' / 'pak_punjab' / 'bangladesh'
        for split in SPLITS:
            img_dir = state_dir / split / "images"
            if not img_dir.exists():
                continue
            for p in img_dir.glob(f"*{IMG_EXT}"):
                stem = p.stem
                try:
                    lat4, lon4 = parse_patch_id(stem)
                except Exception:
                    # Skip files that don't match 'lat_lon.png'
                    continue
                patch_id = f"{lat4:.4f}_{lon4:.4f}"
                yield state, split, p, patch_id, lat4, lon4

def build_patch_index_gdf() -> gpd.GeoDataFrame:
    """
    Main builder: returns GeoDataFrame with [state, split, patch_id, geometry]
    """
    meta = load_all_metadata()
    idx = build_index(meta)

    records = []
    missing = 0

    for state, split, path, patch_id, lat4, lon4 in image_iter(BASE):
        geom = idx.get((state, lat4, lon4))
        if geom is None:
            missing += 1
        records.append({
            "state": state,
            "split": split,
            "patch_id": patch_id,
            "geometry": geom
        })

    if missing:
        print(f"⚠️  {missing} images had no geometry match (state + lat4/lon4).")

    gdf = gpd.GeoDataFrame(pd.DataFrame.from_records(records), geometry="geometry", crs="EPSG:4326")
    return gdf

def get_geometry_for_image(img_path: str | Path, meta_index=None):
    """
    Fetch the polygon geometry for a single image path like:
    /.../final_data/uttar_pradesh/train/images/25.2879_80.4283.png

    Usage:
        geom = get_geometry_for_image("/full/path/to/25.2879_80.4283.png")
    """
    img_path = Path(img_path)
    state = img_path.parents[2].name  # .../<state>/<split>/images/<file>
    stem = img_path.stem
    lat4, lon4 = parse_patch_id(stem)

    if meta_index is None:
        meta_gdf = load_all_metadata()
        meta_index = build_index(meta_gdf)

    return meta_index.get((state, lat4, lon4))  # shapely Polygon or None

# =======================
# Run
# =======================
if __name__ == "__main__":
    gdf = build_patch_index_gdf()
    out_geojson = BASE / "patch_index_from_latlon_names.geojson"
    out_csv    = BASE / "patch_index_from_latlon_names.csv"

    gdf.to_file(out_geojson, driver="GeoJSON")
    gdf_csv = gdf.copy()
    gdf_csv["geometry"] = gdf_csv["geometry"].apply(lambda g: g.wkt if g is not None else None)
    gdf_csv.to_csv(out_csv, index=False)

    print(f"✅ Saved:\n  {out_geojson}\n  {out_csv}")

    # Example single-lookup (your example path)
    example = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/uttar_pradesh/train/images/25.2879_80.4283.png"
    geom = get_geometry_for_image(example)
    print("Example geometry:", "FOUND" if geom is not None else "NOT FOUND")