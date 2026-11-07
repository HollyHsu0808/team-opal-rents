"""Match every example record in the organiser OpenAPI spec against our public-data layers.

Pulls every object that carries lat/lng (and/or gnaf_pid, address, mb) out of the spec's examples,
then checks: official suburb polygon, nearest OSM building footprint (and whether it is in our
building table), LiDAR height at the point, mesh-block code against our canopy / heat layers,
and the address format. Writes _derived/api_example_match.csv.
"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from shapely.geometry import Point, Polygon

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "_derived"
spec = json.load(open(ROOT / "organiser" / "openapi.json", encoding="utf-8"))

recs = []
def walk(o, path):
    if isinstance(o, dict):
        keys = set(o)
        if ({"lat", "lng"} <= keys or "gnaf_pid" in keys or "mb" in keys) and not any(isinstance(o.get(k), (dict, list)) for k in ("lat", "lng")):
            recs.append(dict(endpoint=path, **{k: o.get(k) for k in ("gnaf_pid", "address", "lat", "lng", "mb", "floor", "level",
                                                                    "flats_in_building", "bedrooms", "car_spaces", "property_type")}))
        for k, v in o.items():
            walk(v, path)
    elif isinstance(o, list):
        for v in o:
            walk(v, path)

for p, ops in spec.get("paths", {}).items():
    walk(ops, p)
for name, comp in spec.get("components", {}).get("schemas", {}).items():
    walk(comp.get("example", {}), f"schema:{name}")
ex = pd.DataFrame(recs).map(lambda v: json.dumps(v) if isinstance(v, (dict, list)) else v)
ex["lat"] = pd.to_numeric(ex.lat, errors="coerce"); ex["lng"] = pd.to_numeric(ex.lng, errors="coerce")
ex = ex.drop_duplicates(subset=["gnaf_pid", "address", "lat", "lng", "mb"])
print(f"example records with location/id fields: {len(ex)}  (with lat/lng: {ex.lat.notna().sum()}, with mb: {ex.mb.notna().sum()})")

# address formats
def fmt(a):
    if not isinstance(a, str): return None
    return re.sub(r"[A-Z]+", "W", re.sub(r"\d+", "9", a))
ex["address_format"] = ex.address.map(fmt)

g = ex[ex.lat.notna() & ex.lng.notna()].copy()
g = gpd.GeoDataFrame(g, geometry=gpd.points_from_xy(g.lng.astype(float), g.lat.astype(float)), crs=4326)
sub = gpd.read_file(ROOT / "boundary" / "cos_29_suburbs.geojson").to_crs(4326)
g = gpd.sjoin(g, sub[["name", "geometry"]].rename(columns={"name": "official_suburb_polygon"}), how="left", predicate="within").drop(columns="index_right")
g = g[~g.index.duplicated()]

# footprints
b = pd.read_parquet(D / "buildings_features_v2.parquet")
osm = json.load(open(ROOT / "osm" / "osm_buildings_bbox.json", encoding="utf-8"))["elements"]
fp = gpd.GeoDataFrame([dict(osm_id=e["id"], geometry=Polygon([(q["lon"], q["lat"]) for q in e["geometry"]]))
                       for e in osm if e.get("type") == "way" and len(e.get("geometry", [])) >= 4], crs=4326).to_crs(7856)
fp["in_table"] = fp.osm_id.isin(b.loc[b.osm_type == "way", "osm_id"])
gm = g.to_crs(7856)
near = gpd.sjoin_nearest(gm, fp, how="left", distance_col="d_footprint_m")
near = near[~near.index.duplicated()]
g["nearest_osm_id"] = near.osm_id; g["d_footprint_m"] = near.d_footprint_m.round(1); g["footprint_in_table"] = near.in_table
# second-nearest within 10 m (ambiguity from 4-dp coordinates)
amb = []
for i, p in gm.geometry.items():
    d = fp.distance(p)
    amb.append(int((d <= 10).sum()))
g["footprints_within_10m"] = amb

# LiDAR height at point (max nDSM within 8 m)
with rasterio.open(D / "lidar_rasters" / "ndsm_2020.tif") as src:
    pts = g.to_crs(src.crs)
    hh = []
    for p in pts.geometry:
        r, c = src.index(p.x, p.y)
        w = rasterio.windows.Window(c - 8, r - 8, 17, 17)
        a = src.read(1, window=w, boundless=True, fill_value=np.nan)
        hh.append(float(np.nanmax(a)) if np.isfinite(a).any() else np.nan)
g["lidar_h_at_point_m"] = np.round(hh, 1)

# mesh block codes vs our layers
can = gpd.read_file(ROOT / "canopy" / "canopy2022_mmb_bbox.geojson")
uhi = gpd.read_file(ROOT / "heat" / "uhi2016_mb_bbox.geojson")
def codes(df):
    s = set()
    for c in df.columns:
        if df[c].dtype == object:
            v = df[c].astype(str)
            if v.str.fullmatch(r"\d{11}").mean() > 0.5:
                s |= set(v); print("  mesh-block-like column:", c)
    return s
print("canopy layer:"); can_codes = codes(can)
print("uhi 2016 layer:"); uhi_codes = codes(uhi)
ex["mb_in_canopy2022"] = ex.mb.astype(str).isin(can_codes).where(ex.mb.notna())
ex["mb_in_uhi2016"] = ex.mb.astype(str).isin(uhi_codes).where(ex.mb.notna())

out = ex.merge(g.drop(columns="geometry")[["gnaf_pid", "address", "lat", "lng", "official_suburb_polygon", "nearest_osm_id",
                                           "d_footprint_m", "footprint_in_table", "footprints_within_10m", "lidar_h_at_point_m"]],
               on=["gnaf_pid", "address", "lat", "lng"], how="left")
out.to_csv(D / "api_example_match.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 48)
print(out[["endpoint", "address", "lat", "lng", "mb", "official_suburb_polygon", "d_footprint_m", "footprint_in_table",
           "footprints_within_10m", "lidar_h_at_point_m", "floor", "mb_in_canopy2022", "mb_in_uhi2016"]].to_string(index=False))
print("\naddress formats:"); print(ex.address_format.value_counts().to_string())
