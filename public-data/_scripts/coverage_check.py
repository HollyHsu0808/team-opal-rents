"""Check that every public-data layer covers the organiser's 29 City of Sydney suburbs.

Writes public-data/_derived/coverage_check.csv (one row per layer) and
public-data/boundary/cos_29_suburbs.geojson (suburb polygons, OSM, ODbL).
"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from shapely.geometry import LineString, box, Point
from shapely.ops import polygonize, unary_union
from rasterio.mask import mask as rmask

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "_derived"
OFFICIAL = ["Alexandria", "Barangaroo", "Beaconsfield", "Camperdown", "Centennial Park", "Chippendale",
            "Darlinghurst", "Darlington", "Dawes Point", "Elizabeth Bay", "Erskineville", "Eveleigh",
            "Forest Lodge", "Glebe", "Haymarket", "Millers Point", "Moore Park", "Potts Point", "Pyrmont",
            "Redfern", "Rosebery", "Rushcutters Bay", "Surry Hills", "Sydney", "The Rocks", "Ultimo",
            "Waterloo", "Woolloomooloo", "Zetland"]
BBOX = box(151.170, -33.925, 151.235, -33.855)

# ---- suburb polygons: NSW Spatial Services (Administrative Boundaries, layer 2 Suburb; CC BY)
sub = gpd.read_file(ROOT / "boundary" / "nsw_suburbs_cos_area.geojson").to_crs(4326)
sub["name"] = sub.suburbname.str.title().str.replace("Mcmahons", "McMahons")
found = sub[sub.name.isin(OFFICIAL)].dissolve("name").reset_index()
missing_names = sorted(set(OFFICIAL) - set(found.name))
lga = gpd.read_file(ROOT / "boundary" / "nsw_lga_sydney.geojson").to_crs(4326)
lga = lga[lga.lganame == "SYDNEY"]
print("suburbs found:", len(found), "missing:", missing_names)
L = unary_union(lga.geometry)
Um = gpd.GeoSeries([unary_union(found.geometry)], crs=4326).to_crs(7856).iloc[0]
Lm = gpd.GeoSeries([L], crs=4326).to_crs(7856).iloc[0]
print(f"29 suburbs vs LGA: suburbs area {Um.area/1e6:.1f} km2, LGA {Lm.area/1e6:.1f} km2, "
      f"suburb area outside LGA {Um.difference(Lm).area/1e6:.2f} km2, LGA area outside suburbs {Lm.difference(Um).area/1e6:.2f} km2")
for _, r in found.iterrows():
    g = gpd.GeoSeries([r.geometry], crs=4326).to_crs(7856).iloc[0]
    sh = g.intersection(Lm).area / g.area
    if sh < 0.98:
        print(f"  suburb partly outside the LGA: {r['name']} share inside {sh:.2f}")
found[["name", "geometry"]].to_file(ROOT / "boundary" / "cos_29_suburbs.geojson", driver="GeoJSON")
U = unary_union(found.geometry)
U_m = gpd.GeoSeries([U], crs=4326).to_crs(7856).iloc[0]
print(f"29-suburb area {U_m.area/1e6:.1f} km2")

out = []
def rec(layer, what, value, note=""):
    out.append(dict(layer=layer, check=what, value=value, note=note)); print(f"{layer:28s} {what:45s} {value}  {note}")

# ---- fetch bbox
rec("fetch bbox (OSM, cadastre, heat, canopy)", "share of 29-suburb area inside bbox",
    round(gpd.GeoSeries([U.intersection(BBOX)], crs=4326).to_crs(7856).area.iloc[0] / U_m.area, 4))
outside = found.assign(out_km2=gpd.GeoSeries(found.geometry.difference(BBOX), crs=4326).to_crs(7856).area / 1e6)
for _, r in outside[outside.out_km2 > 0.01].iterrows():
    rec("fetch bbox", f"area outside bbox: {r['name']}", round(r.out_km2, 3), "km2")

# ---- buildings table
b = pd.read_parquet(D / "buildings_features_v2.parquet")
pts = gpd.GeoDataFrame(b, geometry=gpd.points_from_xy(b.lon, b.lat), crs=4326)
j = gpd.sjoin(pts, found[["name", "geometry"]].rename(columns={"name": "official_suburb"}), how="left", predicate="within")
j = j[~j.index.duplicated()]
rec("buildings_features_v2", "buildings total", len(b))
rec("buildings_features_v2", "inside the 29 suburbs", int(j.official_suburb.notna().sum()))
rec("buildings_features_v2", "outside (extra context, not scored)", int(j.official_suburb.isna().sum()))
per = j.official_suburb.value_counts()
for s in OFFICIAL:
    rec("buildings_features_v2", f"buildings in {s}", int(per.get(s, 0)))
j[["osm_type", "osm_id", "official_suburb"]].to_csv(D / "buildings_official_suburb.csv", index=False)

# ---- LiDAR rasters: data coverage inside the 29 suburbs
for y in (2013, 2020):
    p = D / "lidar_rasters" / f"dsm_{y}.tif"
    with rasterio.open(p) as src:
        g = gpd.GeoSeries([U], crs=4326).to_crs(src.crs).iloc[0]
        rb = box(*src.bounds)
        rec(f"lidar dsm_{y}", "share of 29-suburb area inside raster extent", round(g.intersection(rb).area / g.area, 4))
        arr, _ = rmask(src, [g], crop=True, filled=False)
        a = arr[0]
        valid = (~a.mask) & np.isfinite(a.data) & (a.data > -100)
        inside = ~a.mask
        rec(f"lidar dsm_{y}", "share of 1 m cells inside 29 suburbs with data", round(float(valid.sum() / max(inside.sum(), 1)), 4))
        for _, r in found.iterrows():
            gg = gpd.GeoSeries([r.geometry], crs=4326).to_crs(src.crs).iloc[0]
            share = gg.intersection(rb).area / gg.area
            if share < 0.999:
                rec(f"lidar dsm_{y}", f"suburb not fully in extent: {r['name']}", round(share, 4))

# ---- other layers (bbox-clipped downloads)
def vec(layer, path):
    g = gpd.read_file(path)
    if g.crs is None: g = g.set_crs(4326)
    ext = box(*g.to_crs(4326).total_bounds)
    rec(layer, "share of 29-suburb area inside layer extent", round(gpd.GeoSeries([U.intersection(ext)], crs=4326).to_crs(7856).area.iloc[0] / U_m.area, 4), f"{len(g)} features")
vec("cadastre sp_lots", ROOT / "cadastre" / "sp_lots_bbox.geojson")
vec("canopy 2022 mesh blocks", ROOT / "canopy" / "canopy2022_mmb_bbox.geojson")
vec("heat HVI 2022 SA1", ROOT / "heat" / "hvi2022_sa1_bbox.geojson")
vec("heat UHI 2016 mesh blocks", ROOT / "heat" / "uhi2016_mb_bbox.geojson")
with rasterio.open(ROOT / "heat" / "lst_sydney_2022-2023_summer_bbox.tif") as src:
    g = gpd.GeoSeries([U], crs=4326).to_crs(src.crs).iloc[0]
    rec("heat LST raster", "share of 29-suburb area inside raster extent", round(g.intersection(box(*src.bounds)).area / g.area, 4))

# ---- bond: postcodes of the 29 suburbs present?
POST = {"2000", "2007", "2008", "2009", "2010", "2011", "2015", "2016", "2017", "2018", "2021", "2037", "2042", "2043", "2050"}
bp = set()
for f in sorted((ROOT / "bond").glob("*.xlsx")):
    try:
        x = pd.read_excel(f, header=None, nrows=4000, dtype=str)
        bp |= set(re.findall(r"\b20\d\d\b", " ".join(x.astype(str).values.ravel()))) & POST
    except Exception as ex:
        rec("bond", f"read error {f.name}", str(ex)[:60])
rec("bond", "CoS postcodes found (of 15)", len(bp), ",".join(sorted(bp)))

pd.DataFrame(out).to_csv(D / "coverage_check.csv", index=False)
print("wrote", D / "coverage_check.csv")
