"""Build public-data/_derived/buildings_features.parquet (+ .csv) - one row per OSM residential building in the bbox.
Run with the rents-heat venv (shapely, pyproj, rasterio, scipy):
    rents-heat/.venv/Scripts/python.exe public-data/_scripts/build_features.py

Residential rule (documented choice):
  res_rule = 'apt'     building in {apartments, residential, flats}
             'yes_ge3' building = yes AND levels >= 3 AND no non-residential tag (shop, office, amenity, tourism,
                       craft, healthcare, leisure, building:use in commercial/retail/office) -> may still be offices
                       in the CBD; kept but flagged so the user can drop them.
  Houses/terraces are excluded (the task is flats).
levels = building:levels (numeric part); if missing and height present, round(height/3.1) (levels_src='height').
All distances computed in EPSG:7856 (GDA2020 / MGA zone 56), metres.
"""
import json
import warnings
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import shapely
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, Polygon, shape
from shapely.ops import polygonize, transform, unary_union

warnings.filterwarnings("ignore", category=DeprecationWarning)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_derived"
OUT.mkdir(exist_ok=True)
TO_M = Transformer.from_crs("EPSG:4326", "EPSG:7856", always_xy=True).transform
NONRES = ("shop", "office", "amenity", "tourism", "craft", "healthcare", "leisure")


def num(s):
    if s is None:
        return np.nan
    m = re.search(r"\d+(\.\d+)?", str(s).replace(",", "."))
    return float(m.group()) if m else np.nan


def osm_polys():
    d = json.load(open(ROOT / "osm" / "osm_buildings_bbox.json", encoding="utf-8"))
    rel_path = ROOT / "osm" / "osm_building_relations_bbox.json"
    rels = {}
    if rel_path.exists():
        for e in json.load(open(rel_path, encoding="utf-8"))["elements"]:
            rels[e["id"]] = e
    rows = []
    for e in d["elements"]:
        t = e.get("tags", {})
        g = None
        if e["type"] == "way" and len(e.get("geometry", [])) >= 4:
            g = Polygon([(p["lon"], p["lat"]) for p in e["geometry"]])
        elif e["type"] == "relation" and e["id"] in rels:
            lines = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]]) for m in rels[e["id"]].get("members", [])
                     if m.get("role") == "outer" and len(m.get("geometry", [])) >= 2]
            polys = list(polygonize(unary_union(lines))) if lines else []
            if polys:
                g = unary_union(polys)
        if g is None or g.is_empty:
            continue
        if not g.is_valid:
            g = g.buffer(0)
        rows.append(dict(osm_type=e["type"], osm_id=e["id"], tags=t, geom=g))
    print(len(rows), "building polygons (", sum(r["osm_type"] == "relation" for r in rows), "relations)")
    return rows, d.get("osm3s", {}).get("timestamp_osm_base")


def main():
    t0 = time.time()
    rows, osm_ts = osm_polys()
    df = pd.DataFrame(dict(osm_type=[r["osm_type"] for r in rows], osm_id=[r["osm_id"] for r in rows]))
    tags = [r["tags"] for r in rows]
    df["building"] = [t.get("building") for t in tags]
    lv = np.array([num(t.get("building:levels")) for t in tags])
    ht = np.array([num(t.get("height")) for t in tags])
    df["levels_tag"] = lv
    df["height_m"] = ht
    df["levels"] = np.where(np.isnan(lv), np.round(ht / 3.1), lv)
    df["levels_src"] = np.where(~np.isnan(lv), "building:levels", np.where(~np.isnan(ht), "height", None))
    df["addr_housenumber"] = [t.get("addr:housenumber") for t in tags]
    df["addr_street"] = [t.get("addr:street") for t in tags]
    df["addr_suburb"] = [t.get("addr:suburb") or t.get("addr:city") for t in tags]
    df["addr_postcode"] = [t.get("addr:postcode") for t in tags]
    df["name"] = [t.get("name") for t in tags]
    df["start_date_osm"] = [t.get("start_date") for t in tags]
    nonres = np.array([any(k in t for k in NONRES) or t.get("building:use") in ("commercial", "retail", "office")
                       for t in tags])
    geoms_ll = [r["geom"] for r in rows]
    geoms_m = [transform(TO_M, g) for g in geoms_ll]
    cen_m = np.array([[g.centroid.x, g.centroid.y] for g in geoms_m])
    cen_ll = [g.centroid for g in geoms_ll]
    df["lon"] = [c.x for c in cen_ll]
    df["lat"] = [c.y for c in cen_ll]
    df["footprint_m2"] = [g.area for g in geoms_m]

    # neighbour max levels within 100 m (all buildings with known levels, excluding self)
    tree = cKDTree(cen_m)
    has = ~np.isnan(df.levels.values)
    lvv = df.levels.values
    nmax = np.full(len(df), np.nan)
    for i, nb in enumerate(tree.query_ball_point(cen_m, r=100.0)):
        vals = [lvv[j] for j in nb if j != i and has[j]]
        if vals:
            nmax[i] = max(vals)
    df["neighbour_max_levels_100m"] = nmax
    df["n_buildings_100m"] = [len(nb) - 1 for nb in tree.query_ball_point(cen_m, r=100.0)]

    # residential selection
    b = df.building.values
    is_apt = np.isin(b, ["apartments", "residential", "flats"])
    is_yes3 = (b == "yes") & (df.levels.values >= 3) & ~nonres
    df["res_rule"] = np.where(is_apt, "apt", np.where(is_yes3, "yes_ge3", None))
    sel = df.res_rule.notna().values
    print("residential:", sel.sum(), "of", len(df), pd.Series(df.res_rule[sel]).value_counts().to_dict())
    R = df[sel].reset_index(drop=True)
    Rg_m = [geoms_m[i] for i in np.where(sel)[0]]
    Rc_m = shapely.points(cen_m[sel])

    # strata plans
    sp = json.load(open(ROOT / "cadastre" / "sp_lots_bbox.geojson", encoding="utf-8"))["features"]
    sp_geom = [transform(TO_M, shape(f["geometry"]).buffer(0)) for f in sp]
    sp_pn = np.array([f["properties"]["plannumber"] for f in sp])
    sp_cd = pd.to_datetime([f["properties"]["createdate"] for f in sp], unit="ms")
    st = shapely.STRtree(sp_geom)
    pi, gi = st.query(Rc_m, predicate="within")
    first = pd.Series(gi, index=pi).groupby(level=0).first()
    R["sp_plannumber"] = pd.Series(sp_pn[first.values], index=first.index).reindex(range(len(R))).values
    R["sp_createdate"] = pd.Series(sp_cd[first.values], index=first.index).reindex(range(len(R))).values
    shrunk = [g.buffer(-1.0) if g.area > 20 else g for g in Rg_m]
    pi2, gi2 = st.query(shrunk, predicate="intersects")
    R["n_sp_plans"] = pd.Series(1, index=pi2).groupby(level=0).size().reindex(range(len(R))).fillna(0).astype(int).values
    # nearest SP within 15 m when centroid is not inside one (centroid can fall in a courtyard)
    miss = np.where(pd.isna(R.sp_plannumber.values))[0]
    if len(miss):
        idx = st.nearest([Rc_m[i] for i in miss])
        d = shapely.distance(np.array(Rc_m)[miss], np.array(sp_geom, dtype=object)[idx])
        ok = d <= 15
        R.loc[miss[ok], "sp_plannumber"] = sp_pn[idx[ok]]
        R.loc[miss[ok], "sp_createdate"] = sp_cd[idx[ok]]
        R["sp_match"] = np.where(pd.notna(R.sp_plannumber), "centroid_within", None)
        R.loc[miss[ok], "sp_match"] = "nearest<=15m"

    # stations
    stn = json.load(open(ROOT / "osm" / "osm_stations_bbox.json", encoding="utf-8"))["elements"]
    srows = []
    for e in stn:
        c = e.get("center") or {"lat": e.get("lat"), "lon": e.get("lon")}
        if c.get("lat") is None:
            continue
        t = e.get("tags", {})
        kind = ("light_rail" if t.get("railway") == "tram_stop" or t.get("light_rail") == "yes" or t.get("tram") == "yes"
                else "rail_metro")
        x, y = TO_M(c["lon"], c["lat"])
        srows.append((x, y, kind, t.get("name")))
    S = pd.DataFrame(srows, columns=["x", "y", "kind", "name"])
    xy = np.c_[[p.x for p in Rc_m], [p.y for p in Rc_m]]
    R["dist_station_m"] = cKDTree(S[["x", "y"]].values).query(xy)[0]
    rm = S[S.kind == "rail_metro"]
    R["dist_rail_metro_m"] = cKDTree(rm[["x", "y"]].values).query(xy)[0]

    # water: coastline lines + water polygons >= 5 ha (harbour/bays/large ponds); small ponds ignored
    wat = json.load(open(ROOT / "osm" / "osm_water_bbox.json", encoding="utf-8"))["elements"]
    wg = []
    for e in wat:
        t = e.get("tags", {})
        if e["type"] == "way" and e.get("geometry"):
            coords = [(p["lon"], p["lat"]) for p in e["geometry"]]
            if t.get("natural") == "coastline":
                wg.append(transform(TO_M, LineString(coords)))
            elif len(coords) >= 4 and coords[0] == coords[-1]:
                pg = transform(TO_M, Polygon(coords).buffer(0))
                if pg.area >= 5e4:
                    wg.append(pg)
        elif e["type"] == "relation":
            lines = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]]) for m in e.get("members", [])
                     if m.get("role") == "outer" and m.get("geometry")]
            if lines:
                polys = list(polygonize(unary_union(lines)))
                if polys:
                    pg = transform(TO_M, unary_union(polys))
                    if pg.area >= 5e4:
                        wg.append(pg)
    W = unary_union(wg)
    R["dist_water_m"] = shapely.distance(np.array(Rc_m), W)
    print("water geoms", len(wg))

    # LST raster (point sample at centroid, ~30 m pixels)
    with rasterio.open(ROOT / "heat" / "lst_sydney_2022-2023_summer_bbox.tif") as src:
        vals = np.array([v[0] for v in src.sample(list(zip(R.lon, R.lat)))], dtype=float)
    R["lst_c"] = vals

    def poly_join(path, fields, prefix=""):
        fc = json.load(open(path, encoding="utf-8"))["features"]
        g = [transform(TO_M, shape(f["geometry"]).buffer(0)) for f in fc if f.get("geometry")]
        props = [f["properties"] for f in fc if f.get("geometry")]
        t = shapely.STRtree(g)
        a, b_ = t.query(Rc_m, predicate="within")
        m = pd.Series(b_, index=a).groupby(level=0).first().reindex(range(len(R)))
        out = {}
        for fsrc, fdst in fields.items():
            col = np.array([p.get(fsrc) for p in props], dtype=object)
            out[fdst] = [col[int(k)] if pd.notna(k) else None for k in m.values]
        return out

    for k, v in poly_join(ROOT / "canopy" / "canopy2022_mmb_bbox.geojson",
                          {"PCT_TREE": "canopy_mmb_pct", "Canopy_Percentage": "canopy_mmb_pct_alt",
                           "MMB_CODE": "mmb_code", "MB_CODE16": "mb_code16_canopy"}).items():
        R[k] = v
    for k, v in poly_join(ROOT / "heat" / "uhi2016_mb_bbox.geojson",
                          {"UHI_16_m": "uhi16_mmb", "MB_CODE16": "mb_code16", "SA1_MAIN16": "sa1_main16"}).items():
        R[k] = v
    for k, v in poly_join(ROOT / "heat" / "hvi2022_sa1_bbox.geojson",
                          {"LST": "lst_sa1_hvi", "SA1_2021_AUST_GDA2020_SA1_CODE2": "sa1_code21", "SA2_CODE21": "sa2_code21",
                           "SA2_NAME21": "sa2_name21", "Per_Tree": "tree_pct_sa1_hvi"}).items():
        R[k] = v
    abs2 = pd.read_csv(ROOT / "abs" / "c21_g02_sa2_median_rent_bbox.csv")
    abs2 = abs2.set_index(abs2.REGION.astype(str)).OBS_VALUE
    R["abs_sa2_median_rent_2021"] = R.sa2_code21.map(lambda s: abs2.get(str(int(s))) if pd.notna(s) else np.nan)
    for c in ["canopy_mmb_pct", "canopy_mmb_pct_alt", "uhi16_mmb", "lst_sa1_hvi", "tree_pct_sa1_hvi", "abs_sa2_median_rent_2021"]:
        R[c] = pd.to_numeric(R[c], errors="coerce")
    R["sp_createdate"] = pd.to_datetime(R.sp_createdate)
    R["osm_timestamp_base"] = osm_ts

    R.to_parquet(OUT / "buildings_features.parquet", index=False)
    R.to_csv(OUT / "buildings_features.csv", index=False)

    # coverage and correlations
    cols = ["footprint_m2", "levels", "neighbour_max_levels_100m", "sp_plannumber", "sp_createdate", "n_sp_plans",
            "dist_station_m", "dist_rail_metro_m", "dist_water_m", "lst_c", "canopy_mmb_pct", "uhi16_mmb",
            "lst_sa1_hvi", "abs_sa2_median_rent_2021"]
    cov = pd.DataFrame({"pct_non_null_all": R[cols].notna().mean() * 100,
                        "pct_non_null_apt": R.loc[R.res_rule == "apt", cols].notna().mean() * 100}).round(1)
    cov.loc["n_sp_plans>=2 (multi-plan share)", :] = [(R.n_sp_plans >= 2).mean() * 100,
                                                      (R.loc[R.res_rule == "apt", "n_sp_plans"] >= 2).mean() * 100]
    num_cols = [c for c in cols if c != "sp_createdate"]
    C = R[num_cols].assign(sp_create_year=R.sp_createdate.dt.year + R.sp_createdate.dt.day_of_year / 365.25)
    corr = C.corr(method="spearman").round(2)
    cov.to_csv(OUT / "buildings_features_coverage.csv")
    corr.to_csv(OUT / "buildings_features_corr_spearman.csv")
    desc = R[num_cols].describe().T.round(2)
    desc.to_csv(OUT / "buildings_features_describe.csv")
    with open(OUT / "buildings_features_summary.md", "w", encoding="utf-8") as f:
        f.write(f"# buildings_features summary\n\nRows: {len(R)} ({R.res_rule.value_counts().to_dict()}); OSM base {osm_ts}\n\n")
        f.write("## Coverage (% non-null)\n\n" + cov.to_markdown() + "\n\n## Describe\n\n" + desc.to_markdown() +
                "\n\n## Spearman correlations\n\n" + corr.to_markdown() + "\n")
    pd.set_option("display.width", 250)
    print(cov.to_string())
    print(corr.to_string())
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
