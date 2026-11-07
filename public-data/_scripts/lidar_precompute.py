"""LiDAR + aircraft precompute for the Microburbs rents hackathon (public data only).

Inputs (public-data/):
  lidar/DATA_1698.zip            ELVIS order: NSW Spatial Services 1 m DEM + C3 AHD point clouds, surveys 2013-04 and 2020-05 (CC BY)
  _derived/buildings_features.csv  4,399 OSM residential buildings (lon/lat, levels, ...)
  osm/osm_buildings_bbox.json    OSM building footprints (ODbL)
  osm/osm_water_bbox.json, osm/osm_parks_bbox.json
  noise/osm_syd_runways.json     Sydney Airport runways (ODbL)
Outputs (public-data/_derived/):
  lidar_rasters/{dem,dsm,ndsm}_{2013,2020}.tif   1 m mosaics (LZW)
  lidar_buildings.csv/.parquet   per building: ground, heights 2013/2020, new-build flag
  lidar_floor_view.csv/.parquet  per building x floor: roofline margin, water/park view
  aircraft_proxy.csv/.parquet    per building: distance to runway centrelines
  buildings_features_v2.csv/.parquet  everything joined (one row per building)
  DATA_DICTIONARY_lidar.md
Run: rents-heat/.venv/Scripts/python.exe public-data/_scripts/lidar_precompute.py
"""
import json
import math
import re
import sys
import time
import zipfile
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import LineString, Polygon, shape
from shapely.ops import transform as shp_transform

ROOT = Path(__file__).resolve().parents[1]
LIDAR = ROOT / "lidar"
RAW = LIDAR / "raw"
DER = ROOT / "_derived"
RAS = DER / "lidar_rasters"
RAS.mkdir(parents=True, exist_ok=True)
T0 = time.time()
FLOOR_H = 3.1      # metres per storey (assumption)
EYE_H = 1.5        # eye above floor
N_RAYS = 36
STEP = 5.0         # metres between ray samples
MAX_D = 1500.0     # view radius
ROOF_R = 100.0     # roofline neighbourhood radius


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- 1. unzip
def unzip():
    zp = LIDAR / "DATA_1698.zip"
    RAW.mkdir(exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        for n in names:
            out = RAW / Path(n).name
            if out.exists() and out.stat().st_size == z.getinfo(n).file_size:
                continue
            if n.endswith("/"):
                continue
            with z.open(n) as src, open(out, "wb") as dst:
                while True:
                    b = src.read(16 << 20)
                    if not b:
                        break
                    dst.write(b)
    # nested zips (ELVIS often nests one zip per file)
    for inner in list(RAW.glob("*.zip")):
        with zipfile.ZipFile(inner) as z:
            for n in z.namelist():
                if n.endswith("/"):
                    continue
                out = RAW / Path(n).name
                if not out.exists():
                    with z.open(n) as src, open(out, "wb") as dst:
                        while True:
                            b = src.read(16 << 20)
                            if not b:
                                break
                            dst.write(b)
    tifs = sorted(RAW.glob("*.tif"))
    lazs = sorted(RAW.glob("*.laz"))
    log(f"unzipped: {len(tifs)} tif, {len(lazs)} laz")
    return tifs, lazs


def year_of(p):
    m = re.search(r"Sydney(\d{4})(\d{2})", p.name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------- 2. mosaics
def build_year(year, tifs, lazs):
    tifs = [t for t in tifs if year_of(t) == year]
    lazs = [l for l in lazs if year_of(l) == year]
    with rasterio.open(tifs[0]) as s:
        crs = s.crs
        res = s.res[0]
    bounds = []
    for t in tifs:
        with rasterio.open(t) as s:
            bounds.append(s.bounds)
    xmin = min(b.left for b in bounds); xmax = max(b.right for b in bounds)
    ymin = min(b.bottom for b in bounds); ymax = max(b.top for b in bounds)
    W = int(round((xmax - xmin) / res)); H = int(round((ymax - ymin) / res))
    tr = from_origin(xmin, ymax, res, res)
    dem = np.full((H, W), np.nan, np.float32)
    for t in tifs:
        with rasterio.open(t) as s:
            a = s.read(1).astype(np.float32)
            if s.nodata is not None:
                a[a == s.nodata] = np.nan
            r0 = int(round((ymax - s.bounds.top) / res)); c0 = int(round((s.bounds.left - xmin) / res))
            dem[r0:r0 + a.shape[0], c0:c0 + a.shape[1]] = a
    log(f"{year}: DEM mosaic {H}x{W} crs={crs} res={res}")
    dsm = np.full((H, W), -np.inf, np.float32)
    bld = np.full((H, W), -np.inf, np.float32)
    flat = dsm.ravel(); flatb = bld.ravel()
    npts = 0
    for i, l in enumerate(lazs):
        with laspy.open(l) as f:
            for ch in f.chunk_iterator(2_000_000):
                cls = np.asarray(ch.classification)
                keep = (cls != 7) & (cls != 18)
                x = np.asarray(ch.x)[keep]; y = np.asarray(ch.y)[keep]; z = np.asarray(ch.z, dtype=np.float32)[keep]
                c = ((x - xmin) / res).astype(np.int64); r = ((ymax - y) / res).astype(np.int64)
                ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
                idx = r[ok] * W + c[ok]; zz = z[ok]
                np.maximum.at(flat, idx, zz)
                isb = cls[keep][ok] == 6
                if isb.any():
                    np.maximum.at(flatb, idx[isb], zz[isb])
                npts += idx.size
        log(f"{year}: laz {i+1}/{len(lazs)} {l.name} cumulative pts {npts:,}")
    dsm[~np.isfinite(dsm)] = np.nan
    bld[~np.isfinite(bld)] = np.nan
    # fill DSM holes with DEM (water / no returns)
    hole = np.isnan(dsm)
    dsm[hole] = dem[hole]
    ndsm = dsm - dem
    prof = dict(driver="GTiff", height=H, width=W, count=1, dtype="float32", crs=crs, transform=tr,
                nodata=np.nan, compress="LZW", tiled=True, blockxsize=512, blockysize=512, predictor=3)
    for name, arr in (("dem", dem), ("dsm", dsm), ("ndsm", ndsm), ("bldmax", bld)):
        with rasterio.open(RAS / f"{name}_{year}.tif", "w", **prof) as d:
            d.write(arr, 1)
    log(f"{year}: wrote rasters; DSM hole share {hole.mean():.3f}")
    return dict(dem=dem, dsm=dsm, ndsm=ndsm, bld=bld, tr=tr, crs=crs, res=res, xmin=xmin, ymax=ymax, H=H, W=W)


# ---------------------------------------------------------------- 3. footprints
def load_footprints(bf, to_proj):
    d = json.load(open(ROOT / "osm" / "osm_buildings_bbox.json", encoding="utf-8"))
    geo = {}
    for e in d["elements"]:
        if e.get("type") == "way" and "geometry" in e and len(e["geometry"]) >= 4:
            pts = [(g["lon"], g["lat"]) for g in e["geometry"]]
            try:
                geo[("way", e["id"])] = Polygon(pts)
            except Exception:
                pass
    polys = []
    for _, r in bf.iterrows():
        p = geo.get((r.osm_type, int(r.osm_id)))
        if p is None or not p.is_valid or p.area == 0:
            x, y = to_proj.transform(r.lon, r.lat)
            pp = Polygon([(x - 8, y - 8), (x + 8, y - 8), (x + 8, y + 8), (x - 8, y + 8)])
            polys.append((pp, "box16m"))
        else:
            polys.append((shp_transform(lambda xx, yy: to_proj.transform(xx, yy), p), "osm"))
    return polys


def per_building_stats(polys, M, year):
    lab = rasterize(((g, i + 1) for i, (g, _) in enumerate(polys)), out_shape=(M["H"], M["W"]),
                    transform=M["tr"], fill=0, dtype="int32", all_touched=False)
    m = lab > 0
    df = pd.DataFrame({"b": lab[m] - 1, "dem": M["dem"][m], "ndsm": M["ndsm"][m], "bld": M["bld"][m] - M["dem"][m]})
    g = df.groupby("b")
    out = pd.DataFrame({
        f"ground_z_{year}": g["dem"].median(),
        f"h_p90_{year}": g["ndsm"].quantile(0.9),
        f"h_max_{year}": g["ndsm"].max(),
        f"h_bldclass_p90_{year}": g["bld"].quantile(0.9),
        f"n_px_{year}": g.size(),
    })
    return out


# ---------------------------------------------------------------- 4. masks
def masks(M, to_proj):
    H, W = M["H"], M["W"]
    shapes_w, shapes_p = [], []
    dw = json.load(open(ROOT / "osm" / "osm_water_bbox.json", encoding="utf-8"))
    for e in dw["elements"]:
        tags = e.get("tags", {})
        if tags.get("natural") == "coastline":
            continue
        if e["type"] == "way" and "geometry" in e and len(e["geometry"]) >= 4:
            try:
                p = Polygon([(g["lon"], g["lat"]) for g in e["geometry"]])
                if p.is_valid:
                    shapes_w.append(shp_transform(lambda x, y: to_proj.transform(x, y), p))
            except Exception:
                pass
        elif e["type"] == "relation":
            for mbr in e.get("members", []):
                if mbr.get("role") == "outer" and "geometry" in mbr and len(mbr["geometry"]) >= 4:
                    try:
                        p = Polygon([(g["lon"], g["lat"]) for g in mbr["geometry"]])
                        if p.is_valid:
                            shapes_w.append(shp_transform(lambda x, y: to_proj.transform(x, y), p))
                    except Exception:
                        pass
    dp = json.load(open(ROOT / "osm" / "osm_parks_bbox.json", encoding="utf-8"))
    for e in dp["elements"]:
        geoms = []
        if e["type"] == "way" and "geometry" in e and len(e["geometry"]) >= 4:
            geoms.append(e["geometry"])
        elif e["type"] == "relation":
            geoms += [m["geometry"] for m in e.get("members", []) if m.get("role") == "outer" and "geometry" in m and len(m["geometry"]) >= 4]
        for gg in geoms:
            try:
                p = Polygon([(g["lon"], g["lat"]) for g in gg])
                if p.is_valid and p.area > 0:
                    shapes_p.append(shp_transform(lambda x, y: to_proj.transform(x, y), p))
            except Exception:
                pass
    water = np.zeros((H, W), bool)
    if shapes_w:
        water |= rasterize(((g, 1) for g in shapes_w), out_shape=(H, W), transform=M["tr"], fill=0, dtype="uint8").astype(bool)
    # sea-level flat surfaces in the LiDAR (harbour where OSM has only coastline)
    water |= (M["dem"] <= 0.5) & (np.abs(M["ndsm"]) < 1.0)
    park = np.zeros((H, W), bool)
    if shapes_p:
        park = rasterize(((g, 1) for g in shapes_p), out_shape=(H, W), transform=M["tr"], fill=0, dtype="uint8").astype(bool)
    log(f"masks: water {water.mean():.3f}, park {park.mean():.3f} (from {len(shapes_w)} water / {len(shapes_p)} park polygons)")
    return water, park


# ---------------------------------------------------------------- 5. views per floor
def views(bf, polys, bstats, M, water, park, VY):
    res = M["res"]; H, W = M["H"], M["W"]; dsm = M["dsm"]
    ang = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)
    rows = []
    for i, (g, src) in enumerate(polys):
        c = g.centroid
        r0 = math.sqrt(max(g.area, 64) / math.pi) + 2.0
        dists = np.arange(r0, MAX_D, STEP, dtype=np.float32)
        X = c.x + np.cos(ang)[:, None] * dists[None, :]
        Y = c.y + np.sin(ang)[:, None] * dists[None, :]
        cc = ((X - M["xmin"]) / res).astype(np.int64); rr = ((M["ymax"] - Y) / res).astype(np.int64)
        inb = (cc >= 0) & (cc < W) & (rr >= 0) & (rr < H)
        cc = np.clip(cc, 0, W - 1); rr = np.clip(rr, 0, H - 1)
        Z = dsm[rr, cc].astype(np.float32); Z[~inb] = np.nan
        WAT = water[rr, cc] & inb; PRK = park[rr, cc] & inb
        Z = np.where(np.isnan(Z), -1e3, Z)
        gz = bstats.loc[i, f"ground_z_{VY}"] if i in bstats.index else np.nan
        ground = float(gz) if np.isfinite(gz) else float(np.nanmedian(np.where(Z[:, :3] > -100, Z[:, :3], np.nan)))
        h = bstats.loc[i, f"h_p90_{VY}"] if i in bstats.index else np.nan
        lev_osm = bf.iloc[i]["levels"]
        n_fl = int(np.nanmax([lev_osm if np.isfinite(lev_osm) else 0, round(h / FLOOR_H) if np.isfinite(h) else 0, 1]))
        n_fl = min(n_fl, 80)
        E = ground + FLOOR_H * np.arange(n_fl, dtype=np.float32)[:, None, None] + EYE_H  # floor 0 = ground floor
        A = (Z[None] - E) / dists[None, None, :]
        prevmax = np.maximum.accumulate(A, axis=2)
        prevmax = np.concatenate([np.full(A.shape[:2] + (1,), -np.inf, np.float32), prevmax[:, :, :-1]], axis=2)
        vis = A >= prevmax - 1e-6
        wv = vis & WAT[None]; pv = vis & PRK[None]
        near = dists <= ROOF_R
        roof_v = float(Z[:, near].max()) if near.any() and Z[:, near].max() > -100 else np.nan
        any_w = wv.any(axis=2); any_p = pv.any(axis=2)
        dW = np.where(wv, dists[None, None, :], np.inf).min(axis=(1, 2))
        dP = np.where(pv, dists[None, None, :], np.inf).min(axis=(1, 2))
        for f in range(n_fl):
            eye = float(E[f, 0, 0])
            rows.append((i, f, eye, roof_v, eye - roof_v, float(any_w[f].mean()), float(dW[f]) if np.isfinite(dW[f]) else np.nan,
                         float(any_p[f].mean()), float(dP[f]) if np.isfinite(dP[f]) else np.nan))
        if (i + 1) % 250 == 0:
            log(f"views: {i+1}/{len(polys)} buildings")
    v = pd.DataFrame(rows, columns=["b", "floor", "eye_z_ahd", "roof_max_100m_ahd", "roof_margin_m",
                                    "water_view_frac", "water_nearest_visible_m", "park_view_frac", "park_nearest_visible_m"])
    return v


# ---------------------------------------------------------------- 6. aircraft proxy
def aircraft(bf, to_proj):
    d = json.load(open(ROOT / "noise" / "osm_syd_runways.json", encoding="utf-8"))
    out = pd.DataFrame(index=range(len(bf)))
    pts = np.array([to_proj.transform(lo, la) for lo, la in zip(bf.lon, bf.lat)])
    for e in d["elements"]:
        ref = e["tags"].get("ref", "")
        if "/" not in ref:
            continue
        g = e["geometry"]
        a = np.array(to_proj.transform(g[0]["lon"], g[0]["lat"])); b = np.array(to_proj.transform(g[-1]["lon"], g[-1]["lat"]))
        u = (b - a) / np.linalg.norm(b - a)
        # orient so u points from the runway toward the north (the City of Sydney side) for 16/34, west->east kept for 07/25
        key = ref.replace("/", "_")
        if ref.startswith("16") or ref.startswith("34"):
            if u[1] < 0:
                a, b, u = b, a, -u
            far = b  # northern threshold
        else:
            far = b
        rel = pts - far
        along = rel @ u
        lat_ = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
        out[f"rwy_{key}_along_m"] = along
        out[f"rwy_{key}_lateral_m"] = lat_
    lat_cols = [c for c in out if c.endswith("_lateral_m") and ("16" in c or "34" in c)]
    out["rwy_16_34_min_lateral_m"] = out[lat_cols].min(axis=1)
    near = out[lat_cols].idxmin(axis=1).str.replace("_lateral_m", "_along_m")
    out["rwy_16_34_along_m"] = [out.loc[i, near[i]] for i in out.index]
    out["under_16_34_path_1km"] = (out.rwy_16_34_min_lateral_m <= 1000) & (out.rwy_16_34_along_m > 0) & (out.rwy_16_34_along_m <= 15000)
    log(f"aircraft: buildings within 1 km of a 16/34 approach path (north side, <=15 km): {int(out.under_16_34_path_1km.sum())}")
    return out


def main():
    tifs, lazs = unzip()
    years = sorted({year_of(p) for p in tifs if year_of(p)})
    log("survey years:", years)
    bf = pd.read_csv(DER / "buildings_features.csv")
    VY = max(years)
    M = {}
    stats = []
    polys = None
    # low-memory order: older surveys first, stats computed and arrays freed before the next survey
    for y in sorted(years):
        My = build_year(y, tifs, lazs)
        tp = Transformer.from_crs("EPSG:4326", My["crs"], always_xy=True)
        py = load_footprints(bf, tp)
        stats.append(per_building_stats(py, My, y))
        if y == VY:
            M[y] = My
            polys = py
            to_proj = tp
            crs = My["crs"]
        else:
            del My
        import gc; gc.collect()
        log(f"{y}: stats done, memory freed")
    bstats = pd.concat(stats, axis=1).reindex(range(len(bf)))
    if 2013 in years and 2020 in years:
        bstats["new_2013_2020"] = (bstats["h_p90_2013"].fillna(0) < 3) & (bstats["h_p90_2020"] >= 9)
        bstats["height_change_2013_2020_m"] = bstats["h_p90_2020"] - bstats["h_p90_2013"]
    bstats[f"levels_from_lidar_{VY}"] = (bstats[f"h_p90_{VY}"] / FLOOR_H).round()
    bstats["footprint_src"] = [s for _, s in polys]
    log("building stats done")
    Mv = M[max(years)]
    water, park = masks(Mv, to_proj)
    v = views(bf, polys, bstats, Mv, water, park, VY)
    air = aircraft(bf, to_proj)
    key = bf[["osm_type", "osm_id"]].reset_index().rename(columns={"index": "b"})
    lb = key.merge(bstats.reset_index().rename(columns={"index": "b"}), on="b").drop(columns="b")
    lv = key.merge(v, on="b").drop(columns="b")
    la = pd.concat([bf[["osm_type", "osm_id"]], air], axis=1)
    # building-level view summaries (top floor and floor where water first visible)
    first_w = v[v.water_view_frac > 0].groupby("b").floor.min().rename("first_floor_water_view")
    top = v.sort_values("floor").groupby("b").tail(1).set_index("b")[["water_view_frac", "park_view_frac", "roof_margin_m"]].add_prefix("top_floor_")
    above = v[v.roof_margin_m > 0].groupby("b").floor.min().rename("first_floor_above_roofline")
    summ = pd.concat([first_w, top, above], axis=1).reindex(range(len(bf)))
    v2 = pd.concat([bf.reset_index(drop=True), bstats.reset_index(drop=True), air.reset_index(drop=True), summ.reset_index(drop=True)], axis=1)
    for name, df in (("lidar_buildings", lb), ("lidar_floor_view", lv), ("aircraft_proxy", la), ("buildings_features_v2", v2)):
        df.to_csv(DER / f"{name}.csv", index=False)
        try:
            df.to_parquet(DER / f"{name}.parquet", index=False)
        except Exception as ex:  # noqa: BLE001
            log("parquet failed", name, ex)
        log(f"wrote {name}: {df.shape}")
    # quick summary for the report
    s = {
        "years": years, "crs": str(crs), "n_buildings": len(bf),
        "coverage_h_latest": float(bstats[f"h_p90_{VY}"].notna().mean()),
        "median_h_latest": float(bstats[f"h_p90_{VY}"].median()),
        "corr_lidar_levels_vs_osm": float(pd.concat([bstats[f"levels_from_lidar_{VY}"], bf.levels], axis=1).corr().iloc[0, 1]),
        "new_2013_2020": int(bstats.get("new_2013_2020", pd.Series(dtype=bool)).sum()),
        "floor_rows": len(v), "share_floors_water_view": float((v.water_view_frac > 0).mean()),
        "share_floors_above_roofline": float((v.roof_margin_m > 0).mean()),
        "under_16_34_path_1km": int(air.under_16_34_path_1km.sum()),
        "runtime_s": time.time() - T0,
    }
    json.dump(s, open(DER / "lidar_precompute_summary.json", "w"), indent=2)
    log("SUMMARY", s)


if __name__ == "__main__":
    main()
