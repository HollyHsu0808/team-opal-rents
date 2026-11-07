"""Recompute per-floor views from the saved 2020 rasters (no LAZ re-read, low memory).

Fixes in v2 (found in the 2026-11-07 08:44 output check):
- The building's own footprint is now excluded from every ray. In v1 rays started at the
  equal-area circle radius, so on elongated footprints they began inside the own roof, which
  made the own roof the "roofline" (99th pct roof_margin was -1.5 m; only 0.67% of floors cleared).
- New per-floor `roofline_clear_frac`: share of the 36 directions in which the floor clears every
  roof / tree top within 100 m. `roof_margin_m` (vs the single highest point within 100 m in any
  direction) is kept but is a strict all-round test.

Rewrites lidar_floor_view.{csv,parquet} and the view summary columns of buildings_features_v2.
"""
import json, math, time
import numpy as np, pandas as pd, rasterio, shapely
from pyproj import Transformer

import lidar_precompute as L

T0 = time.time()
VY = 2020


def load_rasters():
    with rasterio.open(L.RAS / f"dsm_{VY}.tif") as s:
        dsm = s.read(1); tr = s.transform; crs = s.crs; res = s.res[0]
        H, W = s.height, s.width; xmin, ymax = s.bounds.left, s.bounds.top
    with rasterio.open(L.RAS / f"dem_{VY}.tif") as s:
        dem = s.read(1)
    ndsm = dsm - dem
    return dict(dem=dem, dsm=dsm, ndsm=ndsm, tr=tr, crs=crs, res=res, xmin=xmin, ymax=ymax, H=H, W=W)


def views_v2(bf, polys, bstats, M, water, park):
    res = M["res"]; H, W = M["H"], M["W"]; dsm = M["dsm"]
    ang = np.linspace(0, 2 * np.pi, L.N_RAYS, endpoint=False)
    rows = []
    for i, (g, src) in enumerate(polys):
        c = g.centroid
        dists = np.arange(L.STEP, L.MAX_D, L.STEP, dtype=np.float32)
        X = c.x + np.cos(ang)[:, None] * dists[None, :]
        Y = c.y + np.sin(ang)[:, None] * dists[None, :]
        own = shapely.contains_xy(g, X, Y)
        cc = ((X - M["xmin"]) / res).astype(np.int64); rr = ((M["ymax"] - Y) / res).astype(np.int64)
        inb = (cc >= 0) & (cc < W) & (rr >= 0) & (rr < H)
        cc = np.clip(cc, 0, W - 1); rr = np.clip(rr, 0, H - 1)
        Z = dsm[rr, cc].astype(np.float32); Z[~inb] = np.nan
        Z[own] = np.nan                                   # own building never blocks or counts as roofline
        WAT = water[rr, cc] & inb & ~own; PRK = park[rr, cc] & inb & ~own
        Z = np.where(np.isnan(Z), -1e3, Z)
        gz = bstats.at[i, f"ground_z_{VY}"]
        ground = float(gz) if np.isfinite(gz) else float(np.nanmedian(dsm[rr[:, :4], cc[:, :4]]))
        h = bstats.at[i, f"h_p90_{VY}"]
        lev_osm = bf.iloc[i]["levels"]
        n_fl = int(np.nanmax([lev_osm if np.isfinite(lev_osm) else 0, round(h / L.FLOOR_H) if np.isfinite(h) else 0, 1]))
        n_fl = min(n_fl, 80)
        E = ground + L.FLOOR_H * np.arange(n_fl, dtype=np.float32)[:, None, None] + L.EYE_H
        A = (Z[None] - E) / dists[None, None, :]
        prevmax = np.maximum.accumulate(A, axis=2)
        prevmax = np.concatenate([np.full(A.shape[:2] + (1,), -np.inf, np.float32), prevmax[:, :, :-1]], axis=2)
        vis = A >= prevmax - 1e-6
        wv = vis & WAT[None]; pv = vis & PRK[None]
        near = dists <= L.ROOF_R
        ray_roof = Z[:, near].max(axis=1)                 # highest point within 100 m per direction (own excluded)
        ray_roof = np.where(ray_roof > -100, ray_roof, np.nan)
        roof_v = float(np.nanmax(ray_roof)) if np.isfinite(ray_roof).any() else np.nan
        any_w = wv.any(axis=2); any_p = pv.any(axis=2)
        dW = np.where(wv, dists[None, None, :], np.inf).min(axis=(1, 2))
        dP = np.where(pv, dists[None, None, :], np.inf).min(axis=(1, 2))
        for f in range(n_fl):
            eye = float(E[f, 0, 0])
            clear = float(np.mean(np.where(np.isfinite(ray_roof), ray_roof < eye, True)))
            rows.append((i, f, eye, roof_v, eye - roof_v, clear, float(any_w[f].mean()),
                         float(dW[f]) if np.isfinite(dW[f]) else np.nan,
                         float(any_p[f].mean()), float(dP[f]) if np.isfinite(dP[f]) else np.nan))
        if (i + 1) % 500 == 0:
            L.log(f"views v2: {i+1}/{len(polys)}")
    return pd.DataFrame(rows, columns=["b", "floor", "eye_z_ahd", "roof_max_100m_ahd", "roof_margin_m", "roofline_clear_frac",
                                       "water_view_frac", "water_nearest_visible_m", "park_view_frac", "park_nearest_visible_m"])


def main():
    bf = pd.read_csv(L.DER / "buildings_features.csv")
    lb = pd.read_parquet(L.DER / "lidar_buildings.parquet")
    assert (lb.osm_id.values == bf.osm_id.values).all(), "row order mismatch"
    bstats = lb.reset_index(drop=True)
    M = load_rasters()
    L.log(f"rasters loaded {M['H']}x{M['W']}")
    tp = Transformer.from_crs("EPSG:4326", M["crs"], always_xy=True)
    polys = L.load_footprints(bf, tp)
    water, park = L.masks(M, tp)
    del M["dem"], M["ndsm"]
    v = views_v2(bf, polys, bstats, M, water, park)
    key = bf[["osm_type", "osm_id"]].reset_index().rename(columns={"index": "b"})
    lv = key.merge(v, on="b").drop(columns="b")
    first_w = v[v.water_view_frac > 0].groupby("b").floor.min().rename("first_floor_water_view")
    top = v.sort_values("floor").groupby("b").tail(1).set_index("b")[
        ["water_view_frac", "park_view_frac", "roof_margin_m", "roofline_clear_frac"]].add_prefix("top_floor_")
    above = v[v.roof_margin_m > 0].groupby("b").floor.min().rename("first_floor_above_roofline")
    half = v[v.roofline_clear_frac >= 0.5].groupby("b").floor.min().rename("first_floor_clear_half_directions")
    summ = pd.concat([first_w, top, above, half], axis=1).reindex(range(len(bf)))
    v2 = pd.read_parquet(L.DER / "buildings_features_v2.parquet")
    v2 = v2.drop(columns=[c for c in v2.columns if c in summ.columns or c.startswith("top_floor_") or c.startswith("first_floor_")])
    v2 = pd.concat([v2.reset_index(drop=True), summ.reset_index(drop=True)], axis=1)
    for name, df in (("lidar_floor_view", lv), ("buildings_features_v2", v2)):
        df.to_csv(L.DER / f"{name}.csv", index=False); df.to_parquet(L.DER / f"{name}.parquet", index=False)
        L.log(f"wrote {name}: {df.shape}")
    s = json.load(open(L.DER / "lidar_precompute_summary.json"))
    s.update({"views_version": 2, "floor_rows": len(v),
              "share_floors_water_view": float((v.water_view_frac > 0).mean()),
              "share_floors_above_roofline": float((v.roof_margin_m > 0).mean()),
              "share_floors_clear_half_directions": float((v.roofline_clear_frac >= 0.5).mean()),
              "views_v2_runtime_s": time.time() - T0})
    json.dump(s, open(L.DER / "lidar_precompute_summary.json", "w"), indent=2)
    L.log("SUMMARY v2", {k: s[k] for k in s if "share" in k or k == "floor_rows"})


if __name__ == "__main__":
    main()
