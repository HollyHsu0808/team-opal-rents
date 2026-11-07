"""Public building features per flat (read-only inputs from public-data/_derived and public-data/osm).

Join: flats grouped by building key -> group mean point (4-dp coords, ~10 m) -> OSM footprints within 15 m (EPSG:7856);
1 candidate -> it; 2+ -> the one whose levels (OSM, else LiDAR) best match max(floor_level)+1 of the group, then nearest.
Features: per-floor view from lidar_floor_view (floor_level known, clipped to the building's floors) else the building
median over floors; point nDSM height from ndsm_2020.tif via small WINDOWED reads (never the full raster); aircraft proxy.
All sources predate the 2026-07-31 cut-off (LiDAR 2013/2020; see DATA_DICTIONARY_lidar.md).
Used only in the optional pre-specified window-A ablation (run_all.py --window A --ablation).
"""
import json
import os

import numpy as np
import pandas as pd

VIEW_COLS = ["roof_margin_m", "roofline_clear_frac", "water_view_frac", "park_view_frac"]
AIR_COLS = ["rwy_16_34_min_lateral_m", "under_16_34_path_1km"]
CRS_M = "EPSG:7856"


def _paths(c):
    from rents.load import ROOT
    d = os.path.normpath(os.path.join(ROOT, c["paths"]["public"]["derived"]))
    return d, os.path.normpath(os.path.join(ROOT, c["paths"]["public"]["osm_buildings"]))


def load_footprints(c):
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    derived, osm_json = _paths(c)
    b = pd.read_parquet(os.path.join(derived, "buildings_features_v2.parquet"),
                        columns=["osm_type", "osm_id", "lon", "lat", "levels", "levels_from_lidar_2020"])
    b["levels_best"] = b["levels"].fillna(b["levels_from_lidar_2020"])
    want = set(zip(b["osm_type"], b["osm_id"]))
    geoms = {}
    with open(osm_json, encoding="utf-8") as f:
        for el in json.load(f).get("elements", []):
            key = (el.get("type"), el.get("id"))
            g = el.get("geometry")
            if key in want and g and len(g) >= 3:
                geoms[key] = Polygon([(p["lon"], p["lat"]) for p in g])
    pts = gpd.GeoSeries(gpd.points_from_xy(b["lon"], b["lat"]), crs="EPSG:4326").to_crs(CRS_M)
    poly = gpd.GeoSeries([geoms.get(k) for k in zip(b["osm_type"], b["osm_id"])], crs="EPSG:4326").to_crs(CRS_M)
    # buildings without a way polygon (relations etc.): 16 m box around the centroid, as in lidar_precompute
    poly = poly.where(poly.notna() & ~poly.is_empty, pts.buffer(8, cap_style="square"))
    return gpd.GeoDataFrame(b, geometry=poly.values, crs=CRS_M)


def join_buildings(flats, fp, radius_m=15.0):
    """One row per building key: osm_type, osm_id, join_dist_m, n_candidates, x, y."""
    import geopandas as gpd
    from shapely import STRtree
    g = flats.groupby("bkey").agg(lat=("latitude", "mean"), lon=("longitude", "mean"),
                                  max_floor=("floor_level", "max")).reset_index()
    g = g[g["lat"].notna() & g["lon"].notna()].reset_index(drop=True)
    pts = gpd.GeoSeries(gpd.points_from_xy(g["lon"], g["lat"]), crs="EPSG:4326").to_crs(CRS_M)
    g["x"], g["y"] = pts.x.values, pts.y.values
    tree = STRtree(fp.geometry.values)
    qi, ti = tree.query(pts.values, predicate="dwithin", distance=radius_m)
    cand = pd.DataFrame({"gi": qi, "ti": ti})
    cand["dist"] = pts.values[qi].distance(fp.geometry.values[ti])
    cand["levels_best"] = fp["levels_best"].values[ti]
    cand["max_floor"] = g["max_floor"].values[qi]
    cand["lev_err"] = np.abs(cand["levels_best"] - (cand["max_floor"] + 1))
    cand["lev_err"] = cand["lev_err"].fillna(np.inf)
    ncand = cand.groupby("gi").size()
    best = cand.sort_values(["gi", "lev_err", "dist"]).groupby("gi").head(1).set_index("gi")
    g["osm_type"] = fp["osm_type"].values[best["ti"].reindex(g.index).fillna(-1).astype(int).clip(0)]
    g["osm_id"] = fp["osm_id"].values[best["ti"].reindex(g.index).fillna(-1).astype(int).clip(0)]
    hit = g.index.isin(best.index)
    g.loc[~hit, ["osm_type", "osm_id"]] = [None, np.nan]
    g["join_dist_m"] = best["dist"].reindex(g.index).values
    g["n_candidates"] = ncand.reindex(g.index).fillna(0).astype(int).values
    return g


def ndsm_at_points(raster_path, x, y, half=2):
    """Max nDSM in a (2*half+1) m window around each point; windowed reads only."""
    import rasterio
    from rasterio.windows import Window
    out = np.full(len(x), np.nan)
    if not os.path.exists(raster_path):
        import warnings
        warnings.warn(f"{raster_path} missing (download: gh release download data-2026-11-07 -D <that folder>); "
                      "ndsm_point_m left blank")
        return out
    with rasterio.open(raster_path) as src:
        nod = src.nodata
        for i, (xx, yy) in enumerate(zip(x, y)):
            r, c = src.index(xx, yy)
            if r < -half or c < -half or r >= src.height + half or c >= src.width + half:
                continue
            a = src.read(1, window=Window(c - half, r - half, 2 * half + 1, 2 * half + 1), boundless=True,
                         fill_value=nod if nod is not None else np.nan).astype(float)
            if nod is not None:
                a[a == nod] = np.nan
            if np.isfinite(a).any():
                out[i] = np.nanmax(a)
    return out


def flat_features(flats, c, radius_m=15.0):
    derived, _ = _paths(c)
    fp = load_footprints(c)
    g = join_buildings(flats, fp, radius_m)
    g["ndsm_point_m"] = ndsm_at_points(os.path.join(derived, "lidar_rasters", "ndsm_2020.tif"), g["x"].values, g["y"].values)
    fv = pd.read_parquet(os.path.join(derived, "lidar_floor_view.parquet"), columns=["osm_type", "osm_id", "floor"] + VIEW_COLS)
    air = pd.read_parquet(os.path.join(derived, "aircraft_proxy.parquet"), columns=["osm_type", "osm_id"] + AIR_COLS)
    f = flats[["gnaf_id", "bkey", "floor_level"]].merge(
        g[["bkey", "osm_type", "osm_id", "join_dist_m", "n_candidates", "ndsm_point_m"]], on="bkey", how="left")
    top = fv.groupby(["osm_type", "osm_id"])["floor"].max().rename("top_floor").reset_index()
    med = fv.groupby(["osm_type", "osm_id"])[VIEW_COLS].median().reset_index()
    f = f.merge(top, on=["osm_type", "osm_id"], how="left")
    f["floor_use"] = np.minimum(f["floor_level"], f["top_floor"]).clip(lower=0)
    byfloor = f.dropna(subset=["floor_use"]).assign(floor=lambda d: d["floor_use"].astype(int))
    byfloor = byfloor[["gnaf_id", "osm_type", "osm_id", "floor"]].merge(fv, on=["osm_type", "osm_id", "floor"], how="inner")
    f = f.merge(byfloor[["gnaf_id"] + VIEW_COLS], on="gnaf_id", how="left")
    f = f.merge(med, on=["osm_type", "osm_id"], how="left", suffixes=("", "_bmed"))
    f["view_from"] = np.where(f["water_view_frac"].notna(), "floor", np.where(f["water_view_frac_bmed"].notna(), "building_median", "none"))
    for col in VIEW_COLS:
        f[col] = f[col].fillna(f[col + "_bmed"])
    f = f.merge(air, on=["osm_type", "osm_id"], how="left")
    keep = ["gnaf_id", "osm_type", "osm_id", "join_dist_m", "n_candidates", "view_from", "ndsm_point_m"] + VIEW_COLS + AIR_COLS
    return f[keep]


X_COLS = ["water_view_frac", "park_view_frac", "roofline_clear_frac", "ndsm10", "air_path", "new_build", "age_decades"]


def public_matrix(c, flats):
    """flat_features + the derived regressors used by the public adjustment (one row per flat).
    Building-level extras from buildings_features_v2: new_build (LiDAR 2013->2020) and age_decades (strata plan year)."""
    feats = flat_features(flats, c)
    derived, _ = _paths(c)
    b = pd.read_parquet(os.path.join(derived, "buildings_features_v2.parquet"),
                        columns=["osm_type", "osm_id", "new_2013_2020", "sp_createdate"])
    b["new_build"] = b["new_2013_2020"].astype(float)
    b["age_decades"] = (2026 - pd.to_datetime(b["sp_createdate"], errors="coerce").dt.year) / 10
    feats = feats.merge(b[["osm_type", "osm_id", "new_build", "age_decades"]], on=["osm_type", "osm_id"], how="left")
    feats["air_path"] = feats["under_16_34_path_1km"].astype(float)
    feats["ndsm10"] = feats["ndsm_point_m"] / 10
    return feats


def apply_public(base, feats, pub):
    """Multiply the no-history rows of a prediction table by exp(X @ beta - center). A missing regressor is filled with
    its training mean (`pub['means']`), so one missing layer does not cancel the whole adjustment; flats with no
    building join at all keep lift 1. `pub` = frozen public rule from window A: {apply, X_cols, beta, center, means}."""
    X_cols = pub["X_cols"]
    adj = base.merge(feats[["gnaf_id", "osm_id"] + X_cols], on="gnaf_id", how="left")
    X = adj[X_cols].fillna(pd.Series(pub["means"])).values
    lift = np.exp(np.nan_to_num(X @ np.asarray(pub["beta"], float) - pub["center"], nan=0.0))
    lift[adj["osm_id"].isna().values] = 1.0
    nh = ~adj["own"].values
    for col in ["rent_with_car_space", "rent_without_car_space"]:
        adj.loc[nh, col] = adj.loc[nh, col] * lift[nh]
    return adj[base.columns]


def ablation_A(c, flats, fitted, train, test, frozen):
    """Pre-specified: OLS of training residuals (latest advert per flat, without-parking log scale, minus log ladder)
    on X_COLS; applied as a multiplicative adjustment to no-history rows on top of the frozen method. Reports window-A
    metrics with and without, and a `public_rule` that B/final apply when `apply` is True (acceptance rule below)."""
    from rents.model import combine_rule, mnum, score
    feats = public_matrix(c, flats)
    X_cols = X_COLS
    t = fitted.table.merge(feats[["gnaf_id"] + X_cols], on="gnaf_id", how="left")
    a = train.merge(flats[["gnaf_id", "suburb_name", "park_in"]], on="gnaf_id")
    a = a.sort_values("date").groupby("gnaf_id").tail(1)
    a = a.merge(t[["gnaf_id", "ladder_without", "prem"] + X_cols], on="gnaf_id")
    pa = (a["car_spaces"] > 0).astype(float).where(a["car_spaces"].notna()).fillna(a["park_in"]).fillna(0)
    s = np.log(a["price"]) + fitted.index.log_to_now(a["suburb_name"].values, mnum(a["date"]).values) - a["prem"] * pa
    a["resid"] = s - np.log(a["ladder_without"])
    means = feats.loc[feats["osm_id"].notna(), X_cols].mean()  # training-side fill values for missing layers
    a = a[a["gnaf_id"].isin(feats.loc[feats["osm_id"].notna(), "gnaf_id"])]
    a[X_cols] = a[X_cols].fillna(means)
    d = a.dropna(subset=X_cols + ["resid"])
    out = dict(n_train_rows=int(len(d)), join=dict(
        flats_joined=int(feats["osm_id"].notna().sum()), flats_total=int(len(feats)),
        view_from=feats["view_from"].value_counts().to_dict(), median_join_dist_m=float(feats["join_dist_m"].median())))
    if len(d) < 50:
        out["result"] = "not enough rows: incremental value untested"
        out["public_rule"] = dict(apply=False, reason="fewer than 50 training rows with features")
        return out
    X = np.c_[np.ones(len(d)), d[X_cols].values]
    beta, *_ = np.linalg.lstsq(X, d["resid"].values, rcond=None)
    out["beta"] = dict(zip(["intercept"] + X_cols, np.round(beta, 4).tolist()))
    pub = dict(X_cols=X_cols, beta=beta[1:].tolist(), center=float(np.nanmean(d[X_cols].values @ beta[1:])),
               means={k: float(v) for k, v in means.items()})
    base = combine_rule(fitted, frozen)
    adj = apply_public(base, feats, pub)
    out["frozen"] = score(base, flats, test)
    out["frozen_plus_public"] = score(adj, flats, test)
    b, p = out["frozen"], out["frozen_plus_public"]
    ok = (p["no_history"]["medae"] is not None and b["no_history"]["medae"] is not None
          and p["no_history"]["medae"] < b["no_history"]["medae"]
          and p["all"]["medae"] <= b["all"]["medae"] and p["all"]["within100"] >= b["all"]["within100"])
    pub.update(apply=bool(ok), selected_on="A",
               rule="apply only if no-history MedAE improves, overall MedAE does not worsen and share<=100 does not drop",
               A_no_history=dict(base=b["no_history"]["medae"], plus_public=p["no_history"]["medae"]),
               A_all=dict(base=b["all"]["medae"], plus_public=p["all"]["medae"]))
    out["public_rule"] = pub
    out["result"] = "accepted: B/final apply the public adjustment" if ok else "rejected on A: not applied"
    return out
