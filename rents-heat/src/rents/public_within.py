"""Two public-data adjustments that do not double-count what building comparables already contain. Tested on window A.

P1 within-building floor view: for a flat priced from same-building same-bedroom comparables, adjust by
   beta_w . (x_target - mean x of the comparable flats), x = LiDAR [water_view_frac, park_view_frac,
   roofline_clear_frac] at the flat's floor (building median when the floor is unknown). beta_w is estimated WITHIN
   building x bedroom groups on training adverts (fixed effects: demeaned log rent on demeaned x), so building-level
   location, age and quality cancel out. The adjustment is scaled by the comps' weight n/(n+k).
P2 ladder-only rows: flats with neither usable own history nor comparables get the cross-sectional public adjustment
   (residual vs suburb ladder on [views, height, flight path, new build, strata age]) - nothing else in their estimate
   describes the building.

Usage (from rents-heat):  .venv/Scripts/python.exe src/rents/public_within.py --data real --adverts both [--window A]
Prints window metrics for: frozen rule, +P1, +P2, +P1+P2, and the fitted coefficients; writes public_within_<w>.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rents.load import cfg, load_adverts, path_for, read_input, test_slice, train_slice, window  # noqa: E402
from rents.model import advert_parking, combine_rule, fit, mnum, score  # noqa: E402
from rents.public_features import X_COLS, public_matrix  # noqa: E402

XW = ["water_view_frac", "park_view_frac", "roofline_clear_frac"]


def signals(f, flats, train, mc):
    """Latest same-bedroom advert per flat in the comps lookback, as a without-parking log rent moved to the cut-off."""
    m_hi = int(mnum(pd.Series([f.train_end]))[0])
    a = train.merge(flats[["gnaf_id", "suburb_name", "bed", "bkey", "park_in"]], on="gnaf_id")
    a["m"] = mnum(a["date"])
    a = a[a["m"] > m_hi - mc["comps_lookback_months"]]
    prem = f.table.set_index("gnaf_id")["prem"]
    pa = advert_parking(a["car_spaces"], a["park_in"]).fillna(0)
    a["s"] = np.log(a["price"]) + f.index.log_to_now(a["suburb_name"].values, a["m"].values) - a["gnaf_id"].map(prem).values * pa
    return a.sort_values("date").groupby("gnaf_id").tail(1)[["gnaf_id", "bkey", "bed", "s"]]


def within_beta(one, feats):
    d = one.merge(feats[["gnaf_id"] + XW], on="gnaf_id").dropna(subset=XW + ["s"])
    g = d.groupby(["bkey", "bed"])
    d = d[g["gnaf_id"].transform("size") >= 2]
    g = d.groupby(["bkey", "bed"])
    y = d["s"] - g["s"].transform("mean")
    X = d[XW] - g[XW].transform("mean")
    beta, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
    return beta, int(len(d)), int(d.groupby(["bkey", "bed"]).ngroups)


def apply_p1(pred, f, flats, one, feats, beta, k):
    o = one.merge(feats[["gnaf_id"] + XW], on="gnaf_id", how="left")
    o[XW] = o[XW].fillna(o[XW].mean())
    gs = o.groupby(["bkey", "bed"]).agg(**{f"S_{c}": (c, "sum") for c in XW}, N=("gnaf_id", "size")).reset_index()
    t = pred.merge(flats[["gnaf_id", "bed"]], on="gnaf_id").merge(gs, on=["bkey", "bed"], how="left") \
        .merge(o[["gnaf_id"] + XW].rename(columns={c: f"self_{c}" for c in XW}), on="gnaf_id", how="left") \
        .merge(feats[["gnaf_id"] + XW], on="gnaf_id", how="left")
    self_in = t[f"self_{XW[0]}"].notna()
    n = t["N"].fillna(0) - self_in.astype(int)
    delta = np.zeros(len(t))
    for j, c in enumerate(XW):
        mean_c = (t[f"S_{c}"].fillna(0) - t[f"self_{c}"].fillna(0)) / n.replace(0, np.nan)
        delta += beta[j] * (t[c] - mean_c).fillna(0).values
    w = (n / (n + k)).fillna(0).values
    lift = np.exp(np.where(t["used_comps"].values & (n.values > 0), w * delta, 0.0))
    out = pred.copy()
    for col in ("rent_with_car_space", "rent_without_car_space"):
        out[col] = out[col].values * lift
    return out, int((lift != 1).sum())


def cross_beta(f, flats, train, feats):
    from rents.model import mnum as _m
    t = f.table.merge(feats[["gnaf_id"] + X_COLS], on="gnaf_id", how="left")
    a = train.merge(flats[["gnaf_id", "suburb_name", "park_in"]], on="gnaf_id").sort_values("date").groupby("gnaf_id").tail(1)
    a = a.merge(t[["gnaf_id", "ladder_without", "prem"] + X_COLS], on="gnaf_id")
    pa = advert_parking(a["car_spaces"], a["park_in"]).fillna(0)
    a["resid"] = np.log(a["price"]) + f.index.log_to_now(a["suburb_name"].values, _m(a["date"]).values) - a["prem"] * pa - np.log(a["ladder_without"])
    means = feats.loc[feats["osm_id"].notna(), X_COLS].mean()
    a = a[a["gnaf_id"].isin(feats.loc[feats["osm_id"].notna(), "gnaf_id"])]
    a[X_COLS] = a[X_COLS].fillna(means)
    d = a.dropna(subset=X_COLS + ["resid"])
    X = np.c_[np.ones(len(d)), d[X_COLS].values]
    beta, *_ = np.linalg.lstsq(X, d["resid"].values, rcond=None)
    return beta[1:], float(np.mean(d[X_COLS].values @ beta[1:])), means


def apply_p2(pred, feats, beta, center, means):
    t = pred.merge(feats[["gnaf_id", "osm_id"] + X_COLS], on="gnaf_id", how="left")
    X = t[X_COLS].fillna(means).values
    ladder_only = (~t["own"].values | t["stale_own"].values) & ~t["used_comps"].values & t["osm_id"].notna().values
    lift = np.exp(np.where(ladder_only, X @ beta - center, 0.0))
    out = pred.copy()
    for col in ("rent_with_car_space", "rent_without_car_space"):
        out[col] = out[col].values * lift
    return out, int(ladder_only.sum())


def snap(pred, s):
    if not s:
        return pred
    out = pred.copy()
    for col in ("rent_with_car_space", "rent_without_car_space"):
        out[col] = np.round(out[col] / s) * s
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "real", "apiproxy"], default="real")
    ap.add_argument("--adverts", choices=["auto", "past_leases", "api", "both"], default="both")
    ap.add_argument("--window", choices=["A", "B"], default="A")
    a = ap.parse_args(argv)
    c = cfg()
    out_dir = path_for(c, a.data, "out")
    fr = json.load(open(os.path.join(out_dir, "frozen_rule.json"), encoding="utf-8"))
    flats = read_input(path_for(c, a.data, "input"), c)
    adverts, _ = load_adverts(c, a.data, a.adverts)
    w = window(c, a.window)
    tr, te = train_slice(adverts, w["train_end"]), test_slice(adverts, w["test_start"], w["test_end"])
    f = fit(flats, tr, w["train_end"], c["model"])
    feats = public_matrix(c, flats)
    one = signals(f, flats, tr, c["model"])
    bw, n_rows, n_groups = within_beta(one, feats)
    bc, center, means = cross_beta(f, flats, tr, feats)
    base_raw = combine_rule(f, dict(fr, snap=None))
    k = fr.get("k") or 0.5
    p1, n1 = apply_p1(base_raw, f, flats, one, feats, bw, k)
    p2, n2 = apply_p2(base_raw, feats, bc, center, means)
    p12, _ = apply_p2(p1, feats, bc, center, means)
    res = dict(window=a.window, beta_within=dict(zip(XW, np.round(bw, 4).tolist())), within_rows=n_rows, within_groups=n_groups,
               beta_cross=dict(zip(X_COLS, np.round(bc, 4).tolist())), rows_adjusted_p1=n1, rows_ladder_only_p2=n2, variants={})
    for name, p in (("frozen", base_raw), ("frozen+P1_within_floor_view", p1), ("frozen+P2_ladder_only", p2), ("frozen+P1+P2", p12)):
        s = score(snap(p, fr.get("snap")), flats, te)
        res["variants"][name] = s
        print(f"{name:<30} all ${s['all']['medae']:6.2f} / {s['all']['within100']:.4f} / R2 {s['all']['r2']:.4f} | "
              f"own ${s['own_history']['medae']} | no-hist ${s['no_history']['medae']} / {s['no_history']['within100']:.4f} (n={s['no_history']['n']})")
        if name != "frozen":
            ss = score(p, flats, te)
            print(f"{'   (no snapping)':<30} all ${ss['all']['medae']:6.2f} / {ss['all']['within100']:.4f} | no-hist ${ss['no_history']['medae']}")
    print("within-building beta (log points per unit share of directions):", res["beta_within"], f"from {n_rows:,} flats in {n_groups:,} building x bedroom groups")
    print("cross-sectional beta:", res["beta_cross"])
    print(f"rows adjusted: P1 {n1:,} (comps rows), P2 {n2:,} (ladder-only rows with a building match)")
    json.dump(res, open(os.path.join(out_dir, f"public_within_{a.window}.json"), "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
