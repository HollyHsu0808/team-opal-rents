"""Report insight: does the market price what the public layers describe?

Refits window B (train <= B.train_end), predicts Jun-Jul test adverts with the frozen rule, and tabulates the
residual log(actual / predicted) by public-feature band: LiDAR water view, roofline clearance, flight-path proxy,
new build 2013-2020, LST heat, canopy, strata-plan age. Residual > 0 = the market pays more than our model.
Also lists the buildings with the largest within-building spread of asking rents (same bedrooms) for the pitch.

Usage (from rents-heat):  .venv/Scripts/python.exe src/rents/public_insights.py --data real [--window B]
Writes data_private/out[/synthetic]/public_insights_<window>.csv and .md. Nothing here feeds the submission.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rents.load import ROOT, cfg, load_adverts, path_for, read_input, test_slice, train_slice, window  # noqa: E402
from rents.model import advert_parking, fit  # noqa: E402
from rents.public_features import _paths, public_matrix  # noqa: E402


def band(s, edges, labels):
    return pd.cut(s, edges, labels=labels, include_lowest=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "real", "apiproxy"], required=True)
    ap.add_argument("--window", choices=["A", "B"], default="B")
    ap.add_argument("--adverts", choices=["auto", "past_leases", "api", "both"], default="auto")
    a = ap.parse_args(argv)
    c = cfg()
    out_dir = path_for(c, a.data, "out")
    flats = read_input(path_for(c, a.data, "input"), c)
    adverts, _ = load_adverts(c, a.data, a.adverts)
    w = window(c, a.window)
    tr, te = train_slice(adverts, w["train_end"]), test_slice(adverts, w["test_start"], w["test_end"])
    import json
    sys.path.insert(0, ROOT)
    from run_all import load_frozen, predict_frozen
    fr = load_frozen(c, flats, adverts, out_dir, a.data)
    f = fit(flats, tr, w["train_end"], c["model"])
    pred = predict_frozen(c, flats, f, fr, tr)
    t = te[te["gnaf_id"].isin(flats["gnaf_id"])].merge(flats[["gnaf_id", "park_in", "bed", "bkey"]], on="gnaf_id") \
        .merge(pred.drop(columns=["bkey"]), on="gnaf_id", how="left")
    pa = advert_parking(t["car_spaces"], t["park_in"]).fillna(0).values
    t["yhat"] = np.where(pa == 1, t["rent_with_car_space"], t["rent_without_car_space"])
    t["resid_log"] = np.log(t["price"] / t["yhat"])
    t["abs_err"] = (t["price"] - t["yhat"]).abs()

    feats = public_matrix(c, flats)
    derived, _ = _paths(c)
    b = pd.read_parquet(os.path.join(derived, "buildings_features_v2.parquet"),
                        columns=["osm_type", "osm_id", "new_2013_2020", "lst_c", "canopy_mmb_pct", "sp_createdate",
                                 "h_p90_2020", "levels_from_lidar_2020"])
    feats = feats.merge(b, on=["osm_type", "osm_id"], how="left")
    t = t.merge(feats, on="gnaf_id", how="left")
    t["sp_year"] = pd.to_datetime(t["sp_createdate"], errors="coerce").dt.year

    groups = {
        "LiDAR water view (share of directions)": band(t["water_view_frac"], [-0.01, 0, 0.25, 1], ["none", "some (<=25%)", "wide (>25%)"]),
        "LiDAR roofline clear (>=50% of directions)": t["roofline_clear_frac"].ge(0.5).map({True: "clear", False: "blocked"}).where(t["roofline_clear_frac"].notna()),
        "flight path proxy (within 1 km of 16/34 approach)": t["air_path"].map({1.0: "under path", 0.0: "not under"}),
        "new build 2013-2020 (LiDAR)": t["new_2013_2020"].map({True: "new", False: "older"}),
        "LiDAR building height": band(t["h_p90_2020"], [0, 12, 25, 50, 300], ["<12 m", "12-25 m", "25-50 m", ">50 m"]),
        "summer land surface temperature (tertile)": pd.qcut(t["lst_c"], 3, labels=["cool", "mid", "hot"], duplicates="drop") if t["lst_c"].notna().sum() > 30 else None,
        "tree canopy % (tertile)": pd.qcut(t["canopy_mmb_pct"], 3, labels=["low", "mid", "high"], duplicates="drop") if t["canopy_mmb_pct"].notna().sum() > 30 else None,
        "strata plan year (building age proxy)": band(t["sp_year"], [1900, 1980, 2000, 2010, 2030], ["<1980", "1980-99", "2000-09", "2010+"]),
        "own history?": t["own"].map({True: "own history", False: "no history"}),
    }
    rows = []
    for name, g in groups.items():
        if g is None:
            continue
        for lvl, d in t.groupby(g, observed=True):
            if len(d) < 15:
                continue
            rows.append(dict(feature=name, band=str(lvl), n_adverts=int(len(d)), n_flats=int(d["gnaf_id"].nunique()),
                             median_resid_pct=round(100 * float(np.median(d["resid_log"])), 1),
                             median_abs_err=round(float(d["abs_err"].median()), 1),
                             median_rent=round(float(d["price"].median()), 0)))
    tab = pd.DataFrame(rows)
    # within-building spread, same bedrooms, test adverts
    sp = t.groupby(["bkey", "bed"]).agg(n=("price", "size"), flats=("gnaf_id", "nunique"), lo=("price", "min"),
                                         hi=("price", "max"), med=("price", "median")).reset_index()
    sp = sp[sp["flats"] >= 4].assign(spread=lambda d: d["hi"] - d["lo"]).sort_values("spread", ascending=False).head(10)
    tab.to_csv(os.path.join(out_dir, f"public_insights_{a.window}.csv"), index=False)
    with open(os.path.join(out_dir, f"public_insights_{a.window}.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# Public-layer insights, window {a.window} test adverts ({w['test_start'].date()}..{w['test_end'].date()}), "
                 f"frozen rule {fr['method']}\n\nmedian_resid_pct > 0: market pays more than the model predicts for that band "
                 f"(after suburb/bedrooms/parking/own-history/building comps). n < 15 bands omitted.\n\n")
        fh.write(tab.to_markdown(index=False) if hasattr(tab, "to_markdown") else tab.to_string(index=False))
        fh.write("\n\n## Largest within-building spread of asking rents (same bedrooms, >=4 flats)\n\n")
        fh.write(sp.to_string(index=False))
    pd.set_option("display.width", 200)
    print(tab.to_string(index=False))
    print("\nlargest within-building spread:\n", sp.to_string(index=False))
    print(f"\nwrote {out_dir}/public_insights_{a.window}.csv/.md  (test adverts {len(t)}, joined to a building: {int(t['osm_id'].notna().sum())})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
