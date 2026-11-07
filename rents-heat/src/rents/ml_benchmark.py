"""Our evidence-weighting AVM vs a regression AVM vs gradient-boosted trees, on the SAME held-out adverts.

Leak-free stacking: the learned models are trained on labels they could have seen. For training origins
T in {2025-09-30, 2025-11-30, 2026-01-31, 2026-03-31}, features are built from adverts <= T and the labels are the
adverts in the two months after T. The trained model is then applied with features built at the evaluation cut-off
(2026-05-31) to the Jun-Jul 2026 adverts (window B) - the same 2,898 adverts our rule scored $45 on.

Models
  R1 hedonic regression (classic AVM): OLS of log rent on suburb x bedrooms, bathrooms, parking, floor, LiDAR views,
     building height, strata age. No rental history.
  R2 regression on our engineered evidence: OLS of log rent on log own-history estimate (+ age, stale flag), log suburb
     ladder, log one-vote comps (+ n neighbours), parking, bathrooms, LiDAR views.
  ML gradient-boosted trees (L1 loss, sklearn HistGradientBoosting) on everything R1 and R2 see, with suburb as a
     categorical feature.
Usage (from rents-heat): .venv/Scripts/python.exe src/rents/ml_benchmark.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from rents.load import cfg, load_adverts, path_for, read_input  # noqa: E402
from rents.model import advert_parking, fit, metrics  # noqa: E402
from rents.public_features import public_matrix  # noqa: E402

TRAIN_ORIGINS = ["2025-09-30", "2025-11-30", "2026-01-31", "2026-03-31"]
EVAL_ORIGIN = "2026-05-31"


def features(c, flats, adverts, T, feats_pub):
    T = pd.Timestamp(T)
    tr = adverts[adverts["date"] <= T]
    f = fit(flats, tr, T, c["model"])
    t = f.table.copy()
    X = pd.DataFrame({"gnaf_id": t["gnaf_id"]})
    X["own"] = t["own"].astype(float)
    X["log_own"] = np.where(t["own"], np.log(t["floor_without"]), np.nan)
    X["own_age_m"] = t["own_age_m"]
    X["stale"] = (t["own_age_m"] > 36).astype(float).where(t["own"])
    X["log_ladder"] = np.log(t["ladder_without"])
    X["log_comps"] = t["comp_onevote"]
    X["n_comp_flats"] = t["n_comp_flats"]
    X["prem"] = t["prem"]
    X = X.merge(flats[["gnaf_id", "suburb_name", "bed", "bathrooms", "floor_level", "flats_in_building"]], on="gnaf_id", how="left")
    X = X.merge(feats_pub, on="gnaf_id", how="left")
    return X, tr


def labelled(X, flats, adverts, T):
    T = pd.Timestamp(T)
    te = adverts[(adverts["date"] > T) & (adverts["date"] <= T + pd.offsets.MonthEnd(2)) & adverts["gnaf_id"].isin(flats["gnaf_id"])]
    te = te.drop(columns=["bathrooms", "bedrooms"], errors="ignore")
    d = te.merge(flats[["gnaf_id", "park_in"]], on="gnaf_id").merge(X, on="gnaf_id", how="left")
    d["pa"] = advert_parking(d["car_spaces"], d["park_in"]).fillna(0).values
    d["y"] = np.log(d["price"])
    return d


def design(d, kind, cats=None):
    num_pub = ["water_view_frac", "park_view_frac", "roofline_clear_frac", "ndsm10", "new_build", "age_decades"]
    if kind == "R1":
        sb = (d["suburb_name"].astype(str) + "_" + d["bed"].astype(str))
        D = pd.get_dummies(sb, prefix="sb", dtype=float)
        if cats is not None:
            D = D.reindex(columns=cats, fill_value=0.0)
        Z = pd.concat([D, d[["pa", "bathrooms", "floor_level"] + num_pub].astype(float)], axis=1)
        return Z.fillna(Z.mean()), list(D.columns)
    cols = ["own", "own_age_m", "stale", "log_ladder", "n_comp_flats", "prem", "pa", "bathrooms"] + num_pub[:3]
    Z = d[cols].astype(float).copy()
    Z["own_x_log_own"] = d["log_own"].fillna(0) * d["own"]
    Z["has_comps"] = d["log_comps"].notna().astype(float)
    Z["comps_x_log"] = d["log_comps"].fillna(d["log_ladder"]) * Z["has_comps"]
    Z["stale_x_own"] = Z["stale"].fillna(0) * Z["own_x_log_own"]
    return Z.fillna(0), None


def ols(Z, y):
    A = np.c_[np.ones(len(Z)), Z.values]
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return lambda Zn: np.c_[np.ones(len(Zn)), Zn.values] @ beta


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    c = cfg()
    out_dir = path_for(c, "real", "out")
    flats = read_input(path_for(c, "real", "input"), c)
    adverts, _ = load_adverts(c, "real", "both")
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        pub = public_matrix(c, flats)[["gnaf_id", "water_view_frac", "park_view_frac", "roofline_clear_frac", "ndsm10", "new_build", "age_decades"]]
    train = []
    for T in TRAIN_ORIGINS:
        X, _ = features(c, flats, adverts, T, pub)
        train.append(labelled(X, flats, adverts, T))
        print(f"train origin {T}: {len(train[-1])} labelled adverts", flush=True)
    tr = pd.concat(train, ignore_index=True)
    Xe, _ = features(c, flats, adverts, EVAL_ORIGIN, pub)
    te = labelled(Xe, flats, adverts, EVAL_ORIGIN)
    y = te["price"].values
    res = {}
    Z1, cats = design(tr, "R1"); Z1e, _ = design(te, "R1", cats)
    res["R1 hedonic regression (no history)"] = metrics(y, np.exp(ols(Z1, tr["y"].values)(Z1e)))
    Z2, _ = design(tr, "R2"); Z2e, _ = design(te, "R2")
    res["R2 regression on our engineered evidence"] = metrics(y, np.exp(ols(Z2, tr["y"].values)(Z2e)))
    gcols = ["own", "log_own", "own_age_m", "stale", "log_ladder", "log_comps", "n_comp_flats", "prem", "pa", "bed", "bathrooms",
             "floor_level", "flats_in_building", "water_view_frac", "park_view_frac", "roofline_clear_frac", "ndsm10", "new_build", "age_decades", "suburb_code"]
    codes = {s: i for i, s in enumerate(sorted(flats["suburb_name"].unique()))}
    for d in (tr, te):
        d["suburb_code"] = d["suburb_name"].map(codes)
    gb = HistGradientBoostingRegressor(loss="absolute_error", max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                       categorical_features=[gcols.index("suburb_code")], random_state=0)
    gb.fit(tr[gcols].astype(float), tr["y"].values)
    res["ML gradient-boosted trees (L1) on all features"] = metrics(y, np.exp(gb.predict(te[gcols].astype(float))))
    ours = json.load(open(os.path.join(out_dir, "metrics_B.json"), encoding="utf-8"))
    k = next(x for x in ours["variants"] if x.startswith("frozen"))
    res["OURS evidence-weighting AVM (submitted rule)"] = ours["variants"][k]["all"]
    print(f"\nwindow B test adverts: {len(te)} (train rows for learned models: {len(tr)})")
    for name, m in res.items():
        print(f"{name:<52} MedAE ${m['medae']:.1f}  within $100 {m['within100']:.1%}  R2 {m['r2']:.3f}")
    json.dump(dict(train_origins=TRAIN_ORIGINS, eval_origin=EVAL_ORIGIN, n_train=len(tr), n_test=len(te), results=res),
              open(os.path.join(out_dir, "ml_benchmark.json"), "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
