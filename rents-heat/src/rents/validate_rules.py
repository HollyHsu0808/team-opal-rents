"""Validate every rule of the method with walk-forward origins (time-series cross-validation).

Origins (train on adverts <= origin, test on the next two months - the task's horizon):
  structural rules: 2025-03-31, 2025-06-30, 2025-09-30, 2025-12-31, 2026-03-31, 2026-05-31
  fixed constants : 2025-06-30, 2025-12-31, 2026-03-31, 2026-05-31 (each alternative needs a refit)
Each variant changes ONE rule against the submitted rule. Per variant we report, over origins: mean MedAE, mean share
within $100, and on how many origins the submitted setting is better / worse (MedAE, tie -> share within $100).
Results are saved after every origin so a stopped run keeps what it finished.

Usage (from rents-heat; run each part in the foreground):
  .venv/Scripts/python.exe -u src/rents/validate_rules.py structural
  .venv/Scripts/python.exe -u src/rents/validate_rules.py params 1     (groups 1..4)
  .venv/Scripts/python.exe -u src/rents/validate_rules.py summary
"""
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from rents.load import cfg, load_adverts, path_for, read_input  # noqa: E402
from rents.model import combine_rule, fit, score  # noqa: E402

OUT = None
S_ORIGINS = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31", "2026-05-31"]
P_ORIGINS = ["2025-06-30", "2025-12-31", "2026-03-31", "2026-05-31"]
PARAM_GROUPS = {
    1: {"episode_gap_days": [30, 90], "ladder_min_n": [5, 20], "ladder_lookback_months": [6, 24]},
    2: {"comps_lookback_months": [24, 60], "premium_min_n": [5, 20], "premium_clip": [[0.0, 0.25], [0.0, 0.5]]},
    3: {"index_smooth": [1.0, 10.0], "suburb_index_ridge": [0.3, 3.0], "bed_cap": [3, 5]},
    4: {"own_same_bed": [False], "price_bounds": [[150, 5000]]},
}


def snap(pred, s):
    if not s:
        return pred
    out = pred.copy()
    for col in ("rent_with_car_space", "rent_without_car_space"):
        out[col] = np.round(out[col] / s) * s
    return out


def p1(c, flats, f, tr, pred, feats, k):
    from rents.public_within import apply_p1, signals, within_beta
    one = signals(f, flats, tr, c["model"])
    bw, _, _ = within_beta(one, feats)
    out, _ = apply_p1(pred, f, flats, one, feats, bw, k)
    return out


def extrapolate(f, pred, months=1.5):
    city = np.asarray(f.index.city)
    trend = (city[-1] - city[-7]) / 6 if len(city) > 7 else 0.0
    out = pred.copy()
    for col in ("rent_with_car_space", "rent_without_car_space"):
        out[col] = out[col] * np.exp(trend * months)
    return out


def metrics_row(origin, variant, s):
    return dict(origin=origin, variant=variant, n=s["all"]["n"], medae=s["all"]["medae"], within100=s["all"]["within100"],
                r2=s["all"]["r2"], no_hist_medae=s["no_history"]["medae"])


def setup():
    c = cfg()
    out_dir = path_for(c, "real", "out")
    fr = json.load(open(os.path.join(out_dir, "frozen_rule.json"), encoding="utf-8"))
    flats = read_input(path_for(c, "real", "input"), c)
    cw = copy.deepcopy(c)
    cw["model"]["price_min"], cw["model"]["price_max"] = 1, 10 ** 9
    adverts, _ = load_adverts(cw, "real", "both")
    return c, out_dir, fr, flats, adverts


def window_data(c, adverts, origin, mc):
    T = pd.Timestamp(origin)
    a = adverts[(adverts["price"] >= mc["price_min"]) & (adverts["price"] <= mc["price_max"])]
    return T, a[a["date"] <= T], a[(a["date"] > T) & (a["date"] <= T + pd.offsets.MonthEnd(2))]


def save(path, rows):
    json.dump(rows, open(path, "w"), indent=1)


def structural():
    c, out_dir, fr, flats, adverts = setup()
    from rents.public_features import public_matrix
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        feats = public_matrix(c, flats)
    k = fr.get("k") or 0.5
    path = os.path.join(out_dir, "validate_structural.json")
    rows = json.load(open(path)) if os.path.exists(path) else []
    done = {r["origin"] for r in rows}
    for origin in S_ORIGINS:
        if origin in done:
            continue
        t0 = time.time()
        T, tr, te = window_data(c, adverts, origin, c["model"])
        f = fit(flats, tr, T, c["model"])
        base = combine_rule(f, dict(fr, snap=None))
        full = snap(p1(c, flats, f, tr, base, feats, k), 5)
        V = {"submitted rule": full,
             "floor only (no comps, no stale rule)": snap(combine_rule(f, dict(fr, method="floor", stale_m=None, snap=None)), 5),
             "no building comps": snap(p1(c, flats, f, tr, combine_rule(f, dict(fr, method="floor", snap=None)), feats, k), 5),
             "plain comps instead of one vote": snap(p1(c, flats, f, tr, combine_rule(f, dict(fr, method="plain", snap=None)), feats, k), 5),
             "no stale rule": snap(p1(c, flats, f, tr, combine_rule(f, dict(fr, stale_m=None, snap=None)), feats, k), 5),
             "stale threshold 24 months": snap(p1(c, flats, f, tr, combine_rule(f, dict(fr, stale_m=24, snap=None)), feats, k), 5),
             "stale threshold 60 months": snap(p1(c, flats, f, tr, combine_rule(f, dict(fr, stale_m=60, snap=None)), feats, k), 5),
             "stale blend 50% instead of replace": snap(p1(c, flats, f, tr, combine_rule(f, dict(fr, stale_w=0.5, snap=None)), feats, k), 5),
             "one-vote k = 2": snap(p1(c, flats, f, tr, combine_rule(f, dict(fr, k=2, snap=None)), feats, 2), 5),
             "one-vote k = 5": snap(p1(c, flats, f, tr, combine_rule(f, dict(fr, k=5, snap=None)), feats, 5), 5),
             "no LiDAR floor-view (P1)": snap(base, 5),
             "no snapping": p1(c, flats, f, tr, base, feats, k),
             "snap to $10": snap(p1(c, flats, f, tr, base, feats, k), 10),
             "extrapolate 6-month trend after cut-off": snap(extrapolate(f, p1(c, flats, f, tr, base, feats, k)), 5)}
        for name, p in V.items():
            rows.append(metrics_row(origin, name, score(p, flats, te)))
        save(path, rows)
        s = [r for r in rows if r["origin"] == origin and r["variant"] == "submitted rule"][0]
        print(f"{origin}: n={s['n']} submitted ${s['medae']} / {s['within100']:.3f}  ({time.time() - t0:.0f}s)", flush=True)


def params(group):
    c, out_dir, fr, flats_raw, adverts = setup()
    path = os.path.join(out_dir, f"validate_params_{group}.json")
    rows = json.load(open(path)) if os.path.exists(path) else []
    done = {(r["origin"], r["variant"]) for r in rows}
    rule = dict(fr, snap=None)
    for origin in P_ORIGINS:
        t0 = time.time()
        jobs = [("base", None, None)] + [(f"{k} = {v}", k, v) for k, vs in PARAM_GROUPS[group].items() for v in vs]
        for name, key, val in jobs:
            if (origin, name) in done:
                continue
            c2 = copy.deepcopy(c)
            mc = c2["model"]
            if key == "price_bounds":
                mc["price_min"], mc["price_max"] = val
            elif key is not None:
                mc[key] = val
            flats = flats_raw.copy()
            flats["bed"] = flats["bedrooms"].fillna(1).clip(0, mc["bed_cap"]).astype(int)
            T, tr, te = window_data(c2, adverts, origin, mc)
            f = fit(flats, tr, T, mc)
            rows.append(metrics_row(origin, name, score(snap(combine_rule(f, rule), 5), flats, te)))
            save(path, rows)
        print(f"group {group} {origin} done ({time.time() - t0:.0f}s)", flush=True)


def summary():
    c = cfg()
    out_dir = path_for(c, "real", "out")
    md = ["# Rule validation by walk-forward origins\n",
          "Train on adverts <= origin, test on the next two months. 'better' = the submitted setting has lower MedAE "
          "(tie: higher share within $100) on that origin.\n"]
    verdicts = []
    for fname, ref, title in (("validate_structural.json", "submitted rule", "Structural rules (6 origins, 2025-03 .. 2026-05)"),) + tuple(
            (f"validate_params_{g}.json", "base", f"Fixed constants, group {g} (4 origins)") for g in PARAM_GROUPS):
        p = os.path.join(out_dir, fname)
        if not os.path.exists(p):
            md.append(f"\n## {title}\n\nnot run")
            continue
        df = pd.DataFrame(json.load(open(p)))
        refd = df[df.variant == ref].set_index("origin")
        md += [f"\n## {title}\n", "| variant | origins | mean MedAE | mean within $100 | mean R² | submitted better on | submitted worse on | verdict |",
               "|---|---|---|---|---|---|---|---|"]
        for name, g in df.groupby("variant", sort=False):
            g = g.set_index("origin")
            o = g.index.intersection(refd.index)
            dm = g.loc[o, "medae"] - refd.loc[o, "medae"]
            dw = g.loc[o, "within100"] - refd.loc[o, "within100"]
            better = int(((dm > 0) | ((dm == 0) & (dw < 0))).sum())
            worse = int(((dm < 0) | ((dm == 0) & (dw > 0))).sum())
            if name == ref:
                verdict = "reference"
            elif worse > better and g.loc[o, "medae"].mean() < refd.loc[o, "medae"].mean():
                verdict = "ALTERNATIVE BETTER - review"
            elif better >= worse:
                verdict = "submitted setting supported"
            else:
                verdict = "mixed"
            verdicts.append((title, name, verdict))
            md.append(f"| {name} | {len(o)} | ${g.loc[o, 'medae'].mean():.2f} | {g.loc[o, 'within100'].mean():.2%} | "
                      f"{g.loc[o, 'r2'].mean():.3f} | {better} | {worse} | {verdict} |")
    txt = "\n".join(md)
    open(os.path.join(out_dir, "rule_validation.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "structural":
        structural()
    elif mode == "params":
        params(int(sys.argv[2]))
    else:
        summary()
