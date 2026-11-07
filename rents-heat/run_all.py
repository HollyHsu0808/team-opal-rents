"""Run the rent pipeline end to end.

  python run_all.py --data synthetic --window A          # select rules (k grid, floor vs plain vs one-vote), freeze
  python run_all.py --data synthetic --window B          # evaluate the frozen rule once
  python run_all.py --data real --window final --team NAME   # predict all input flats -> rents_NAME.csv

Outputs: real -> data_private/out/, synthetic -> data_private/out/synthetic/ (metrics_<window>.json, frozen_rule.json,
rents_<team>.csv). Training rows are hard-asserted <= the window's train_end; nothing after 2026-07-31 is ever loaded
for training.
"""
import argparse
import ctypes
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from rents.load import (cfg, load_adverts, path_for, read_input, read_scoring, test_slice, train_slice,  # noqa: E402
                        window)
from rents.model import combine, combine_rule, fit, round_half, score  # noqa: E402

OUT_COLS = ["gnaf_id", "rent_with_car_space", "rent_without_car_space"]


def peak_mem_mb():
    try:
        if os.name == "nt":
            class PMC(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
            k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
            return round(pmc.PeakWorkingSetSize / 2 ** 20, 1)
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except Exception:
        return None


def check_contract(sub: pd.DataFrame, input_ids) -> list:
    """Official output contract. Returns a list of problems (empty = pass)."""
    errs = []
    if list(sub.columns) != OUT_COLS:
        errs.append(f"columns {list(sub.columns)} != {OUT_COLS}")
        return errs
    ids = set(pd.Series(input_ids).astype(str))
    got = sub["gnaf_id"].astype(str)
    if got.duplicated().any():
        errs.append(f"{int(got.duplicated().sum())} duplicate gnaf_id")
    if set(got) - ids:
        errs.append(f"{len(set(got) - ids)} extra gnaf_id not in input")
    if ids - set(got):
        errs.append(f"{len(ids - set(got))} input gnaf_id missing")
    if got.str.strip().eq("").any() or sub["gnaf_id"].isna().any():
        errs.append("blank gnaf_id")
    for c in OUT_COLS[1:]:
        v = pd.to_numeric(sub[c], errors="coerce")
        if v.isna().any():
            errs.append(f"{int(v.isna().sum())} blank/non-numeric {c}")
        elif not np.isfinite(v).all():
            errs.append(f"{int((~np.isfinite(v)).sum())} non-finite {c}")
        elif (v <= 0).any():
            errs.append(f"non-positive {c}")
        elif not np.allclose(v * 2, np.round(v * 2)):
            errs.append(f"{c} not rounded to $0.50")
    if not errs:
        w, wo = pd.to_numeric(sub[OUT_COLS[1]]), pd.to_numeric(sub[OUT_COLS[2]])
        if (w < wo).any():
            errs.append(f"{int((w < wo).sum())} rows with rent_with_car_space < rent_without_car_space")
    return errs


def print_table(title, variants):
    print(f"\n{title}")
    print(f"{'variant':<16}{'slice':<13}{'n':>7}{'MedAE':>9}{'<=100':>8}{'R2':>8}{'cover':>7}")
    for name, s in variants.items():
        for sl in ("own_history", "no_history", "all"):
            m = s[sl]
            f = lambda x, p: "-" if x is None else f"{x:.{p}f}"  # noqa: E731
            print(f"{name:<16}{sl:<13}{m['n']:>7}{f(m['medae'], 1):>9}{f(m['within100'], 3):>8}{f(m['r2'], 3):>8}"
                  f"{f(m['coverage'], 2):>7}")
        ca = s.get("comps_applied")
        if ca and ca["test_adverts"]:
            print(f"{'':<16}comps applied to {ca['test_adverts']} test adverts / {ca['test_flats']} flats / "
                  f"{ca['test_buildings']} buildings")


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)


def run_window_A(c, flats, adverts, out_dir, data, ablation=False):
    w = window(c, "A")
    tr = train_slice(adverts, w["train_end"])
    te = test_slice(adverts, w["test_start"], w["test_end"])
    f = fit(flats, tr, w["train_end"], c["model"])
    mc = c["model"]
    grid = mc["k_grid"]
    stale_opts = [(None, 0.0)] + [(T, sw) for T in mc.get("stale_grid_months", [24, 36, 60]) for sw in mc.get("stale_weights", [0.0, 0.5])]
    variants = {"floor": score(combine(f, "floor"), flats, te)}
    rules = {"floor": dict(method="floor", k=None, stale_m=None, stale_w=0.0)}
    for sm, sw in stale_opts:  # pre-specified candidate grid: comps method x shrinkage x stale-own-history handling
        tag = "" if sm is None else f"_stale{sm}w{sw}"
        rules[f"plain{tag}"] = dict(method="plain", k=None, stale_m=sm, stale_w=sw)
        for k in grid:
            rules[f"onevote_k{k}{tag}"] = dict(method="onevote", k=k, stale_m=sm, stale_w=sw)
    for name, r in rules.items():
        if name != "floor":
            variants[name] = score(combine(f, r["method"], r["k"] or 0.0, r["stale_m"], r["stale_w"]), flats, te)
    def key(name):  # None metrics (no scorable test adverts) sort last instead of crashing
        m = variants[name]["all"]
        return (m["medae"] if m["medae"] is not None else float("inf"), -(m["within100"] or 0.0))
    best = min((n for n in rules if n != "floor"), key=key)
    fl, ov = variants["floor"]["all"], variants[best]["all"]
    validated = ov["n"] > 0 and ov["medae"] is not None and fl["medae"] is not None
    accept = validated and ov["medae"] < fl["medae"] and ov["within100"] >= fl["within100"]
    if not validated:
        print("WARNING: window A has no scorable test adverts -> validation unavailable, freezing the floor")
    chosen = dict(rules[best]) if accept else dict(rules["floor"])
    # snapping on top of the chosen rule: keep only if MedAE improves (ties keep no snapping)
    snap_scores = {}
    if validated:
        for s in mc.get("snap_grid", [5, 10]):
            snap_scores[s] = score(combine(f, chosen["method"], chosen["k"] or 0.0, chosen["stale_m"], chosen["stale_w"], s), flats, te)
    best_snap = min([None] + list(snap_scores), key=lambda s: ((variants[best if accept else "floor"] if s is None else snap_scores[s])["all"]["medae"]))
    chosen["snap"] = best_snap
    frozen = dict(**chosen, candidate=best if accept else "floor", selected_on="A" if validated else "A-unavailable",
                  data=data, validation="ok" if validated else "unavailable: no scorable test adverts in window A",
                  rule="best of {plain, one-vote k grid} x {stale handling} on overall MedAE (tie: share<=100); accepted "
                       "only if overall MedAE beats the floor and share<=100 does not drop; snap kept only if MedAE improves",
                  floor_all=fl, best_all=ov, snap_all={str(s): v["all"]["medae"] for s, v in snap_scores.items()},
                  frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    best_k = chosen.get("k")
    write_json(os.path.join(out_dir, "frozen_rule.json"), frozen)
    res = dict(window=w, fit=f.info, variants=variants, selection=frozen)
    if ablation:
        from rents.public_features import ablation_A
        res["public_ablation"] = ablation_A(c, flats, f, tr, te, frozen)
        pr = res["public_ablation"].get("public_rule")
        if pr:
            frozen["public"] = pr
            write_json(os.path.join(out_dir, "frozen_rule.json"), frozen)
            print(f"PUBLIC ADJUSTMENT: {res['public_ablation'].get('result')} "
                  f"(A no-history {pr.get('A_no_history')}, all {pr.get('A_all')})")
    else:
        res["public_ablation"] = "not run: incremental value untested"
    show = ["floor", "plain", f"onevote_k{grid[0]}", best] if best not in ("plain", f"onevote_k{grid[0]}") else ["floor", "plain", f"onevote_k{grid[0]}"]
    print_table(f"window A (train <= {w['train_end'].date()}, test {w['test_start'].date()}..{w['test_end'].date()})",
                {k: variants[k] for k in show})
    ranked = sorted((n for n in rules if n != "floor"), key=key)[:8]
    print("\ntop candidates (all MedAE / <=100): " + ", ".join(f"{n}: {variants[n]['all']['medae']} / {variants[n]['all']['within100']}" for n in ranked))
    print(f"no stale handling: plain {variants['plain']['all']['medae']}, " + ", ".join(f"k={k}: {variants[f'onevote_k{k}']['all']['medae']}" for k in grid))
    if snap_scores:
        print("snapping (all MedAE): none " + str(ov["medae"] if accept else fl["medae"]) + ", " + ", ".join(f"${s}: {v['all']['medae']}" for s, v in snap_scores.items()))
    print(f"FROZEN: {frozen['candidate']} -> method={frozen['method']} k={best_k} stale_m={chosen['stale_m']} "
          f"stale_w={chosen['stale_w']} snap={best_snap} -> {os.path.join(out_dir, 'frozen_rule.json')}")
    res["candidates_all"] = {n: variants[n]["all"] for n in rules}
    return res


def load_frozen(c, flats, adverts, out_dir, data, force_floor=False):
    p = os.path.join(out_dir, "frozen_rule.json")
    if force_floor:  # 12:00 fallback: floor estimate only, no window-A selection needed
        frozen = dict(method="floor", k=None, selected_on="forced", data=data,
                      rule="floor forced by --force-floor (no A validation)", frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        write_json(p, frozen)
        print(f"FORCED: method=floor -> {p}")
        return frozen
    if not os.path.exists(p):
        print("no frozen rule yet: running window A selection first")
        run_window_A(c, flats, adverts, out_dir, data)
    with open(p) as fh:
        fr = json.load(fh)
    if fr.get("data") != data:
        raise SystemExit(f"frozen rule at {p} was selected on data={fr.get('data')}, not {data}; rerun --window A")
    return fr


def predict_frozen(c, flats, f, fr, train=None):
    """combine() with the frozen method, plus the window-A public adjustments the frozen rule accepted:
    `public_within` (P1: LiDAR floor-view difference to the comparable flats, within-building coefficients re-estimated
    on this window's training adverts) and `public` (cross-sectional adjustment on no-history rows)."""
    if fr.get("public_within") and train is not None:
        from rents.public_features import public_matrix
        from rents.public_within import apply_p1, signals, snap, within_beta
        pred = combine_rule(f, dict(fr, snap=None))
        feats = public_matrix(c, flats)
        one = signals(f, flats, train, c["model"])
        bw, n_rows, _ = within_beta(one, feats)
        pred, n_adj = apply_p1(pred, f, flats, one, feats, bw, fr.get("k") or 0.5)
        pred = snap(pred, fr.get("snap"))
        print(f"P1 within-building floor-view adjustment: beta {np.round(bw, 4).tolist()} from {n_rows:,} flats; {n_adj:,} rows adjusted")
    else:
        pred = combine_rule(f, fr)
    pub = fr.get("public") or {}
    if pub.get("apply"):
        from rents.public_features import apply_public, public_matrix
        pred = apply_public(pred, public_matrix(c, flats), pub)
        print("public adjustment applied to no-history rows (accepted on window A)")
    return pred


def run_window_B(c, flats, adverts, out_dir, data, force_floor=False):
    fr = load_frozen(c, flats, adverts, out_dir, data, force_floor)
    w = window(c, "B")
    tr = train_slice(adverts, w["train_end"])
    te = test_slice(adverts, w["test_start"], w["test_end"])
    f = fit(flats, tr, w["train_end"], c["model"])
    label = (f"frozen_{fr['method']}" + (f"_stale{fr['stale_m']}" if fr.get("stale_m") else "")
             + (f"_snap{fr['snap']}" if fr.get("snap") else "") + ("+public" if (fr.get("public") or {}).get("apply") else ""))[:15]
    variants = {label: score(predict_frozen(c, flats, f, fr, tr), flats, te)}
    if fr["method"] != "floor":
        variants["reference_floor"] = score(combine(f, "floor"), flats, te)
    print_table(f"window B (train <= {w['train_end'].date()}, test {w['test_start'].date()}..{w['test_end'].date()})",
                variants)
    return dict(window=w, fit=f.info, frozen=fr, variants=variants)


def run_final(c, flats, adverts, out_dir, data, team, force_floor=False):
    fr = load_frozen(c, flats, adverts, out_dir, data, force_floor)
    w = window(c, "final")
    tr = train_slice(adverts, w["train_end"])
    f = fit(flats, tr, w["train_end"], c["model"])
    pred = predict_frozen(c, flats, f, fr, tr)
    n_nan = int(pred[OUT_COLS[1:]].isna().any(axis=1).sum())
    if n_nan:  # safety net: citywide median of the column
        for col in OUT_COLS[1:]:
            pred[col] = pred[col].fillna(pred[col].median())
    sub = pred[OUT_COLS].copy()
    sub["rent_with_car_space"] = round_half(sub["rent_with_car_space"])
    sub["rent_without_car_space"] = round_half(sub["rent_without_car_space"])
    errs = check_contract(sub, flats["gnaf_id"])
    out_csv = os.path.join(out_dir, f"rents_{team}.csv")
    if errs:
        print("CONTRACT CHECK FAILED:", errs)
        sub.to_csv(out_csv + ".FAILED", index=False)
        raise SystemExit(2)
    sub.to_csv(out_csv, index=False)
    print(f"\nCONTRACT CHECK PASS: {len(sub)} rows, {sub['gnaf_id'].nunique()} unique ids = input "
          f"{flats['gnaf_id'].nunique()}, no blanks, $0.50 rounding -> {out_csv}")
    res = dict(window=w, fit=f.info, frozen=fr, contract="pass", n_rows=len(sub), n_nan_filled=n_nan,
               n_used_comps=int(pred["used_comps"].sum()), n_own_history=int(pred["own"].sum()), output=out_csv)
    if data == "synthetic":
        sc = read_scoring(path_for(c, data, "scoring"))
        s = c["windows"]["synthetic_scoring"]
        sc = test_slice(sc, pd.Timestamp(s["test_start"]), pd.Timestamp(s["test_end"]))
        scored = pred.copy()
        scored["rent_with_car_space"] = sub["rent_with_car_space"].values
        scored["rent_without_car_space"] = sub["rent_without_car_space"].values
        variants = {f"final_{fr['method']}": score(scored, flats, sc)}
        print_table("final vs synthetic Aug-Sep 2026 scoring adverts (synthetic only)", variants)
        res["synthetic_scoring"] = variants
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "real", "apiproxy"], required=True)
    ap.add_argument("--window", choices=["A", "B", "final"], required=True)
    ap.add_argument("--team", default="team")
    ap.add_argument("--adverts", choices=["auto", "past_leases", "api", "both"], default="auto",
                    help="auto = past-leases file if present, else API parquet")
    ap.add_argument("--n-flats", type=int, default=8000, help="synthetic only, when (re)generating")
    ap.add_argument("--regen", action="store_true", help="synthetic only: regenerate the synthetic files")
    ap.add_argument("--ablation", action="store_true", help="window A: run the pre-specified public-feature ablation")
    ap.add_argument("--force-floor", action="store_true",
                    help="B/final: skip window-A selection and use the floor estimate only (12:00 fallback)")
    a = ap.parse_args(argv)
    t0 = time.time()
    c = cfg()
    out_dir = path_for(c, a.data, "out")
    os.makedirs(out_dir, exist_ok=True)
    if a.data == "synthetic" and (a.regen or not os.path.exists(path_for(c, "synthetic", "input"))):
        sys.path.insert(0, os.path.join(ROOT, "synthetic"))
        from make_synthetic import make
        print("synthetic:", make(n_flats=a.n_flats))
    flats = read_input(path_for(c, a.data, "input"), c)
    adverts, ainfo = load_adverts(c, a.data, a.adverts)
    print(f"data={a.data}: {len(flats)} input flats, {flats['bkey'].nunique()} building keys, adverts {ainfo}")
    if a.window == "A":
        res = run_window_A(c, flats, adverts, out_dir, a.data, a.ablation)
    elif a.window == "B":
        res = run_window_B(c, flats, adverts, out_dir, a.data, a.force_floor)
    else:
        res = run_final(c, flats, adverts, out_dir, a.data, a.team, a.force_floor)
    res.update(data=a.data, adverts=ainfo, n_input_flats=len(flats), runtime_s=round(time.time() - t0, 1),
               peak_mem_mb=peak_mem_mb())
    mp = os.path.join(out_dir, f"metrics_{a.window}.json")
    write_json(mp, res)
    print(f"\nmetrics -> {mp}  runtime {res['runtime_s']} s  peak working set {res['peak_mem_mb']} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
