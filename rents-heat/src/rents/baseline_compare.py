"""Baseline ladder on IDENTICAL test adverts (windows A and B): government bond data -> organiser-style suburb medians
-> + own history -> our frozen rule. For the pitch and the report.

1. bond_postcode_bed  NSW Fair Trading rental bond lodgements (flats, City of Sydney postcodes): median weekly rent of
                      the last 12 months by postcode x bedrooms (fallback bedrooms). Only files published before the
                      window cut-off: the 2025 annual file and monthly files for months <= cut-off month - 1 (each
                      monthly file appears about a month later). Bond rents are agreed rents at lodgement, not asking
                      rents, and carry no parking -> same value in both columns.
2. suburb_medians     the official "simple estimate" idea on our adverts: suburb x bedrooms x parking medians of the
                      last 12 months (index-adjusted), for every flat, ignoring the flat's own history.
3. floor              2 + the flat's own last advert (same bedrooms) moved with the repeat-rent index.
4. ours (frozen)      the rule frozen on window A (building comps one-flat-one-vote, stale own history, $5 snapping).

Usage (from rents-heat):  .venv/Scripts/python.exe src/rents/baseline_compare.py --data real --adverts both
Writes data_private/out[/<profile>]/baselines.json and baselines.md.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rents.load import ROOT, cfg, load_adverts, path_for, read_input, test_slice, train_slice, window  # noqa: E402
from rents.model import advert_parking, combine_rule, fit, metrics  # noqa: E402

BOND_DIR = os.path.normpath(os.path.join(ROOT, "..", "public-data", "bond"))
SUBURB_POSTCODE = {"Alexandria": 2015, "Barangaroo": 2000, "Beaconsfield": 2015, "Camperdown": 2050,
                   "Centennial Park": 2021, "Chippendale": 2008, "Darlinghurst": 2010, "Darlington": 2008,
                   "Dawes Point": 2000, "Elizabeth Bay": 2011, "Erskineville": 2043, "Eveleigh": 2015,
                   "Forest Lodge": 2037, "Glebe": 2037, "Haymarket": 2000, "Millers Point": 2000, "Moore Park": 2021,
                   "Potts Point": 2011, "Pyrmont": 2009, "Redfern": 2016, "Rosebery": 2018, "Rushcutters Bay": 2011,
                   "Surry Hills": 2010, "Sydney": 2000, "The Rocks": 2000, "Ultimo": 2007, "Waterloo": 2017,
                   "Woolloomooloo": 2011, "Zetland": 2017, "Newtown": 2042, "Paddington": 2021, "St Peters": 2044,
                   "Annandale": 2038}
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}


def load_bond(cutoff):
    """Bond lodgements (flats) from files published before `cutoff`; returns (frame, list of files used)."""
    frames, used = [], []
    for f in sorted(glob.glob(os.path.join(BOND_DIR, "rentalbond_lodgements_*.xlsx"))):
        name = os.path.basename(f)
        m = re.search(r"lodgements_([a-z]+)_(\d{4})", name)
        if "year_2025" in name:
            ok = cutoff >= pd.Timestamp("2026-01-31")
        elif m:
            fm = pd.Timestamp(year=int(m.group(2)), month=MONTHS[m.group(1)], day=1)
            ok = fm <= (cutoff.to_period("M") - 1).to_timestamp()  # published ~1 month after the month
        else:
            ok = False
        if not ok:
            continue
        x = pd.ExcelFile(f)
        sheet = [s for s in x.sheet_names if "definition" not in s.lower()][0]
        raw = pd.read_excel(f, sheet_name=sheet, header=None)
        hdr = raw.index[raw.iloc[:, 0].astype(str).str.strip().str.lower() == "lodgement date"][0]
        d = raw.iloc[hdr + 1:, :5].copy()
        d.columns = ["date", "postcode", "dtype", "bed", "rent"]
        frames.append(d)
        used.append(name)
    d = pd.concat(frames, ignore_index=True)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for c in ("postcode", "bed", "rent"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["date", "postcode", "bed", "rent"])
    d = d[(d["dtype"] == "F") & d["postcode"].isin(set(SUBURB_POSTCODE.values())) & d["rent"].between(150, 5000)]
    d = d[(d["date"] <= cutoff) & (d["date"] > cutoff - pd.DateOffset(months=12))]
    d["bed"] = d["bed"].clip(upper=4).astype(int)
    return d, used


def flat_postcode(flats):
    pc = flats["address"].astype(str).str.extract(r"(\d{4})\s*$")[0]
    pc = pd.to_numeric(pc, errors="coerce")
    return pc.fillna(flats["suburb_name"].map(SUBURB_POSTCODE))


def evaluate(c, flats, adverts, wname, fr):
    w = window(c, wname)
    tr, te = train_slice(adverts, w["train_end"]), test_slice(adverts, w["test_start"], w["test_end"])
    f = fit(flats, tr, w["train_end"], c["model"])
    t = te[te["gnaf_id"].isin(flats["gnaf_id"])].merge(flats[["gnaf_id", "park_in", "bed", "pc"]], on="gnaf_id")
    pa = advert_parking(t["car_spaces"], t["park_in"]).fillna(0).values
    y = t["price"].values
    out = {}
    # 1. bond
    bond, used = load_bond(w["train_end"])
    pb = bond.groupby(["postcode", "bed"])["rent"].median()
    b1 = bond.groupby("bed")["rent"].median()
    yb = np.array([pb.get((p, b), np.nan) for p, b in zip(t["pc"], t["bed"])], float)
    yb = np.where(np.isnan(yb), t["bed"].map(b1).values, yb)
    out["1. government bond data (postcode x bedrooms)"] = dict(metrics(y, yb), files=used, n_bond_rows=int(len(bond)))
    # 2. suburb medians (ladder only)
    tab = f.table.set_index("gnaf_id")
    lw, lwo = t["gnaf_id"].map(tab["ladder_with"]).values, t["gnaf_id"].map(tab["ladder_without"]).values
    out["2. suburb x bedrooms x parking medians (organiser-style simple estimate)"] = metrics(y, np.where(pa == 1, lw, lwo))
    # 3. floor and 4. ours
    for name, rule in (("3. + the flat's own past rent x repeat-rent index", dict(method="floor")),
                       ("4. ours: + same-building comps (one flat, one vote), stale history out, $5 grid", fr)):
        p = combine_rule(f, rule).set_index("gnaf_id")
        yh = np.where(pa == 1, t["gnaf_id"].map(p["rent_with_car_space"]).values, t["gnaf_id"].map(p["rent_without_car_space"]).values)
        out[name] = metrics(y, yh)
    return dict(window=wname, train_end=str(w["train_end"].date()), test=f"{w['test_start'].date()}..{w['test_end'].date()}",
                n_test=int(len(t)), rows=out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "real", "apiproxy"], default="real")
    ap.add_argument("--adverts", choices=["auto", "past_leases", "api", "both"], default="both")
    a = ap.parse_args(argv)
    c = cfg()
    out_dir = path_for(c, a.data, "out")
    flats = read_input(path_for(c, a.data, "input"), c)
    flats["pc"] = flat_postcode(flats)
    adverts, _ = load_adverts(c, a.data, a.adverts)
    fr = json.load(open(os.path.join(out_dir, "frozen_rule.json"), encoding="utf-8"))
    res = [evaluate(c, flats, adverts, wn, fr) for wn in ("A", "B")]
    json.dump(res, open(os.path.join(out_dir, "baselines.json"), "w", encoding="utf-8"), indent=1, default=str)
    lines = ["# Baseline ladder on identical test adverts\n"]
    for r in res:
        lines.append(f"\n## Window {r['window']}: train <= {r['train_end']}, test {r['test']} ({r['n_test']:,} adverts)\n")
        lines.append("| step | median abs error | within $100 | R2 |\n|---|---|---|---|")
        for k, m in r["rows"].items():
            lines.append(f"| {k} | ${m['medae']:.1f} | {100 * m['within100']:.1f}% | {m['r2']:.3f} |")
        bf = r["rows"]["1. government bond data (postcode x bedrooms)"]
        lines.append(f"\nbond files used (published before the cut-off): {', '.join(bf['files'])}; {bf['n_bond_rows']:,} flat lodgements in the last 12 months")
    txt = "\n".join(lines)
    open(os.path.join(out_dir, "baselines.md"), "w", encoding="utf-8").write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
