"""Real-data baseline study on NSW Rental Bond lodgements (public, NSW Fair Trading).

Mimics the task timing: train on lodgements <= 31 Jul 2026, test on Aug-Sep 2026 lodgements, City of Sydney postcodes, flats only.
Bond data has postcode, bedrooms, weekly rent and date only (no address, no parking), so it measures:
  - how far a location x bedrooms median gets (the official baseline's family)
  - how much residual spread is left for building / unit / parking information
Note: bond rent = agreed rent at lodgement, not asking rent; postcode is coarser than suburb.
"""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data_public", "raw", "bond")
COS_POSTCODES = [2000, 2007, 2008, 2009, 2010, 2011, 2015, 2016, 2017, 2018, 2021, 2037, 2042, 2043, 2050]


def load():
    frames = []
    for f in sorted(glob.glob(os.path.join(RAW, "*.xlsx"))):
        x = pd.ExcelFile(f)
        sheet = [s for s in x.sheet_names if "definition" not in s.lower()][0]
        raw = pd.read_excel(f, sheet_name=sheet, header=None)
        hdr = raw.index[raw.iloc[:, 0].astype(str).str.strip().str.lower() == "lodgement date"][0]
        d = raw.iloc[hdr + 1:, :5].copy()
        d.columns = ["date", "postcode", "dtype", "bed", "rent"]
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["date"] = pd.to_datetime(d.date, errors="coerce")
    for c in ("postcode", "bed", "rent"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["date", "postcode", "bed", "rent"])  # no dedupe: identical anonymous rows can be separate leases; source files do not overlap (2025 annual + 2026 monthly)
    d = d[(d.dtype == "F") & d.postcode.isin(COS_POSTCODES) & (d.rent >= 150) & (d.rent <= 5000) & (d.bed <= 4)]
    d["bed"] = d.bed.astype(int)
    d["postcode"] = d.postcode.astype(int)
    d["m"] = d.date.dt.to_period("M")
    return d


def med_err(pred, y):
    e = np.abs(pred - y)
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return dict(n=len(y), median_abs_err=round(float(np.median(e)), 1), within100=round(float(np.mean(e <= 100)), 3),
                r2=round(float(r2), 3))


def ladder_pred(train, test, ladders, min_n=10):
    out = pd.Series(np.nan, index=test.index)
    for keys in ladders:
        g = train.groupby(keys).rent.agg(["median", "size"])
        g = g[g["size"] >= min_n]["median"].rename("v").reset_index()
        v = test[keys].merge(g, on=keys, how="left").v.values
        out = out.fillna(pd.Series(v, index=test.index))
    return out.fillna(train.rent.median()).values


def main():
    d = load()
    print("rows (CoS flats):", len(d), "| months:", d.m.min(), "->", d.m.max())
    print(d.groupby(d.date.dt.year).size().to_string())
    cut = pd.Timestamp("2026-07-31")
    test = d[(d.date > cut) & (d.date <= "2026-09-30")]
    res = {}
    for label, months in (("last 12m", 12), ("last 6m", 6), ("last 3m", 3)):
        tr = d[(d.date <= cut) & (d.date > cut - pd.DateOffset(months=months))]
        res[f"bedrooms only ({label})"] = med_err(ladder_pred(tr, test, [["bed"]]), test.rent.values)
        res[f"postcode x bedrooms ({label})"] = med_err(ladder_pred(tr, test, [["postcode", "bed"], ["bed"]]), test.rent.values)
    # drift-adjusted: last 12 months medians scaled by citywide (CoS) median growth, month-by-month repeat of cell medians
    tr = d[(d.date <= cut) & (d.date > cut - pd.DateOffset(months=12))].copy()
    mm = tr.groupby(["m", "bed"]).rent.median().unstack()
    rel = (mm / mm.mean()).mean(axis=1)  # monthly level relative to 12m average, bedroom-mix neutral
    slope = np.polyfit(np.arange(len(rel)), np.log(rel.values), 1)[0]
    months_ahead = 6.5 + 1.5  # test centre (mid Aug-Sep) vs centre of 12m training window
    adj = np.exp(slope * months_ahead)
    p = ladder_pred(tr, test, [["postcode", "bed"], ["bed"]]) * adj
    res["postcode x bedrooms (12m, drift-adjusted)"] = med_err(p, test.rent.values)
    # oracle: test-period cell median (upper bound for any location x bedrooms rule at postcode level)
    res["REFERENCE (hindsight, not a lower bound): test-period postcode x bedrooms median"] = med_err(ladder_pred(test, test, [["postcode", "bed"], ["bed"]], 1), test.rent.values)
    out = pd.DataFrame(res).T
    print(out.to_string())
    # within-cell dispersion that only building / unit / parking info can reduce
    cell = test.groupby(["postcode", "bed"]).rent
    resid = np.log(test.rent) - np.log(cell.transform("median"))
    print("within postcode x bed (test period) log-sd:", round(float(resid.std()), 3),
          "| median abs log dev:", round(float(resid.abs().median()), 3))
    print("monthly drift (log/month, 12m window):", round(float(slope), 4))
    print(test.groupby("bed").rent.median().to_string())
    out.to_csv(os.path.join(HERE, "bond_results.csv"))


if __name__ == "__main__":
    main()
