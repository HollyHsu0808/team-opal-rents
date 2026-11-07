"""Strata-plan number -> registration year calibration from NSW cadastre createdate.
createdate = date the lot entered the digital cadastre. For plans registered before the DCDB went live, that is a
bulk digitisation date (many identical dates in 1992-1993), NOT the registration date. We exclude:
  (a) any createdate shared by >= 5 SP plans (bulk loads), and (b) createdate before 1994-01-01.
Then: Spearman(plannumber, createdate), isotonic fit year ~ plannumber, implied year for SP 10k/50k/80k/100k.
Outputs: public-data/_derived/sp_plans.csv, sp_year_calibration.json, sp_year_calibration.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_derived"

fc = json.load(open(ROOT / "cadastre" / "sp_lots_bbox.geojson", encoding="utf-8"))["features"]
rows = []
for f in fc:
    p, g = f["properties"], f["geometry"]
    ring = g["coordinates"][0] if g["type"] == "Polygon" else g["coordinates"][0][0]
    xs, ys = np.array(ring).T[:2]
    rows.append(dict(plannumber=p["plannumber"], planlabel=p["planlabel"], createdate=p["createdate"],
                     startdate=p["startdate"], lon=xs.mean(), lat=ys.mean(), n_vertices=len(ring)))
sp = pd.DataFrame(rows)
for c in ("createdate", "startdate"):
    sp[c] = pd.to_datetime(sp[c], unit="ms")
# planar area (m2) via simple equirectangular approx at -33.9
sp["area_m2_approx"] = np.nan
for i, f in enumerate(fc):
    g = f["geometry"]
    polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    a = 0.0
    for poly in polys:
        x, y = np.array(poly[0]).T[:2]
        x = x * 111320 * np.cos(np.radians(-33.9))
        y = y * 110574
        a += 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    sp.loc[i, "area_m2_approx"] = a
sp = sp.groupby("plannumber", as_index=False).agg(createdate=("createdate", "min"), startdate=("startdate", "min"),
                                                  lon=("lon", "mean"), lat=("lat", "mean"),
                                                  area_m2_approx=("area_m2_approx", "sum"), n_polys=("planlabel", "size"))
dup = sp.createdate.map(sp.createdate.value_counts())
sp["bulk_date"] = (dup >= 5) | (sp.createdate < "1994-01-01")
sp["createdate_iso"] = sp.createdate.dt.strftime("%Y-%m-%d")
sp.to_csv(OUT / "sp_plans.csv", index=False)

ok = sp[~sp.bulk_date].copy()
ok["year"] = ok.createdate.dt.year + (ok.createdate.dt.day_of_year - 1) / 365.25
rho, pval = spearmanr(ok.plannumber, ok.year)
iso = IsotonicRegression(out_of_bounds="nan").fit(ok.plannumber, ok.year)
resid = ok.year - iso.predict(ok.plannumber)
# share of plans that are "out of order" by > 1 year vs the isotonic fit
out_of_order = float((resid.abs() > 1).mean())
lin = np.polyfit(ok.plannumber, ok.year, 1)
targets = [10_000, 50_000, 80_000, 100_000]
implied = {}
for t in targets:
    v = float(iso.predict([t])[0])
    near = ok[(ok.plannumber - t).abs() <= 1500]
    implied[t] = dict(isotonic_year=None if np.isnan(v) else round(v, 2),
                      median_year_within_1500=None if near.empty else round(float(near.year.median()), 2),
                      n_within_1500=int(len(near)))
# Registration-like subset: SP >= 45,000 (numbers issued after the DCDB went live, ~1994) with non-bulk dates.
# Within it, late re-captures sit ABOVE the curve, so use the lower envelope: per 1,000-plan bin, 10th percentile of
# year, made monotone (cumulative max) and linearly interpolated.
reg = ok[ok.plannumber >= 45_000].copy()
rho_reg, _ = spearmanr(reg.plannumber, reg.year)
reg["bin"] = (reg.plannumber // 1000) * 1000 + 500
env = reg.groupby("bin").year.quantile(0.10)
env = env[reg.groupby("bin").size() >= 3].cummax()
reg["env_year"] = np.interp(reg.plannumber, env.index, env.values)
reg_resid = reg.year - reg.env_year
implied_env = {t: (round(float(np.interp(t, env.index, env.values)), 2) if env.index.min() <= t <= env.index.max()
                   else "out of range (pre-DCDB numbers; createdate is a capture date, not registration)") for t in targets}
pd.DataFrame({"sp_bin_mid": env.index, "year_lower_envelope": env.values}).to_csv(OUT / "sp_year_lookup.csv", index=False)
res_reg = dict(n=int(len(reg)), spearman_rho=round(float(rho_reg), 4),
               share_within_1yr_of_envelope=round(float((reg_resid.abs() <= 1).mean()), 3),
               share_more_than_2yr_above=round(float((reg_resid > 2).mean()), 3), implied_year_lower_envelope=implied_env,
               plans_per_year_recent=round(float(1 / np.polyfit(reg.plannumber[reg.year > 2015], reg.env_year[reg.year > 2015], 1)[0]), 0))
res = dict(registration_like_subset_ge45000=res_reg, n_sp_plans=int(len(sp)), n_bulk_or_pre1994_excluded=int(sp.bulk_date.sum()), n_used=int(len(ok)),
           spearman_rho=round(float(rho), 4), spearman_p=float(pval), share_resid_gt_1yr_vs_isotonic=round(out_of_order, 3),
           median_abs_resid_years=round(float(resid.abs().median()), 3),
           plannumber_range_used=[int(ok.plannumber.min()), int(ok.plannumber.max())],
           linear_fit_year_per_10k_plans=round(lin[0] * 1e4, 3), implied=implied,
           createdate_year_counts_all=sp.createdate.dt.year.value_counts().sort_index().to_dict())
json.dump(res, open(OUT / "sp_year_calibration.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if k != "createdate_year_counts_all"}, indent=1, default=str))

fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
bulk = sp[sp.bulk_date]
ax.scatter(bulk.plannumber, bulk.createdate.dt.year + bulk.createdate.dt.day_of_year / 365.25, s=8, c="#B4B4B4",
           alpha=0.6, linewidths=0, label=f"excluded: bulk / pre-1994 dates (n={len(bulk)})")
ax.scatter(ok.plannumber, ok.year, s=8, c="#2A6FDB", alpha=0.6, linewidths=0, label=f"used (n={len(ok)})")
xs = np.linspace(ok.plannumber.min(), ok.plannumber.max(), 400)
ax.plot(env.index, env.values, c="#1B1B1B", lw=2, label="lower envelope (SP>=45k, registration-like)")
ax.set_xlabel("Strata plan number (SP)")
ax.set_ylabel("Cadastre createdate (year)")
ax.set_title(f"SP number vs cadastre createdate, Sydney bbox (Spearman {rho:.3f}; SP>=45k {rho_reg:.3f})", fontsize=11, loc="left")
ax.grid(alpha=0.25, lw=0.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "sp_year_calibration.png")
print("saved plot")
