"""Summarise sim_public_results.csv into compact markdown tables -> sim_public_summary.md. Run after sim_public.py.
Cell format for deltas: mean ΔMedAE over seeds ($, negative = better) and [k↓/k↑] = number of seeds whose paired-bootstrap
95% CI lies entirely below / above 0. Mean CI shown as (lo, hi) averaged over seeds."""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "sim_public_results.csv"))
ns = df.seed.nunique()


def cell(x):
    if x.empty or x.d_medae.isna().all():
        return ""
    return (f"{x.d_medae.mean():+.1f} ({x.ci_lo.mean():+.1f},{x.ci_hi.mean():+.1f}) "
            f"[{int((x.ci_hi < 0).sum())}↓/{int((x.ci_lo > 0).sum())}↑]")


out = [f"# sim_public summary (seeds = {ns}; Δ = MedAE − previous rung; [k↓/k↑] = seeds with CI entirely <0 / >0)\n"]

# 1. ladder per window, plausible scenario
for bn in (0.03, 0.10):
    rows = []
    x = df[(df.scenario == "plausible") & (df.block_noise == bn)]
    for rung in [r for r in x.rung.unique() if not r.endswith("_nomap") and r not in
                 ("G_age", "G_view", "G_station", "G_water", "G_heatcanopy", "G_abs")]:
        row = {"rung": rung, "prev": x[x.rung == rung].prev.iloc[0]}
        for w in ("A", "B", "S"):
            for sl, lab in (("all", "all"), ("no_history", "no-hist")):
                q = x[(x.window == w) & (x.rung == rung) & (x.slice == sl)]
                row[f"{w} MedAE {lab}"] = round(q.medae.mean(), 1)
                if w in ("B", "S"):
                    row[f"{w} Δ {lab}"] = cell(q)
            q = x[(x.window == w) & (x.rung == rung) & (x.slice == "all")]
            if w == "S":
                row["S within100"] = round(q.within100.mean(), 3)
                row["S R2"] = round(q.r2.mean(), 3)
                row["S cov(all)"] = round(q.coverage.mean(), 3)
        rows.append(row)
    out.append(f"\n## Ladder, scenario = plausible, block_noise = {bn}\n")
    out.append(pd.DataFrame(rows).to_markdown(index=False))
    nrow = x[(x.rung == "M0")].groupby(["window", "slice"]).n.mean().unstack().round(0)
    out.append("\nTest sizes (mean): " + "; ".join(f"{w}: " + ", ".join(f"{k}={int(v)}" for k, v in r.items())
                                                 for w, r in nrow.iterrows()))

# 2. public groups vs G_base (window S and B), by scenario
groups = ["age", "view", "station", "water", "heatcanopy", "abs", "all"]
for w in ("S", "B"):
    for sl in ("all", "no_history", "no_history_no_bldg_comps"):
        rows = []
        for bn in (0.03, 0.10):
            for suf in ("", "_nomap"):
                for g in groups:
                    row = {"block_noise": bn, "block map in model": "no" if suf else "yes", "group": g}
                    for scen in ("zero", "plausible", "strong"):
                        q = df[(df.window == w) & (df.scenario == scen) & (df.block_noise == bn) &
                               (df.rung == f"G_{g}{suf}") & (df.slice == sl)]
                        row[scen] = cell(q)
                    rows.append(row)
        n = df[(df.window == w) & (df.rung == "G_base") & (df.slice == sl)].n.mean()
        out.append(f"\n## Window {w}, slice {sl} (mean n = {n:.0f}): G_base + one public group vs G_base\n")
        out.append(pd.DataFrame(rows).to_markdown(index=False))

# 3. diagnostic: all Aug-Sep adverts treated as no-history & no comps
d = df[df.window == "S_diag_nocomp"]
if not d.empty:
    rows = []
    for bn in (0.03, 0.10):
        for suf in ("", "_nomap"):
            for g in groups:
                row = {"block_noise": bn, "block map in model": "no" if suf else "yes", "group": g}
                for scen in ("zero", "plausible", "strong"):
                    row[scen] = cell(d[(d.scenario == scen) & (d.block_noise == bn) & (d.rung == f"G_{g}{suf}")])
                rows.append(row)
    base = d[d.rung == "G_base"].groupby(["scenario", "block_noise"]).medae.mean().round(1)
    nomap = d[d.rung == "G_base_nomap"].groupby(["scenario", "block_noise"]).medae.mean().round(1)
    out.append(f"\n## Diagnostic (large n = {d.n.mean():.0f}): every Aug-Sep advert scored as no-history AND no building "
               f"comps (new-development case)\n\nG_base MedAE: {base.to_dict()}; G_base_nomap MedAE: {nomap.to_dict()}\n")
    out.append(pd.DataFrame(rows).to_markdown(index=False))

txt = "\n".join(out)
with open(os.path.join(HERE, "sim_public_summary.md"), "w", encoding="utf-8") as f:
    f.write(txt)
print(txt)
