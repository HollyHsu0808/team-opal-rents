"""Export the final model's parameters (fit on adverts <= 31 Jul 2026) as aggregates for the live-demo page.

Only aggregates leave: monthly index factors per suburb, suburb x bedroom ladder medians, parking premiums, per-building
one-vote comparable signals (mean of neighbour flats, count), LiDAR view per floor (public data) and the frozen rule.
No individual advert, price or flat id is exported. Buildings: the 400 with the most comparable flats.
Usage (from rents-heat): .venv/Scripts/python.exe src/rents/export_demo_params.py  -> data_private/out/demo_params.json
"""
import io, contextlib, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rents.load import cfg, load_adverts, path_for, read_input, train_slice, window  # noqa: E402
from rents.model import fit, mnum  # noqa: E402
from rents.public_features import _paths, flat_features  # noqa: E402
from rents.public_within import signals, within_beta  # noqa: E402

c = cfg()
out_dir = path_for(c, "real", "out")
fr = json.load(open(os.path.join(out_dir, "frozen_rule.json"), encoding="utf-8"))
flats = read_input(path_for(c, "real", "input"), c)
adverts, _ = load_adverts(c, "real", "both")
w = window(c, "final")
tr = train_slice(adverts, w["train_end"])
f = fit(flats, tr, w["train_end"], c["model"])
t = f.table
m_hi = int(mnum(pd.Series([w["train_end"]]))[0])

# index factor to the cut-off, per suburb and month 2016-01 .. 2026-07
months = list(range(2016 * 12, m_hi + 1))
subs = sorted(flats["suburb_name"].unique())
idx = {s: [round(float(np.exp(x)), 4) for x in f.index.log_to_now(np.array([s] * len(months)), np.array(months))] for s in subs}

# ladder and premium per suburb x bed
lad = t.groupby(["suburb_name", "bed"]).agg(without=("ladder_without", "first"), with_=("ladder_with", "first"), prem=("prem", "first")).reset_index()
ladder = {f"{r.suburb_name}|{int(r.bed)}": [round(float(r.without), 1), round(float(r.with_), 1), round(float(r.prem), 4)] for r in lad.itertuples()}

# building comps: one-vote signal per building x bed (mean of neighbour flats' latest adverts, log, without parking)
one = signals(f, flats, tr, c["model"])
g = one.groupby(["bkey", "bed"])["s"].agg(["sum", "size"]).reset_index()
with contextlib.redirect_stdout(io.StringIO()):
    ff = flat_features(flats, c)
from rents.public_features import public_matrix  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    pm = public_matrix(c, flats)
bw, n_rows, n_groups = within_beta(one, pm)
fb = flats[["gnaf_id", "bkey", "suburb_name", "building_address"]].merge(ff[["gnaf_id", "osm_type", "osm_id"]], on="gnaf_id", how="left")
bmap = fb.dropna(subset=["osm_id"]).groupby("bkey").agg(osm_type=("osm_type", "first"), osm_id=("osm_id", "first"),
                                                         suburb=("suburb_name", "first"), label=("building_address", "first"), flats=("gnaf_id", "size"))
derived, _ = _paths(c)
fv = pd.read_parquet(os.path.join(derived, "lidar_floor_view.parquet"), columns=["osm_type", "osm_id", "floor", "water_view_frac", "park_view_frac", "roofline_clear_frac"])
# neighbours' mean view per building x bed: mean over the comparable flats' own view features
nv = one.merge(pm[["gnaf_id", "water_view_frac", "park_view_frac", "roofline_clear_frac"]], on="gnaf_id", how="left").groupby(["bkey", "bed"])[["water_view_frac", "park_view_frac", "roofline_clear_frac"]].mean()
top = g.groupby("bkey")["size"].sum().sort_values(ascending=False)
top = [b for b in top.index if b in bmap.index][:400]
buildings = []
for b in top:
    info = bmap.loc[b]
    views = fv[(fv.osm_type == info.osm_type) & (fv.osm_id == info.osm_id)].sort_values("floor")
    comps = {}
    for r in g[g.bkey == b].itertuples():
        nvr = nv.loc[(b, r.bed)] if (b, r.bed) in nv.index else None
        comps[str(int(r.bed))] = [round(float(r.sum / r.size), 5), int(r.size)] + ([round(float(x), 3) if pd.notna(x) else None for x in nvr.values] if nvr is not None else [None, None, None])
    buildings.append(dict(key=b, label=f"{str(info.label).title()}, {info.suburb}", suburb=info.suburb, flats=int(info.flats), comps=comps,
                          views=[[int(v.floor), round(float(v.water_view_frac), 3), round(float(v.park_view_frac), 3), round(float(v.roofline_clear_frac), 3)] for v in views.itertuples()]))
out = dict(cutoff=str(w["train_end"].date()), month0=months[0], index=idx, ladder=ladder, buildings=buildings,
           rule=dict(method=fr["method"], k=fr.get("k"), stale_months=fr.get("stale_m"), snap=fr.get("snap")),
           view_beta=dict(water=round(float(bw[0]), 4), park=round(float(bw[1]), 4), roofline=round(float(bw[2]), 4)),
           view_beta_n=dict(flats=n_rows, groups=n_groups), premium_citywide=f.info.get("premium_citywide"),
           heldout=dict(medae=45.0, within100=0.771, r2=0.88, n=2898))
p = os.path.join(out_dir, "demo_params.json")
json.dump(out, open(p, "w"), separators=(",", ":"))
print(f"wrote {p}: {os.path.getsize(p)/1e3:.0f} kB, {len(buildings)} buildings, {len(ladder)} ladder cells, view beta {out['view_beta']}")
