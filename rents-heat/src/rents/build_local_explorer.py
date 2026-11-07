"""Local-only explorer: type any address, compare the actual advertised rent with the model's prediction.

Contains organiser data (addresses, advert rents), so it is written to data_private/ (gitignored) and is meant to be
opened from this computer only - never published or committed.

Two views:
  Held-out check: every Jun-Jul 2026 advert (window B) with the prediction the model made from adverts <= 31 May only.
                  This is the honest comparison: the model never saw these rents.
  Any flat:       all 65,777 input flats with the submitted prediction (trained <= 31 Jul) and the flat's latest advert
                  on or before 31 Jul as context (in-sample - not a test).
Usage (from rents-heat): .venv/Scripts/python.exe src/rents/build_local_explorer.py   -> opens nothing, prints the path
"""
import io, contextlib, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from rents.load import cfg, load_adverts, path_for, read_input, test_slice, train_slice, window  # noqa: E402
from rents.model import advert_parking, fit  # noqa: E402
from run_all import predict_frozen  # noqa: E402


def tier(row):
    if row["own"] and not row.get("stale_own", False):
        return "own history"
    if row["used_comps"]:
        return "building neighbours" + (" (own advert stale)" if row.get("stale_own", False) else "")
    return "suburb ladder"


def main():
    c = cfg()
    out_dir = path_for(c, "real", "out")
    fr = json.load(open(os.path.join(out_dir, "frozen_rule.json"), encoding="utf-8"))
    flats = read_input(path_for(c, "real", "input"), c)
    adverts, _ = load_adverts(c, "real", "both")
    A = flats.set_index("gnaf_id")
    # held-out window B
    w = window(c, "B")
    tr, te = train_slice(adverts, w["train_end"]), test_slice(adverts, w["test_start"], w["test_end"])
    f = fit(flats, tr, w["train_end"], c["model"])
    with contextlib.redirect_stdout(io.StringIO()):
        pb = predict_frozen(c, flats, f, fr, tr)
    pb["tier"] = pb.apply(tier, axis=1)
    t = te[te["gnaf_id"].isin(flats["gnaf_id"])].merge(flats[["gnaf_id", "park_in"]], on="gnaf_id").merge(pb, on="gnaf_id", how="left")
    pa = advert_parking(t["car_spaces"], t["park_in"]).fillna(0).values
    t["pred"] = np.where(pa == 1, t["rent_with_car_space"], t["rent_without_car_space"])
    held = [[r.gnaf_id, str(A.at[r.gnaf_id, "address"]).title(), str(A.at[r.gnaf_id, "suburb_name"]), int(A.at[r.gnaf_id, "bed"]), int(p),
             str(r.date.date()), float(r.price), round(float(r.pred), 1), r.tier] for r, p in zip(t.itertuples(), pa)]
    # any flat: submitted prediction + latest advert <= cut-off
    sub = pd.read_csv(os.path.join(out_dir, "rents_Team-Opal.csv")) if os.path.exists(os.path.join(out_dir, "rents_Team-Opal.csv")) else pd.read_csv(os.path.join(out_dir, "rents_team-opal.csv"))
    wf = window(c, "final")
    trf = train_slice(adverts, wf["train_end"])
    ff = fit(flats, trf, wf["train_end"], c["model"])
    from rents.model import combine_rule
    pf = combine_rule(ff, fr)
    pf["tier"] = pf.apply(tier, axis=1)
    last = trf.sort_values("date").groupby("gnaf_id").tail(1).set_index("gnaf_id")
    s = sub.merge(pf[["gnaf_id", "tier"]], on="gnaf_id", how="left").merge(flats[["gnaf_id", "address", "suburb_name", "bed", "park_in", "floor_level"]], on="gnaf_id")
    anyf = []
    for r in s.itertuples():
        la = last.loc[r.gnaf_id] if r.gnaf_id in last.index else None
        anyf.append([r.gnaf_id, str(r.address).title(), r.suburb_name, int(r.bed), None if pd.isna(r.park_in) else int(r.park_in),
                     None if pd.isna(r.floor_level) else int(r.floor_level), float(r.rent_with_car_space), float(r.rent_without_car_space), r.tier,
                     None if la is None else str(la["date"].date()), None if la is None else float(la["price"])])
    data = dict(held=held, any=anyf, cutoffB=str(w["train_end"].date()), cutoff=str(wf["train_end"].date()))
    tpl = open(os.path.join(HERE, "explorer_template.html"), encoding="utf-8").read()
    os.makedirs(os.path.join(out_dir, "demo_local"), exist_ok=True)
    p = os.path.join(out_dir, "demo_local", "flat-rent-explorer.html")
    open(p, "w", encoding="utf-8").write(tpl.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")))
    e = np.abs(t["price"] - t["pred"])
    print(f"wrote {p}: {os.path.getsize(p)/1e6:.1f} MB; held-out {len(held)} adverts (MedAE ${np.median(e):.1f}); any-flat {len(anyf)} flats")


if __name__ == "__main__":
    main()
