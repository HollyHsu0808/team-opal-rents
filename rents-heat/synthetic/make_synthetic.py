"""Synthetic data shaped exactly like the official files, for running the pipeline before the real files arrive.

Writes into rents-heat/data_private/synthetic/:
  rents-input-flats-city-of-sydney.csv        19 official columns + quirks (suburb 'X (NSW)', 0001-01-01 dates,
                                              blank car_space ~9%, floor_level missing ~48%, 4-dp coords,
                                              building_address in mixed formats, ~2% blank)
  api/raw/rent_history_<suburb>.json          API envelope pages (GET /v1/suburbs/{s}/listings/rent-history, to_date=2026-07-31)
  api/listings_rent_history.parquet           the same, flattened by rents.api_fetch.flatten_pages
  rents-past-leases-city-of-sydney.csv        the same adverts in the SPEC's ASSUMED past-leases schema (<= 2026-07-31)
  scoring_adverts_2026-08_09.csv              Aug-Sep 2026 adverts, synthetic evaluation only (never used for training)

Buildings sit at REAL coordinates sampled from public-data/_derived/buildings_features_v2.parquet inside the 33 suburbs,
so the public-feature join is exercised. World parameters are copied from sim/sim_study.py PARAMS (not imported:
that module imports lightgbm at top level).
Run: .venv/Scripts/python.exe synthetic/make_synthetic.py [--n-flats 8000] [--seed 7] [--out DIR]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from rents.api_fetch import flatten_pages  # noqa: E402

# sim/sim_study.py PARAMS (assumptions)
P = dict(base_rent=520.0, sd_suburb=0.10, sd_block=0.07, sd_building=0.10, sd_unit=0.10, sd_advert=0.05,
         floor_per_level=0.006, parking=0.08, bath2=0.06,
         bed_effect={0: -0.37, 1: 0.0, 2: 0.32, 3: 0.63, 4: 0.92, 5: 1.10},
         bed_p=[0.10, 0.38, 0.40, 0.10, 0.019, 0.001], p_parking=0.55, mean_flats_per_building=22,
         p_floor_known=0.52, p_car_space_blank=0.088, p_ever_advertised=0.76, episodes_per_year=0.45,
         p_price_cut=0.25, p_relist=0.10, p_no_price=0.02, p_block_blank=0.065, p_bldaddr_blank=0.02,
         share_non_input_adverts=0.03, market_growth_pa=0.04, sd_suburb_growth=0.01)
ADV_START, ADV_END = pd.Timestamp("2024-01-01"), pd.Timestamp("2026-09-30")
CUTOFF = pd.Timestamp("2026-07-31")
POSTCODE = {"Alexandria": 2015, "Barangaroo": 2000, "Beaconsfield": 2015, "Camperdown": 2050, "Centennial Park": 2021,
            "Chippendale": 2008, "Darlinghurst": 2010, "Darlington": 2008, "Dawes Point": 2000, "Elizabeth Bay": 2011,
            "Erskineville": 2043, "Eveleigh": 2015, "Forest Lodge": 2037, "Glebe": 2037, "Haymarket": 2000,
            "Millers Point": 2000, "Moore Park": 2021, "Potts Point": 2011, "Pyrmont": 2009, "Redfern": 2016,
            "Rosebery": 2018, "Rushcutters Bay": 2011, "Surry Hills": 2010, "Sydney": 2000, "The Rocks": 2000,
            "Ultimo": 2007, "Waterloo": 2017, "Woolloomooloo": 2011, "Zetland": 2017, "Newtown": 2042,
            "Paddington": 2021, "St Peters": 2044, "Annandale": 2038}
EXTRA_SA2 = {"Newtown (NSW)": "Newtown", "Paddington - Moore Park": "Paddington",
             "Sydenham - Tempe - St Peters": "St Peters", "Annandale (NSW)": "Annandale"}
STREET_NAMES = ["HARRIS", "BOURKE", "CROWN", "RILEY", "ELIZABETH", "PITT", "KENT", "MACARTHUR", "BOTANY", "WYNDHAM",
                "BAPTIST", "JOHN", "MARY", "GEORGE", "WILLIAM", "VICTORIA", "OXFORD", "CLEVELAND", "REGENT", "ABERCROMBIE"]
STREET_TYPES = ["STREET", "AVENUE", "ROAD", "PLACE", "LANE", "PARADE", "CRESCENT"]
INPUT_COLS = ["gnaf_id", "address", "building_address", "suburb", "mb", "latitude", "longitude", "property_type",
              "bedrooms", "bathrooms", "car_space", "car_spaces", "floor_level", "flats_in_building",
              "block_new_homes_2021_2026", "block_homes_2026", "ever_advertised_for_lease", "first_ever_listed",
              "attrs_last_seen"]


def load_buildings(derived):
    b = pd.read_parquet(os.path.join(derived, "buildings_features_v2.parquet"),
                        columns=["osm_type", "osm_id", "lon", "lat", "levels", "levels_from_lidar_2020",
                                 "addr_housenumber", "addr_street", "mb_code16", "sa2_name21"])
    s = pd.read_csv(os.path.join(derived, "buildings_official_suburb.csv"))
    b = b.merge(s, on=["osm_type", "osm_id"], how="left")
    b["suburb"] = b["official_suburb"].fillna(b["sa2_name21"].map(EXTRA_SA2))
    b = b[b["suburb"].notna()].reset_index(drop=True)
    b["levels_use"] = b["levels"].fillna(b["levels_from_lidar_2020"]).fillna(3).clip(1, 60).round().astype(int)
    return b


def make(n_flats=8000, seed=7, out=None, derived=None):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    out = out or os.path.join(ROOT, "data_private", "synthetic")
    derived = derived or os.path.normpath(os.path.join(ROOT, "..", "public-data", "_derived"))
    os.makedirs(os.path.join(out, "api", "raw"), exist_ok=True)
    bl = load_buildings(derived)

    # ---- buildings: sizes, real footprints, addresses
    sizes, total = [], 0
    while total < n_flats:
        s = max(1, int(rng.lognormal(np.log(P["mean_flats_per_building"]) - 0.5, 1.0)))
        sizes.append(min(s, 400))
        total += sizes[-1]
    sizes[-1] -= total - n_flats
    sizes = np.array([s for s in sizes if s > 0])
    nb = len(sizes)
    pick = rng.choice(len(bl), nb, replace=nb > len(bl))
    B = bl.iloc[pick].reset_index(drop=True)
    B["size"] = sizes
    sub_names = sorted(POSTCODE)
    sub_eff = dict(zip(sub_names, rng.normal(0, P["sd_suburb"], len(sub_names))))
    sub_growth = dict(zip(sub_names, rng.normal(0, P["sd_suburb_growth"], len(sub_names))))
    blocks = B["mb_code16"].astype(str)
    ublk = blocks.unique()
    blk_eff = dict(zip(ublk, rng.normal(0, P["sd_block"], len(ublk))))
    blk_mb21 = dict(zip(ublk, [f"11{rng.integers(200000000, 299999999)}" for _ in ublk]))
    blk_new = dict(zip(ublk, rng.integers(0, 120, len(ublk))))
    blk_homes = dict(zip(ublk, rng.integers(50, 900, len(ublk))))
    used, num, street = set(), [], []
    for i, r in B.iterrows():
        hn = str(r["addr_housenumber"]) if isinstance(r["addr_housenumber"], str) and r["addr_housenumber"].strip() else None
        st = str(r["addr_street"]).upper() if isinstance(r["addr_street"], str) and r["addr_street"].strip() else None
        if not hn or not st or not hn.isdigit():
            hn = str(rng.integers(1, 400))
            st = f"{rng.choice(STREET_NAMES)} {rng.choice(STREET_TYPES)}"
        while (hn, st, r["suburb"]) in used:
            hn = str(int(hn) + rng.integers(1, 9)) if hn.isdigit() else str(rng.integers(1, 400))
        used.add((hn, st, r["suburb"]))
        num.append(hn)
        street.append(st)
    B["hn"], B["street"] = num, street
    B["pc"] = B["suburb"].map(POSTCODE)
    B["bld_eff"] = rng.normal(0, P["sd_building"], nb)
    B["seq_numbering"] = rng.random(nb) < 0.30
    B["blank_block"] = rng.random(nb) < P["p_block_blank"]

    # ---- flats
    bi = np.repeat(np.arange(nb), B["size"].values)
    n = len(bi)
    F = pd.DataFrame({"b": bi})
    lev = B["levels_use"].values[bi]
    F["floor"] = rng.integers(0, lev)  # 0 = ground
    F = F.sort_values(["b", "floor"]).reset_index(drop=True)
    bi = F["b"].values
    F["pos"] = F.groupby(["b", "floor"]).cumcount() + 1
    F["seq"] = F.groupby("b").cumcount() + 1
    seqn = B["seq_numbering"].values[bi]
    unit = np.where(seqn, F["seq"].astype(str),
                    np.where(F["floor"] == 0, "G" + F["pos"].astype(str).str.zfill(2),
                             F["floor"].astype(str) + F["pos"].astype(str).str.zfill(2)))
    F["unit"] = unit
    F["suburb_name"] = B["suburb"].values[bi]
    hn, st, pc, sb = B["hn"].values[bi], B["street"].values[bi], B["pc"].values[bi], F["suburb_name"].str.upper().values
    F["address"] = [f"UNIT {u}, {h} {s} {x} {p}" for u, h, s, x, p in zip(unit, hn, st, sb, pc)]  # properties style
    F["api_address"] = [f"UNIT {u}, {h} {s}, {x} NSW {p}" for u, h, s, x, p in zip(unit, hn, st, sb, pc)]  # rent-history style
    style = rng.random(n)
    F["building_address"] = np.where(
        style < 0.6, [f"{h} {s} {x} NSW {p}" for h, s, x, p in zip(hn, st, sb, pc)],
        [f"{h} {s.title()}, {x.title()} NSW {p}" for h, s, x, p in zip(hn, st, sb, pc)])
    F.loc[rng.random(n) < P["p_bldaddr_blank"], "building_address"] = ""
    F["gnaf_id"] = ["GANSW7" + str(x) for x in (10_000_000 + rng.permutation(89_000_000)[:n])]
    jit = rng.normal(0, 4.0 / 111_000, (n, 2))
    F["latitude"] = np.round(B["lat"].values[bi] + jit[:, 0], 4)
    F["longitude"] = np.round(B["lon"].values[bi] + jit[:, 1] * 1.2, 4)
    bed = rng.choice(len(P["bed_p"]), n, p=P["bed_p"])
    F["bedrooms"] = bed
    F["bathrooms"] = np.where((bed >= 2) & (rng.random(n) < 0.45), 2, 1) + (bed >= 4)
    park = (rng.random(n) < P["p_parking"]).astype(int)
    F["park_true"] = park
    cs = np.where(park == 1, np.where((bed >= 3) & (rng.random(n) < 0.3), 2, 1), 0).astype(float)
    blank = rng.random(n) < P["p_car_space_blank"]
    F["car_space"] = np.where(blank, np.nan, park.astype(float))
    F["car_spaces"] = np.where(blank & (rng.random(n) < 0.5), np.nan, cs)
    F["floor_level"] = np.where(rng.random(n) < P["p_floor_known"], F["floor"], np.nan)
    F["flats_in_building"] = B["size"].values[bi]
    blk = blocks.values[bi]
    bb = B["blank_block"].values[bi]
    F["mb"] = [blk_mb21[x] for x in blk]
    F["block_new_homes_2021_2026"] = np.where(bb, np.nan, [blk_new[x] for x in blk])
    F["block_homes_2026"] = np.where(bb, np.nan, [blk_homes[x] for x in blk])
    F["property_type"] = np.where(rng.random(n) < 0.9, "Unit", "Apartment")
    F["suburb"] = F["suburb_name"] + " (NSW)"
    # optional public-feature effects (assumed sizes, log points) from the REAL building/floor of each flat
    eff_name = os.environ.get("SYN_PUBLIC_EFFECTS", "none")
    EFF = {"none": None,
           "realistic": dict(water=0.10, clear=0.03, park=0.02, air=-0.04),
           "strong": dict(water=0.20, clear=0.06, park=0.04, air=-0.08)}[eff_name]
    pub = np.zeros(n)
    if EFF:
        fv = pd.read_parquet(os.path.join(derived, "lidar_floor_view.parquet"),
                             columns=["osm_type", "osm_id", "floor", "water_view_frac", "park_view_frac", "roofline_clear_frac"])
        top = fv.groupby(["osm_type", "osm_id"])["floor"].max().rename("top").reset_index()
        air = pd.read_parquet(os.path.join(derived, "aircraft_proxy.parquet"), columns=["osm_type", "osm_id", "under_16_34_path_1km"])
        q = pd.DataFrame({"osm_type": B["osm_type"].values[bi], "osm_id": B["osm_id"].values[bi], "fl": F["floor"].values})
        q = q.merge(top, on=["osm_type", "osm_id"], how="left")
        q["floor"] = np.minimum(q["fl"], q["top"].fillna(0)).astype(int)
        q = q.merge(fv, on=["osm_type", "osm_id", "floor"], how="left").merge(air, on=["osm_type", "osm_id"], how="left")
        pub = (EFF["water"] * q["water_view_frac"].fillna(0).values + EFF["clear"] * q["roofline_clear_frac"].fillna(0).values
               + EFF["park"] * q["park_view_frac"].fillna(0).values
               + EFF["air"] * q["under_16_34_path_1km"].fillna(False).astype(float).values)
    F["public_effect_true"] = pub
    bed_e = np.array([P["bed_effect"][b] for b in bed])
    F["level"] = (np.log(P["base_rent"]) + F["suburb_name"].map(sub_eff).values + np.array([blk_eff[x] for x in blk])
                  + B["bld_eff"].values[bi] + rng.normal(0, P["sd_unit"], n) + bed_e
                  + P["bath2"] * (F["bathrooms"].values >= 2) + P["floor_per_level"] * F["floor"].values
                  + P["parking"] * park + pub)

    # ---- adverts: lease episodes 2024-01..2026-09 with price cuts and relists
    months = pd.period_range(ADV_START, ADV_END, freq="M")
    mkt = P["market_growth_pa"] * np.arange(len(months)) / 12 + rng.normal(0, 0.005, len(months))
    ever = rng.random(n) < P["p_ever_advertised"]
    span_years = (ADV_END - ADV_START).days / 365.25
    k = np.where(ever, rng.poisson(P["episodes_per_year"] * span_years, n), 0)
    fi = np.repeat(np.arange(n), k)
    days = rng.integers(0, (ADV_END - ADV_START).days + 1, fi.size)
    d0 = ADV_START + pd.to_timedelta(days, unit="D")
    mi = (d0.year - ADV_START.year) * 12 + d0.month - 1
    growth = F["suburb_name"].map(sub_growth).values[fi] * mi / 12
    p0 = np.exp(F["level"].values[fi] + mkt[mi] + growth + rng.normal(0, P["sd_advert"], fi.size))
    p0 = np.round(p0 / 5) * 5
    ep = pd.DataFrame({"f": fi, "date": d0, "price": p0, "ep": np.arange(fi.size)})
    cut = rng.random(fi.size) < P["p_price_cut"]
    e2 = ep[cut].copy()
    e2["date"] = e2["date"] + pd.to_timedelta(rng.integers(7, 29, cut.sum()), unit="D")
    e2["price"] = np.round(e2["price"] * rng.uniform(0.94, 0.98, cut.sum()) / 5) * 5
    rel = rng.random(fi.size) < P["p_relist"]
    e3 = ep[rel].copy()
    e3["date"] = e3["date"] + pd.to_timedelta(rng.integers(30, 61, rel.sum()), unit="D")
    A = pd.concat([ep, e2, e3], ignore_index=True)
    A = A[A["date"] <= ADV_END].sort_values(["f", "date"]).reset_index(drop=True)
    last_in_ep = ~A.duplicated("ep", keep="last")
    A["leased"] = last_in_ep & (A["date"] < ADV_END - pd.Timedelta(days=10))
    na = len(A)
    fa = A["f"].values
    A["gnaf_pid"] = F["gnaf_id"].values[fa]
    A["address"] = F["api_address"].values[fa]
    A["building_address"] = F["building_address"].values[fa]
    A["bedrooms"] = F["bedrooms"].values[fa]
    A["bathrooms"] = F["bathrooms"].values[fa]
    A["car_spaces"] = np.where(rng.random(na) < 0.05, np.nan, cs[fa])
    A["lat"], A["lng"] = F["latitude"].values[fa], F["longitude"].values[fa]
    A["property_type"] = F["property_type"].values[fa]
    A["suburb_name"] = F["suburb_name"].values[fa]
    A["weekly_rent"] = np.where(rng.random(na) < P["p_no_price"], np.nan, A["price"])
    # adverts for addresses that are not input flats (houses / terraces), as in the API
    nx = int(P["share_non_input_adverts"] * na)
    X = A.sample(nx, random_state=seed).copy()
    X["gnaf_pid"] = ["GANSW9" + str(x) for x in (10_000_000 + rng.permutation(89_000_000)[:nx])]
    X["address"] = [f"{rng.integers(1, 300)} {rng.choice(STREET_NAMES)} STREET, {s.upper()} NSW {POSTCODE[s]}" for s in X["suburb_name"]]
    X["building_address"] = ""
    X["property_type"] = "House"
    X["weekly_rent"] = np.round(X["weekly_rent"] * 1.4 / 5) * 5
    A = pd.concat([A, X], ignore_index=True)
    A["listing_date"] = A["date"].dt.strftime("%Y-%m-%d")

    train = A[A["date"] <= CUTOFF]
    score = A[(A["date"] > CUTOFF) & A["gnaf_pid"].isin(F["gnaf_id"])]

    # ---- input flags derived from the full advert history (attrs_last_seen may be after the cut-off: never a feature)
    g = A[A["gnaf_pid"].str.startswith("GANSW7")].groupby("gnaf_pid")["date"]
    first, last = g.min(), g.max()
    F["ever_advertised_for_lease"] = (F["gnaf_id"].isin(first.index) | (ever & (rng.random(n) < 0.4))).astype(int)
    F["first_ever_listed"] = F["gnaf_id"].map(first).dt.strftime("%Y-%m-%d").fillna("0001-01-01")
    F["attrs_last_seen"] = F["gnaf_id"].map(last).dt.strftime("%Y-%m-%d").fillna("0001-01-01")
    inp = F[INPUT_COLS].copy()
    for c in ["car_space", "car_spaces", "floor_level", "block_new_homes_2021_2026", "block_homes_2026"]:
        inp[c] = inp[c].astype("Int64")
    inp = inp.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    inp.to_csv(os.path.join(out, "rents-input-flats-city-of-sydney.csv"), index=False)

    # ---- API-shaped pages (server-side to_date=2026-07-31), flattened like api_fetch does
    api_cols = ["address", "bathrooms", "bedrooms", "car_spaces", "gnaf_pid", "lat", "leased", "listing_date", "lng",
                "property_type", "weekly_rent"]
    pages_all = []
    for s, grp in train.groupby("suburb_name"):
        recs = json.loads(grp[api_cols].to_json(orient="records"))
        for r in recs:
            if r.get("weekly_rent") is None:
                r.pop("weekly_rent", None)  # some API records lack a price
        pages = []
        for off in range(0, len(recs), 1000):
            chunk = recs[off:off + 1000]
            pages.append({"data": {"area_level": "suburb", "area_name": f"{s} (NSW)",
                                   "meta": {"limit": 1000, "offset": off, "returned": len(chunk), "total": len(recs)},
                                   "results": chunk}})
        with open(os.path.join(out, "api", "raw", f"rent_history_{s.replace(' ', '_')}.json"), "w", encoding="utf-8") as f:
            json.dump(pages, f)
        pages_all.append((f"{s} (NSW)", pages))
    flat = pd.concat([flatten_pages(p, s) for s, p in pages_all], ignore_index=True)
    flat.to_parquet(os.path.join(out, "api", "listings_rent_history.parquet"), index=False)

    # ---- assumed past-leases schema
    pl = pd.DataFrame({"gnaf_id": train["gnaf_pid"], "building_address": train["building_address"],
                       "date": train["listing_date"], "price": train["weekly_rent"], "bedrooms": train["bedrooms"],
                       "bathrooms": train["bathrooms"], "car_spaces": train["car_spaces"].astype("Int64")})
    pl.to_csv(os.path.join(out, "rents-past-leases-city-of-sydney.csv"), index=False)
    sc = pd.DataFrame({"gnaf_id": score["gnaf_pid"], "date": score["listing_date"], "price": score["weekly_rent"],
                       "car_spaces": score["car_spaces"].astype("Int64"), "leased": score["leased"], "address": score["address"]})
    sc.to_csv(os.path.join(out, "scoring_adverts_2026-08_09.csv"), index=False)
    info = dict(n_flats=int(n), n_buildings=int(nb), n_suburbs=int(F["suburb_name"].nunique()),
                n_adverts_train=int(len(train)), n_adverts_scoring=int(len(sc)),
                share_flats_with_train_advert=float(F["gnaf_id"].isin(train["gnaf_pid"]).mean()),
                runtime_s=round(time.time() - t0, 1), seed=seed, out=out)
    with open(os.path.join(out, "synthetic_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-flats", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    print(json.dumps(make(a.n_flats, a.seed, a.out), indent=1))
