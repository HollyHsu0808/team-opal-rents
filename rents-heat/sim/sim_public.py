"""Simulation: what can each PUBLIC dataset add to rent prediction, on real City-of-Sydney geography?

Synthetic flats are placed in the REAL OSM residential buildings of public-data/_derived/buildings_features.parquet
(footprint, levels, neighbour heights, strata plan number/date, station/water distance, LST, canopy, UHI, ABS SA2 rent).
Suburb = real ABS SA2 (2021); block = real ABS mesh block (2016, from the UHI layer).
EVERYTHING ABOUT RENTS IS ASSUMED (see PARAMS / EFFECTS and their labels). Each public effect is run at 0, at a
"plausible" level and at a "strong" level so the reader sees detectability, not just what was planted.

World (log weekly rent):
  base + suburb(+ABS-aligned part) + suburb growth + block + building + wing + STACK (same suffix in same wing: shared
  bedrooms/bath/layout, sd 0.06) + unit + bedrooms (bond-calibrated) + bath + parking + floor
  + VIEW  b_view  * min(15, max(0, floor - neighbour_max_levels_100m))
  + AGE   b_age   * (build_year - 2000)/10            (newer premium per decade)
  + STATION b_stn * dist_station_km
  + WATER b_wat   * min(dist_water_km, 2)
  + HEAT  b_heat  * z(LST) ; CANOPY b_can * z(canopy %)
  + market index + advert noise ; price rounded to $10 (70%) or $5.
Microburbs-like "block rent map" = block mean of (block + building + wing + public effects) + noise (0.03 or 0.10).
Unit numbers are generated per building (floor+suffix / letter / sequential schemes; multi-wing buildings duplicate
numbers) and parsed back with src/stack/parse_unit.py, so the modeller sees only parsed floor/suffix (with errors).

Windows: A train <= 2026-04, test May (selection); B train <= 2026-05, test Jun-Jul (evaluate once);
         S ("scored-like") train <= 2026-07, test 1,699 adverts Aug-Sep. Test sets stratified to 70% own-history.
Methods (cumulative rungs): M0 ladder; M1 own history x repeat index; M2 same-building same-bed comps (LOO);
  M3 validated stack comps; G_base LightGBM (Microburbs-side features) for no-history rows; G_<group> = G_base + one
  public group; G_all; + stale-rent gate; + MedAE shift calibration (chosen on A); + snapping $5 / $10.
Run: python sim/sim_public.py [--seeds 3] [--quick]
"""
import argparse
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src", "stack"))
from parse_unit import parse_unit  # noqa: E402
from sim_study import MONTHS, M_IDX, market_index  # noqa: E402  (helpers only; sim_study is not modified)

PUB = os.path.join(HERE, "..", "..", "public-data")
FEATS = os.path.join(PUB, "_derived", "buildings_features.parquet")
SP_LOOKUP = os.path.join(PUB, "_derived", "sp_year_lookup.csv")

PARAMS = dict(
    n_flats=65_777, base_rent=None,  # base calibrated so M0 (suburb x bed x parking ladder) MedAE ~ $100
    sd_suburb=0.10, sd_block=0.07, sd_building=0.08, sd_wing=0.04, sd_stack=0.06, sd_unit=0.08, sd_advert=0.05,
    suburb_growth_sd=0.01, floor_per_level=0.006, parking=0.08, bath2=0.06,
    bed_effect={0: -0.37, 1: 0.0, 2: 0.32, 3: 0.63, 4: 0.92},  # NSW bond data, CoS flats (R4)
    bed_p=[0.10, 0.38, 0.40, 0.10, 0.02], p_parking=0.55, p_ever=0.76, adverts_per_year=0.30,
    gross_m2_per_flat=95.0, max_flats_per_building=400,
    scheme_p={"floorsuffix": 0.45, "letter": 0.07, "sequential": 0.48},
    p_linkable=0.80,  # ASSUMPTION: share of flats whose adverts carry a parseable unit number (own history linkable)
    stack_shrink_k=2.5,  # M3 blend weight n_s/(n_s+k); k = (sd_unit^2+sd_advert^2)/sd_stack^2 from the ASSUMED sds
    p_hist_test=0.70, n_scored=1699,
)
# Effect sizes. Labels: all are ASSUMPTIONS unless a source is given; "plausible" is a judgement call informed by the
# R4 literature notes (green/heat rent evidence thin, renters value greenery ~1/5 of owners), not an estimate.
EFFECTS = {
    "age":     dict(plausible=0.015, strong=0.05, unit="log rent per decade newer (assumption)"),
    "view":    dict(plausible=0.005, strong=0.015, unit="log rent per level above neighbours' max, capped 15 (assumption)"),
    "station": dict(plausible=-0.015, strong=-0.05, unit="log rent per km to nearest station (assumption)"),
    "water":   dict(plausible=-0.02, strong=-0.06, unit="log rent per km to harbour/water, capped 2 km (assumption; "
                                                         "RBA-style hedonics put water VIEWS much higher, distance lower)"),
    "heat":    dict(plausible=-0.005, strong=-0.03, unit="log rent per SD of summer LST (assumption; R4: thin evidence)"),
    "canopy":  dict(plausible=0.005, strong=0.03, unit="log rent per SD of MMB canopy % (assumption; R4)"),
    "abs":     dict(plausible=0.5, strong=0.9, unit="corr of suburb effect with ABS 2021 SA2 log median rent (assumption)"),
}
SCENARIOS = ["zero", "plausible", "strong"]
GROUPS = {
    "age": ["sp_plannumber", "sp_year_obs"],
    "view": ["levels_obs", "nbr_max_obs", "view_guess"],
    "station": ["dist_station_m", "dist_rail_metro_m"],
    "water": ["dist_water_m"],
    "heatcanopy": ["lst_c", "uhi16_mmb", "canopy_mmb_pct"],
    "abs": ["abs_sa2_median_rent_2021"],
}
BASE_COLS = ["sub", "bed", "bath", "park", "floor_obs", "fib", "block_map", "comp", "stack_comp"]
WINDOWS = {"A": ("2026-04", ["2026-05"]), "B": ("2026-05", ["2026-06", "2026-07"]), "S": ("2026-07", ["2026-08", "2026-09"])}
LAST_M = len(MONTHS) - 1
N_BOOT = 1000
_PARSE_CACHE = {}


# ----------------------------------------------------------------------------------------------- static buildings
def load_buildings():
    B = pd.read_parquet(FEATS).reset_index(drop=True)
    lv = B.levels.copy()
    B["levels_obs"] = lv
    B["levels_true"] = np.clip(lv.fillna(4), 1, 80).astype(int)
    B["nbr_max_obs"] = B.neighbour_max_levels_100m
    # suburb = SA2; fill missing by nearest building with an SA2
    xy = np.c_[B.lon * 92_000, B.lat * 111_000]
    has = B.sa2_code21.notna().values
    _, j = cKDTree(xy[has]).query(xy[~has])
    B.loc[~has, "sa2_code21"] = B.sa2_code21.values[has][j]
    B.loc[~has, "abs_sa2_median_rent_2021"] = B.abs_sa2_median_rent_2021.values[has][j]
    B["sub"] = B.sa2_code21.astype("int64").astype("category").cat.codes
    blk = B.mb_code16.astype(object).fillna(pd.Series("bld" + B.index.astype(str), index=B.index)).astype(str)
    B["blk"] = blk.astype("category").cat.codes
    # flats per building from footprint x levels, capped, scaled to the CoS total
    raw = B.footprint_m2 * B.levels_true / PARAMS["gross_m2_per_flat"]
    n = raw.copy()
    for _ in range(20):
        n = np.clip(n * PARAMS["n_flats"] / n.sum(), 2, PARAMS["max_flats_per_building"])
    n = np.round(n).astype(int)
    diff = PARAMS["n_flats"] - n.sum()
    order = np.argsort(-n.values)
    k = 0
    while diff != 0:
        i = order[k % 50]
        step = 1 if diff > 0 else -1
        if 2 <= n.iat[i] + step <= PARAMS["max_flats_per_building"]:
            n.iat[i] += step
            diff -= step
        k += 1
    B["n_flats"] = n.values
    # strata plan -> observed year (registration-like only for SP >= 45,000; see sp_year_calibration.json)
    lk = pd.read_csv(SP_LOOKUP)
    pn = B.sp_plannumber
    reg = pn >= 45_000
    B["sp_year_obs"] = np.where(reg, np.interp(pn.fillna(0), lk.sp_bin_mid, lk.year_lower_envelope), np.nan)
    B["dist_station_km"] = B.dist_station_m / 1000
    B["dist_water_km"] = np.minimum(B.dist_water_m / 1000, 2.0)
    z = lambda s: (s - s.mean()) / s.std()  # noqa: E731
    B["z_lst"] = z(B.lst_c).fillna(0)
    B["z_can"] = z(B.canopy_mmb_pct).fillna(0)
    B["z_abs"] = z(np.log(B.abs_sa2_median_rent_2021)).fillna(0)
    B["n_wings"] = np.where((B.n_sp_plans >= 2) & (B.n_flats >= 8), 2, 1)
    return B


def cached_parse(unit, max_floor):
    key = (unit, max_floor)
    if key not in _PARSE_CACHE:
        _PARSE_CACHE[key] = parse_unit(unit, max_floor)
    return _PARSE_CACHE[key]


# ----------------------------------------------------------------------------------------------- world
def make_world(Bs, rng, scen, block_noise, base_rent):
    P = PARAMS
    B = Bs.copy()
    eff = {g: (0.0 if scen == "zero" else EFFECTS[g][scen]) for g in EFFECTS}
    nb = len(B)
    # true build year
    yr = B.sp_year_obs.values + rng.uniform(0, 0.5, nb)
    pre = B.sp_plannumber.notna().values & np.isnan(B.sp_year_obs.values)
    yr[pre] = rng.uniform(1965, 1994, pre.sum())
    nosp = B.sp_plannumber.isna().values
    pool = B.sp_year_obs.dropna().values
    yr[nosp] = np.where(rng.random(nosp.sum()) < 0.5, rng.choice(pool, nosp.sum()), rng.uniform(1930, 1994, nosp.sum()))
    B["year_true"] = np.minimum(yr, 2026.6)
    # suburb effects (ABS-aligned part)
    S = B["sub"].max() + 1
    zs = B.groupby("sub").z_abs.mean().reindex(range(S)).fillna(0).values
    rho = eff["abs"]
    sub_eff = P["sd_suburb"] * (rho * zs / (zs.std() + 1e-9) + np.sqrt(1 - rho ** 2) * rng.normal(0, 1, S))
    sub_growth = rng.normal(0, P["suburb_growth_sd"], S)
    NB = B.blk.max() + 1
    blk_eff = rng.normal(0, P["sd_block"], NB)
    bld_eff = rng.normal(0, P["sd_building"], nb)
    scheme = rng.choice(list(P["scheme_p"]), nb, p=list(P["scheme_p"].values()))
    scheme = np.where(B.n_wings.values == 2, "floorsuffix", scheme)
    g_flat_rows = []
    for b in range(nb):
        L = int(B.levels_true.iat[b])
        n = int(B.n_flats.iat[b])
        W = int(B.n_wings.iat[b])
        floors = list(range(1, L)) if L >= 3 else list(range(0, L))
        sizes = [n // W + (1 if w < n % W else 0) for w in range(W)]
        seq = 0
        sch = scheme[b]
        k_all = int(np.ceil(max(sizes) / len(floors)))
        if k_all > 99 and sch != "sequential":
            sch = "sequential"
            scheme[b] = sch
        g0 = rng.random() < 0.5  # ground floor written as G01 (else plain 1)
        for w, nw in enumerate(sizes):
            k = int(np.ceil(nw / len(floors)))
            for i in range(nw):
                f = floors[min(i // k, len(floors) - 1)]
                s = i % k + 1
                seq += 1
                if sch == "sequential":
                    u = str(seq)
                elif sch == "letter":
                    u = (f"G{(s + 1) // 2:02d}" if f == 0 else f"{f}{(s + 1) // 2:02d}") + "AB"[(s + 1) % 2]
                else:
                    u = (f"G{s:02d}" if g0 else str(s)) if f == 0 else f"{f}{s:02d}"
                g_flat_rows.append((b, w, f, s, u))
    F = pd.DataFrame(g_flat_rows, columns=["bld", "wing", "floor", "suf", "unit"])
    n = len(F)
    F["flat"] = np.arange(n)
    # parse unit numbers (what the modeller sees)
    lv_obs = B.levels_obs.values
    parsed = [cached_parse(u, None if np.isnan(lv_obs[b]) else int(lv_obs[b])) for u, b in zip(F.unit.values, F.bld.values)]
    conf = np.array([p.confidence for p in parsed])
    okp = np.isin(conf, ["high", "medium"])
    link = rng.random(n) < P["p_linkable"]
    F["linkable"] = link
    okp = okp & link
    F["floor_obs"] = np.where(okp, [p.floor_guess if p.floor_guess is not None else np.nan for p in parsed], np.nan).astype(float)
    F["suffix_obs"] = np.where(okp, [p.suffix or "" for p in parsed], "")
    F["scheme"] = scheme[F.bld.values]
    # stacks: same bld, wing, suffix -> shared bedrooms, bath, layout effect
    st_key = F.bld.astype(np.int64) * 1000 + F.wing * 200 + F.suf
    st_codes, st_idx = np.unique(st_key.values, return_inverse=True)
    ns = len(st_codes)
    st_bed = rng.choice(5, ns, p=P["bed_p"])
    st_bath = np.where((st_bed >= 2) & (rng.random(ns) < 0.4), 2, 1)
    st_eff = rng.normal(0, P["sd_stack"], ns)
    F["stack"] = st_idx
    F["bed"] = st_bed[st_idx]
    F["bath"] = st_bath[st_idx]
    F["park"] = (rng.random(n) < P["p_parking"]).astype(int)
    wing_eff = rng.normal(0, P["sd_wing"], (nb, 2))
    bi = F.bld.values
    for c in ["sub", "blk", "sp_plannumber", "sp_year_obs", "levels_obs", "nbr_max_obs", "dist_station_m",
              "dist_rail_metro_m", "dist_water_m", "lst_c", "uhi16_mmb", "canopy_mmb_pct", "abs_sa2_median_rent_2021",
              "n_flats", "levels_true"]:
        F[c] = B[c].values[bi]
    F = F.rename(columns={"n_flats": "fib"})
    F["view_guess"] = np.clip(F.floor_obs - F.nbr_max_obs, 0, None)
    nbr_true = B.nbr_max_obs.fillna(B.levels_true.astype(float)).values[bi]
    view_true = np.minimum(np.maximum(0, F.floor.values - nbr_true), 15)
    pub = dict(
        age=eff["age"] * (B.year_true.values[bi] - 2000) / 10,
        view=eff["view"] * view_true,
        station=eff["station"] * B.dist_station_km.values[bi],
        water=eff["water"] * B.dist_water_km.values[bi],
        heat=eff["heat"] * B.z_lst.values[bi],
        canopy=eff["canopy"] * B.z_can.values[bi],
    )
    pub_tot = sum(pub.values())
    bed_e = np.array([P["bed_effect"][k] for k in range(5)])[F.bed.values]
    unit_eff = rng.normal(0, P["sd_unit"], n)
    loc_part = blk_eff[F.blk.values] + bld_eff[bi] + wing_eff[bi, F.wing.values] + pub_tot
    F["level"] = (np.log(base_rent) + sub_eff[F["sub"].values] + loc_part + st_eff[st_idx] + unit_eff + bed_e
                  + P["bath2"] * (F.bath.values == 2) + P["floor_per_level"] * F.floor.values + P["parking"] * F.park.values)
    bm = pd.Series(loc_part).groupby(F.blk.values).mean()
    bm = bm + rng.normal(0, block_noise, len(bm))
    F["block_map"] = bm.reindex(F.blk.values).values
    F["sub_growth"] = sub_growth[F["sub"].values]
    # existence: flats exist from their build month (if after 2016-01)
    start = np.clip(np.floor((B.year_true.values[bi] - 2016) * 12), 0, LAST_M).astype(int)
    mkt = market_index(len(MONTHS), rng)
    ever = rng.random(n) < P["p_ever"]
    months_avail = LAST_M - start + 1
    k = np.where(ever, rng.poisson(P["adverts_per_year"] * months_avail / 12.0), 0)
    a_flat = np.repeat(np.arange(n), k)
    a_m = start[a_flat] + (rng.random(a_flat.size) * months_avail[a_flat]).astype(int)
    ads = pd.DataFrame(dict(flat=a_flat, m=a_m))
    lp = (F.level.values[a_flat] + mkt[a_m] + F.sub_growth.values[a_flat] * a_m / 12.0
          + rng.normal(0, P["sd_advert"], a_flat.size))
    step = np.where(rng.random(a_flat.size) < 0.7, 10, 5)
    ads["price"] = np.round(np.exp(lp) / step) * step
    ads = ads.sort_values(["m", "flat"]).reset_index(drop=True)
    info = dict(n_flats=n, n_ads=len(ads), n_stacks=ns,
                floor_obs_share=float(np.mean(~np.isnan(F.floor_obs.values))),
                scheme_share_flats=F.scheme.value_counts(normalize=True).round(3).to_dict(),
                multiwing_flats=float((F.wing == 1).mean() * 2), median_price=float(ads.price.median()),
                linkable_share=float(F.linkable.mean()))
    return F, ads, info


# ----------------------------------------------------------------------------------------------- methods
def repeat_index(ads, T):
    a = ads[ads.m <= T].sort_values(["flat", "m"])
    prev = a.groupby("flat")[["m", "price"]].shift(1)
    pr = a.assign(m0=prev.m, p0=prev.price).dropna()
    pr = pr[pr.m0 < pr.m]
    rows = np.arange(len(pr))
    X = coo_matrix((np.r_[np.ones(len(pr)), -np.ones(len(pr))],
                    (np.r_[rows, rows], np.r_[pr.m.values.astype(int), pr.m0.values.astype(int)])),
                   shape=(len(pr), T + 1)).tocsr()
    y = np.log(pr.price.values) - np.log(pr.p0.values)
    beta = lsqr(X[:, 1:], y, damp=1e-3)[0]
    idx = np.r_[0.0, beta]
    return np.r_[idx, np.full(len(MONTHS) - T - 1, idx[-1])]


def ladder(ads_tr, F, keys_df, T):
    recent = ads_tr[ads_tr.m > T - 12].merge(F[["flat", "sub", "bed", "park"]], on="flat")
    out = np.full(len(keys_df), np.nan)
    for keys in (["sub", "bed", "park"], ["sub", "bed"], ["bed"]):
        g = recent.groupby(keys).price.agg(["median", "size"])
        g = g[g["size"] >= 10]["median"]
        vals = keys_df[keys].merge(g.rename("v").reset_index(), on=keys, how="left").v.values
        out = np.where(np.isnan(out), vals, out)
    return np.where(np.isnan(out), recent.price.median(), out)


def loo_mean(df, a, keys, return_n=False):
    """Leave-own-flat-out mean of a.adj_log within groups `keys` (NaN if no other flat). Own adverts are only
    removable when the flat is linkable (unit number known); otherwise they stay in the comps, as in real data."""
    g_all = a.groupby(keys).adj_log.agg(["sum", "size"]).rename(columns={"sum": "S", "size": "N"}).reset_index()
    g_flat = a.groupby("flat").adj_log.agg(["sum", "size"]).rename(columns={"sum": "Sf", "size": "Nf"}).reset_index()
    x = df[["flat"] + keys].merge(g_all, on=keys, how="left").merge(g_flat, on="flat", how="left")
    x = x.fillna({"Sf": 0, "Nf": 0, "S": 0, "N": 0})
    if "linkable" in df:
        unl = ~df.linkable.values
        x.loc[unl, ["Sf", "Nf"]] = 0
    # own flat's adverts only count if they fall in the same group
    no = x.N.values - x.Nf.values
    no = np.where(no > 0, no, np.nan)
    if return_n:
        return (x.S - x.Sf).values / no, np.nan_to_num(no)
    return ((x.S - x.Sf).values / no)


def validate_stacks(ads, F, T):
    """Buildings where within-stack spread < within-building spread (per-flat mean residuals, last 36 months <= T)."""
    a = ads[(ads.m <= T) & (ads.m > T - 36)].merge(F[["flat", "bld", "wing", "bed", "suffix_obs"]], on="flat")
    idx = repeat_index(ads, T)
    a["r"] = np.log(a.price) - idx[a.m.values]
    fl = a.groupby(["flat", "bld", "wing", "bed", "suffix_obs"], as_index=False).r.mean()
    fl = fl[fl.suffix_obs != ""]
    fl["r"] -= fl.groupby(["bld", "bed"]).r.transform("mean")
    fl["n_bb"] = fl.groupby(["bld", "bed"]).r.transform("size")
    fl = fl[fl.n_bb >= 3]
    bsd = fl.groupby("bld").r.apply(lambda s: np.sqrt(np.mean(s ** 2)))
    fl["rs"] = fl.r - fl.groupby(["bld", "wing", "suffix_obs", "bed"]).r.transform("mean")
    fl["ns"] = fl.groupby(["bld", "wing", "suffix_obs", "bed"]).r.transform("size")
    st = fl[fl.ns >= 2]
    n_st = st.groupby("bld").apply(lambda d: d.groupby(["wing", "suffix_obs", "bed"]).ngroups, include_groups=False)
    ssd = st.groupby("bld").apply(lambda d: np.sqrt(np.sum(d.rs ** 2) / max(1, len(d) - d.groupby(
        ["wing", "suffix_obs", "bed"]).ngroups)), include_groups=False)
    ok = (ssd < 0.9 * bsd.reindex(ssd.index)) & (n_st.reindex(ssd.index) >= 2)
    return set(ok[ok].index)


def make_test(ads, F, T, months, n_target, rng):
    te = ads[ads.m.isin(months)].drop_duplicates("flat")
    hist_flats = set(ads.loc[ads.m <= T, "flat"].unique()) & set(F.flat[F.linkable].values)
    te = te.assign(has_hist=te.flat.isin(hist_flats).values)
    h, nh = te[te.has_hist], te[~te.has_hist]
    p = PARAMS["p_hist_test"]
    if n_target is None:
        n_h = min(len(h), int(round(len(nh) * p / (1 - p))))
        te = pd.concat([h.sample(n_h, random_state=int(rng.integers(1e9))), nh])
    else:
        n_nh = min(len(nh), int(round(n_target * (1 - p))))
        te = pd.concat([h.sample(n_target - n_nh, random_state=int(rng.integers(1e9))),
                        nh.sample(n_nh, random_state=int(rng.integers(1e9)))])
    return te.reset_index(drop=True)


def run_window(F, ads, wname, T, months, n_target, rng, validated, seed, shared):
    te = make_test(ads, F, T, months, n_target, rng)
    tf = te.merge(F, on="flat", how="left")
    ads_tr = ads[ads.m <= T]
    idx = repeat_index(ads_tr, T)
    P = {}
    cov = {}
    nh = ~tf.has_hist.values
    m0 = ladder(ads_tr, F, tf, T)
    P["M0"] = m0
    last = ads_tr[ads_tr.flat.isin(F.flat[F.linkable].values)].groupby("flat").tail(1).set_index("flat")
    own = tf.flat.map(last.price).values
    own_m = tf.flat.map(last.m).values
    m1 = np.where(~np.isnan(own), own * np.exp(idx[tf.m.values] - idx[np.nan_to_num(own_m).astype(int)]), m0)
    P["M1"] = m1
    cov["M1"] = ~np.isnan(own)
    a = ads_tr[ads_tr.m > T - 24].merge(F[["flat", "bld", "wing", "bed", "park", "suffix_obs"]], on="flat")
    a["adj_log"] = np.log(a.price) - idx[a.m.values]
    pk = a.groupby("park").adj_log.median()
    park_gap = pk.get(1, 0) - pk.get(0, 0)
    a["adj_log"] -= park_gap * a.park
    a_st = a[a.suffix_obs != ""]
    tadj = idx[tf.m.values] + park_gap * tf.park.values
    comp = loo_mean(tf, a, ["bld", "bed"])
    comp_pred = np.exp(comp + tadj)
    te["no_comp"] = np.isnan(comp)
    P["M2"] = np.where(nh & ~np.isnan(comp), comp_pred, m1)
    cov["M2"] = nh & ~np.isnan(comp)
    stk, n_s = loo_mean(tf, a_st, ["bld", "wing", "suffix_obs", "bed"], return_n=True)
    stk = np.where(tf.suffix_obs.values != "", stk, np.nan)
    stk_nw, n_snw = loo_mean(tf, a_st, ["bld", "suffix_obs", "bed"], return_n=True)
    stk_nw = np.where(tf.suffix_obs.values != "", stk_nw, np.nan)
    val = tf.bld.isin(validated).values
    k = PARAMS["stack_shrink_k"]

    def blend(sv, ns):
        w = ns / (ns + k)
        return np.where(np.isnan(comp), sv, w * sv + (1 - w) * np.nan_to_num(comp))

    use3 = nh & val & ~np.isnan(stk)
    P["M3"] = np.where(use3, np.exp(blend(stk, n_s) + tadj), P["M2"])
    cov["M3"] = use3
    use3n = nh & val & ~np.isnan(stk_nw)
    P["M3_nowing"] = np.where(use3n, np.exp(blend(stk_nw, n_snw) + tadj), P["M2"])
    cov["M3_nowing"] = use3n
    # LightGBM for no-history rows
    tr = ads_tr[ads_tr.m > T - 12].merge(F, on="flat")
    tr_comp = loo_mean(tr, a, ["bld", "bed"])
    tr_stk = np.where(tr.suffix_obs.values != "", loo_mean(tr, a_st, ["bld", "wing", "suffix_obs", "bed"]), np.nan)
    y_tr = np.log(tr.price.values) - idx[tr.m.values]
    Xtr0 = tr[BASE_COLS[:-2]].assign(comp=tr_comp, stack_comp=tr_stk)
    Xte0 = tf[BASE_COLS[:-2]].assign(comp=comp, stack_comp=stk)
    variants = [("G_base", [])] + [(f"G_{g}", c) for g, c in GROUPS.items()] + [("G_all", sum(GROUPS.values(), []))]
    # positive control: same ablations WITHOUT the Microburbs block rent map (which already absorbs block-level effects)
    variants += [(n + "_nomap", c) for n, c in variants]
    # large-n diagnostic (window S only): every Aug-Sep advert (one per flat) scored as if it had no own history and
    # no same-building / stack comps (new-development case), using the same fitted models.
    diag = None
    if wname == "S":
        dte = ads[ads.m.isin(months)].drop_duplicates("flat").reset_index(drop=True)
        dtf = dte.merge(F, on="flat", how="left")
        Xd0 = dtf[BASE_COLS[:-2]].assign(comp=np.nan, stack_comp=np.nan)
        diag = dict(te=dte.assign(has_hist=False, no_comp=True), P={}, cov={})
    for name, extra in variants:
        drop = ["block_map"] if name.endswith("_nomap") else []
        Xtr = pd.concat([Xtr0.drop(columns=drop), tr[extra]], axis=1)
        Xte = pd.concat([Xte0.drop(columns=drop), tf[extra]], axis=1)
        mdl = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=30,
                                subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1, random_state=seed)
        mdl.fit(Xtr, y_tr, categorical_feature=["sub"])
        g = np.exp(mdl.predict(Xte) + idx[tf.m.values])
        P[name] = np.where(nh, g, m1)
        cov[name] = nh
        if diag is not None:
            Xd = pd.concat([Xd0.drop(columns=drop), dtf[extra]], axis=1)
            diag["P"][name] = np.exp(mdl.predict(Xd) + idx[dtf.m.values])
            diag["cov"][name] = np.ones(len(dtf), bool)
    # stale-rent gate on G_all: own advert > 12 months old and > 20% off the median of >= 3 recent same-bld same-bed comps
    a12 = a[a.m > T - 12]
    g12 = a12.groupby(["bld", "bed"]).adj_log.agg(["median", "size"]).reset_index()
    x = tf[["bld", "bed"]].merge(g12, on=["bld", "bed"], how="left")
    c12 = np.exp(x["median"].values + tadj)
    stale = (~nh) & ((tf.m.values - np.nan_to_num(own_m, nan=1e9)) > 12) & (x["size"].fillna(0).values >= 3) & \
            (np.abs(P["G_all"] / c12 - 1) > 0.20)
    P["G_all_gate"] = np.where(stale, 0.5 * P["G_all"] + 0.5 * c12, P["G_all"])
    cov["G_all_gate"] = stale
    # MedAE shift calibration: chosen on window A only, then frozen
    y = te.price.values
    if wname == "A":
        shifts = {s: np.median(np.abs(P["G_all_gate"] + s - y)) for s in (-10, 0, 10)}
        shared["shift"] = min(shifts, key=shifts.get)
    P["G_all_gate_calib"] = P["G_all_gate"] + shared["shift"]
    P["snap5"] = np.round(P["G_all_gate_calib"] / 5) * 5
    P["snap10"] = np.round(P["G_all_gate_calib"] / 10) * 10
    shared["diag"] = diag
    return te, P, cov


RUNGS = [("M0", None), ("M1", "M0"), ("M2", "M1"), ("M3", "M2"), ("M3_nowing", "M2"), ("G_base", "M3")] + \
        [(f"G_{g}", "G_base") for g in GROUPS] + [("G_all", "G_base"), ("G_all_gate", "G_all"),
                                                   ("G_all_gate_calib", "G_all_gate"), ("snap5", "G_all_gate_calib"),
                                                   ("snap10", "G_all_gate_calib"), ("G_base_nomap", "G_base")] +         [(f"G_{g}_nomap", "G_base_nomap") for g in GROUPS] + [("G_all_nomap", "G_base_nomap")]


def metrics(te, P, cov, rng, rungs=None):
    y = te.price.values
    nh = ~te.has_hist.values
    slices = {"all": np.ones(len(y), bool), "own_history": ~nh, "no_history": nh,
              "no_history_no_bldg_comps": nh & te.no_comp.values}
    out = []
    bidx = {k: rng.integers(0, m.sum(), (N_BOOT, m.sum())) for k, m in slices.items() if m.sum() > 0}
    for r, prev in (rungs or RUNGS):
        e = np.abs(P[r] - y)
        ep = np.abs(P[prev] - y) if prev else None
        for sname, m in slices.items():
            if m.sum() == 0:
                continue
            ys, ps, es = y[m], P[r][m], e[m]
            row = dict(rung=r, prev=prev, slice=sname, n=int(m.sum()), medae=float(np.median(es)),
                       within100=float(np.mean(es <= 100)),
                       r2=float(1 - np.sum((ys - ps) ** 2) / np.sum((ys - ys.mean()) ** 2)),
                       coverage=float(np.mean(cov[r][m])) if r in cov else np.nan)
            if prev:
                d = np.median(es) - np.median(ep[m])
                bi = bidx[sname]
                bs = np.median(es[bi], axis=1) - np.median(ep[m][bi], axis=1)
                row.update(d_medae=float(d), ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5)))
            out.append(row)
    return out


def calibrate_base(Bs):
    base = 520.0
    for it in range(3):
        rng = np.random.default_rng(1000)
        F, ads, info = make_world(Bs, rng, "plausible", 0.03, base)
        T = M_IDX[pd.Period("2026-07", "M")]
        te = make_test(ads, F, T, [M_IDX[pd.Period(m, "M")] for m in ("2026-08", "2026-09")], PARAMS["n_scored"], rng)
        tf = te.merge(F, on="flat")
        m0 = ladder(ads[ads.m <= T], F, tf, T)
        mae = float(np.median(np.abs(m0 - te.price.values)))
        print(f"calibration iter {it}: base {base:.1f} -> M0 MedAE {mae:.1f}, median advert {info['median_price']:.0f}",
              flush=True)
        if abs(mae - 100) < 3:
            break
        base *= 100.0 / mae
    return base, mae, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    Bs = load_buildings()
    print(f"{len(Bs)} buildings, {Bs.n_flats.sum()} flats, {Bs['sub'].nunique()} suburbs(SA2), {Bs.blk.nunique()} blocks(MB)")
    base, m0cal, info0 = calibrate_base(Bs)
    PARAMS["base_rent"] = base
    print("world info (calibration world):", info0, flush=True)
    grid = [(s, bn) for s in SCENARIOS for bn in (0.03, 0.10)]
    seeds = range(args.seeds)
    if args.quick:
        grid, seeds = [("strong", 0.03)], range(1)
    rows, infos = [], []
    for seed in seeds:
        for scen, bn in grid:
            t1 = time.time()
            rng = np.random.default_rng(seed)
            F, ads, info = make_world(Bs, rng, scen, bn, base)
            Ta = M_IDX[pd.Period(WINDOWS["A"][0], "M")]
            validated = validate_stacks(ads, F, Ta)
            shared = {}
            for w, (tend, tms) in WINDOWS.items():
                T = M_IDX[pd.Period(tend, "M")]
                months = [M_IDX[pd.Period(m, "M")] for m in tms]
                te, P, cov = run_window(F, ads, w, T, months, PARAMS["n_scored"] if w == "S" else None, rng, validated,
                                        seed, shared)
                for r in metrics(te, P, cov, rng):
                    rows.append(dict(scenario=scen, block_noise=bn, seed=seed, window=w, **r))
                if shared.get("diag"):
                    d = shared.pop("diag")
                    rg = [(r, p) for r, p in RUNGS if r.startswith("G_") and "gate" not in r]
                    rg = [(r, (None if r == "G_base" else p)) for r, p in rg]
                    for r in metrics(d["te"], d["P"], d["cov"], rng, rungs=rg):
                        if r["slice"] == "all":
                            rows.append(dict(scenario=scen, block_noise=bn, seed=seed, window="S_diag_nocomp", **r))
            info.update(scenario=scen, block_noise=bn, seed=seed, n_validated_buildings=len(validated),
                        shift_chosen_on_A=shared["shift"])
            infos.append(info)
            print(f"seed {seed} {scen:9s} noise {bn:.2f}: {time.time()-t1:.0f}s, validated bld {len(validated)}, "
                  f"floor_obs {info['floor_obs_share']:.2f}, shift {shared['shift']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "sim_public_results.csv"), index=False)
    pd.DataFrame(infos).to_csv(os.path.join(HERE, "sim_public_world_info.csv"), index=False)
    with open(os.path.join(HERE, "sim_public_calibration.txt"), "w") as f:
        f.write(f"base_rent={base:.2f}\nM0_MedAE_calibration_world={m0cal:.1f}\n{info0}\nPARAMS={PARAMS}\nEFFECTS={EFFECTS}\n")
    # compact summaries
    pd.set_option("display.width", 250)
    s = df[(df.window == "S")].groupby(["scenario", "block_noise", "rung", "slice"], sort=False)[
        ["n", "medae", "within100", "r2", "coverage", "d_medae"]].mean().round(2)
    print(s.to_string())
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
