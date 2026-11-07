"""Simulation study: which factors lower median absolute rent error, and is a heat effect detectable?

Synthetic City-of-Sydney-shaped world (all parameters are ASSUMPTIONS, listed in PARAMS / scenario grid).
Scoring mimics the task: 1,699 test adverts dated Aug-Sep 2026, training adverts <= 31 Jul 2026,
median absolute $ error ranks, share within $100 breaks ties, R^2 reported.

Methods (cumulative):
  M0  ladder median: suburb x bedrooms x parking -> suburb x bedrooms -> bedrooms (>=10 adverts, last 12 months)  [official example]
  M1  own last advert x citywide repeat-rent index; rows without own history use M0
  M2  M1 + no-history rows use same-building same-bedroom recent adverts (time-adjusted) when available
  M3  M2 + no-history rows use LightGBM on Microburbs-side features (suburb, bed, bath, parking, floor, flats_in_building,
      block rent map, building-comps median)
  M4  M3 + heat and canopy features
Run: python sim/sim_study.py [--quick]
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

PARAMS = dict(
    n_flats=65_777, n_suburbs=33, n_blocks=700, mean_flats_per_building=22,
    base_rent=520.0,
    sd_suburb=0.10, sd_block=0.07, sd_building=0.10, sd_unit=0.10, sd_advert=0.05,
    floor_per_level=0.006, parking=0.08, bath2=0.06,
    bed_effect={0: -0.37, 1: 0.0, 2: 0.32, 3: 0.63, 4: 0.92},  # from NSW bond data, CoS flats Aug-Sep 2026 medians
    bed_p=[0.10, 0.38, 0.40, 0.10, 0.02],
    p_parking=0.55, p_floor_known=0.52, p_ever_advertised=0.76, adverts_per_year=0.30,
    corr_heat_affluence=-0.4, corr_heat_canopy=-0.6, block_map_noise=0.03,
    n_test=1699,
)
# calibration target (NSW bond data, CoS flats): within postcode x bedrooms median abs log deviation 0.119
MONTHS = pd.period_range("2016-01", "2026-09", freq="M")
M_IDX = {p: i for i, p in enumerate(MONTHS)}
TRAIN_END = M_IDX[pd.Period("2026-07", "M")]
TEST_MONTHS = [M_IDX[pd.Period("2026-08", "M")], M_IDX[pd.Period("2026-09", "M")]]


def market_index(n_months, rng):
    t = np.arange(n_months) / 12.0
    idx = 0.03 * t
    covid = (np.arange(n_months) >= M_IDX[pd.Period("2020-04", "M")]) & (np.arange(n_months) <= M_IDX[pd.Period("2021-12", "M")])
    idx = idx - 0.12 * covid
    rebound = np.clip((np.arange(n_months) - M_IDX[pd.Period("2022-01", "M")]) / 24.0, 0, 1) * 0.08
    return idx + rebound + rng.normal(0, 0.005, n_months)


def make_world(rng, heat_beta, canopy_beta, p_hist_test):
    P = PARAMS
    S, B = P["n_suburbs"], P["n_blocks"]
    suburb_eff = rng.normal(0, P["sd_suburb"], S)
    suburb_growth = rng.normal(0, 0.01, S)
    block_suburb = rng.integers(0, S, B)
    block_eff = rng.normal(0, P["sd_block"], B)
    z_aff = block_eff / P["sd_block"]
    r = P["corr_heat_affluence"]
    z_heat = r * z_aff + np.sqrt(1 - r * r) * rng.normal(0, 1, B)
    rc = P["corr_heat_canopy"]
    z_can = rc * z_heat + np.sqrt(1 - rc * rc) * rng.normal(0, 1, B)
    env_eff = heat_beta * z_heat + canopy_beta * z_can
    # Microburbs block rent map is built from rents, so it already contains the environmental effect
    block_map = block_eff + env_eff + rng.normal(0, P["block_map_noise"], B)

    # buildings
    sizes = []
    total = 0
    while total < P["n_flats"]:
        s = max(1, int(rng.lognormal(np.log(P["mean_flats_per_building"]) - 0.5, 1.0)))
        sizes.append(s)
        total += s
    sizes[-1] -= total - P["n_flats"]
    nb = len(sizes)
    bld_block = rng.integers(0, B, nb)
    bld_eff = rng.normal(0, P["sd_building"], nb)
    bld_height = np.clip((np.array(sizes) / 4).astype(int) + 1, 1, 60)

    bld = np.repeat(np.arange(nb), sizes)
    n = P["n_flats"]
    blk = bld_block[bld]
    sub = block_suburb[blk]
    bed = rng.choice(5, n, p=P["bed_p"])
    bath = np.where((bed >= 2) & (rng.random(n) < 0.4), 2, 1)
    park = (rng.random(n) < P["p_parking"]).astype(int)
    floor = rng.integers(0, bld_height[bld] + 1)
    floor_obs = np.where(rng.random(n) < P["p_floor_known"], floor, np.nan)
    unit_eff = rng.normal(0, P["sd_unit"], n)
    bed_e = np.array([P["bed_effect"][b] for b in range(5)])[bed]
    level = (np.log(P["base_rent"]) + suburb_eff[sub] + block_eff[blk] + env_eff[blk] + bld_eff[bld] + unit_eff
             + bed_e + P["bath2"] * (bath == 2) + P["floor_per_level"] * floor)
    flats = pd.DataFrame(dict(flat=np.arange(n), bld=bld, blk=blk, sub=sub, bed=bed, bath=bath, park=park,
                              floor_obs=floor_obs, fib=np.array(sizes)[bld], level=level,
                              block_map=block_map[blk], heat=z_heat[blk] + rng.normal(0, 0.2, n),
                              canopy=z_can[blk] + rng.normal(0, 0.2, n)))

    mkt = market_index(len(MONTHS), rng)
    ever = rng.random(n) < P["p_ever_advertised"]
    lam = P["adverts_per_year"] * (TRAIN_END + 1) / 12.0
    k = np.where(ever, np.maximum(1, rng.poisson(lam, n)), 0)
    a_flat = np.repeat(np.arange(n), k)
    a_m = rng.integers(0, TRAIN_END + 1, a_flat.size)
    ads = pd.DataFrame(dict(flat=a_flat, m=a_m))

    hist_flats = np.unique(a_flat)
    no_hist = np.setdiff1d(np.arange(n), hist_flats)
    n_h = int(round(P["n_test"] * p_hist_test))
    test_flats = np.concatenate([rng.choice(hist_flats, n_h, replace=False),
                                 rng.choice(no_hist, P["n_test"] - n_h, replace=False)])
    test = pd.DataFrame(dict(flat=test_flats, m=rng.choice(TEST_MONTHS, test_flats.size)))

    def price(df):
        f = flats.loc[df.flat.values]
        lp = (f.level.values + P["parking"] * f.park.values + mkt[df.m.values]
              + suburb_growth[f["sub"].values] * df.m.values / 12.0 + rng.normal(0, P["sd_advert"], len(df)))
        return np.round(np.exp(lp) / 5) * 5

    ads["price"] = price(ads)
    test["price"] = price(test)
    test["has_hist"] = np.isin(test.flat.values, hist_flats)
    return flats, ads, test


def repeat_index(ads):
    """Citywide monthly log index from repeat adverts (consecutive pairs), least squares, month 0 = 0."""
    a = ads.sort_values(["flat", "m"])
    prev = a.groupby("flat").shift(1)
    pr = a.assign(m0=prev.m, p0=prev.price).dropna()
    pr = pr[pr.m0 < pr.m]
    nm = TRAIN_END + 1
    rows = np.arange(len(pr))
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr
    X = coo_matrix((np.r_[np.ones(len(pr)), -np.ones(len(pr))],
                    (np.r_[rows, rows], np.r_[pr.m.values.astype(int), pr.m0.values.astype(int)])), shape=(len(pr), nm)).tocsr()
    y = np.log(pr.price.values) - np.log(pr.p0.values)
    beta = lsqr(X[:, 1:], y, damp=1e-3)[0]
    idx = np.r_[0.0, beta]
    # carry forward to test months
    return np.r_[idx, np.full(len(MONTHS) - nm, idx[-1])]


def ladder(ads, flats, keys_df):
    recent = ads[ads.m > TRAIN_END - 12].merge(flats[["flat", "sub", "bed", "park"]], on="flat")
    out = np.full(len(keys_df), np.nan)
    for keys in (["sub", "bed", "park"], ["sub", "bed"], ["bed"]):
        g = recent.groupby(keys).price.agg(["median", "size"])
        g = g[g["size"] >= 10]["median"]
        vals = keys_df[keys].merge(g.rename("v").reset_index(), on=keys, how="left").v.values
        out = np.where(np.isnan(out), vals, out)
    return np.where(np.isnan(out), recent.price.median(), out)


def evaluate(pred, test):
    err = np.abs(pred - test.price.values)
    y = test.price.values
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    nh = ~test.has_hist.values
    return dict(mae_med=np.median(err), within100=np.mean(err <= 100), r2=r2,
                mae_med_nohist=np.median(err[nh]) if nh.any() else np.nan)


def run_once(seed, heat_beta, canopy_beta, p_hist_test):
    rng = np.random.default_rng(seed)
    flats, ads, test = make_world(rng, heat_beta, canopy_beta, p_hist_test)
    idx = repeat_index(ads)
    tf = test.merge(flats, on="flat", how="left")

    preds = {}
    m0 = ladder(ads, flats, tf)
    preds["M0"] = m0

    last = ads.sort_values("m").groupby("flat").tail(1).set_index("flat")
    own = tf.flat.map(last.price)
    own_m = tf.flat.map(last.m)
    m1 = np.where(own.notna(), own * np.exp(idx[tf.m.values] - idx[own_m.fillna(0).astype(int).values]), m0)
    preds["M1"] = m1

    # same building, same bedrooms, last 24 months, time-adjusted to test month, excluding the flat itself
    a = ads[ads.m > TRAIN_END - 24].merge(flats[["flat", "bld", "bed", "park"]], on="flat")
    a["adj_log"] = np.log(a.price) - idx[a.m.values] - PARAMS["parking"] * 0  # parking handled below
    pk = a.groupby("park").adj_log.median()
    park_gap = pk.get(1, 0) - pk.get(0, 0)
    a["adj_log"] -= park_gap * a.park
    # leave-own-flat-out mean of other flats' adverts in the same building and bedroom count
    g_all = a.groupby(["bld", "bed"]).adj_log.agg(["sum", "size"]).rename(columns={"sum": "S", "size": "N"})
    g_flat = a.groupby("flat").adj_log.agg(["sum", "size"]).rename(columns={"sum": "Sf", "size": "Nf"})

    def loo_comp(df):
        x = df[["flat", "bld", "bed"]].merge(g_all.reset_index(), on=["bld", "bed"], how="left")
        x = x.merge(g_flat.reset_index(), on="flat", how="left").fillna({"Sf": 0, "Nf": 0, "S": 0, "N": 0})
        n_other = x.N - x.Nf
        return np.where(n_other > 0, (x.S - x.Sf) / n_other.replace(0, np.nan), np.nan)

    te_comp = loo_comp(tf)
    comp_pred = np.exp(te_comp + idx[tf.m.values] + park_gap * tf.park.values)
    nh = ~tf.has_hist.values
    m2 = np.where(nh & ~np.isnan(comp_pred), comp_pred, m1)
    preds["M2"] = m2

    # GBM for no-history rows: train on last-12-month adverts (time-adjusted to index), features Microburbs-side
    tr = ads[ads.m > TRAIN_END - 12].merge(flats, on="flat")
    tr_comp = loo_comp(tr)
    y_tr = np.log(tr.price.values) - idx[tr.m.values]
    base_cols = ["sub", "bed", "bath", "park", "floor_obs", "fib", "block_map"]
    Xb_tr = tr[base_cols].assign(comp=tr_comp)
    Xb_te = tf[base_cols].assign(comp=te_comp)
    env_cols = ["heat", "canopy"]
    out = {}
    for name, cols_extra in (("M3", []), ("M4", env_cols)):
        Xtr = pd.concat([Xb_tr, tr[cols_extra]], axis=1) if cols_extra else Xb_tr
        Xte = pd.concat([Xb_te, tf[cols_extra]], axis=1) if cols_extra else Xb_te
        mdl = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=30,
                                subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1, random_state=seed)
        mdl.fit(Xtr, y_tr, categorical_feature=["sub"])
        g = np.exp(mdl.predict(Xte) + idx[tf.m.values])
        out[name] = np.where(nh, g, m1)
    preds.update(out)

    res = {k: evaluate(v, test) for k, v in preds.items()}
    # paired bootstrap for M4 - M3 on median abs error (all rows and no-history rows)
    e3 = np.abs(preds["M3"] - test.price.values)
    e4 = np.abs(preds["M4"] - test.price.values)
    bs = []
    for _ in range(500):
        i = rng.integers(0, len(e3), len(e3))
        bs.append(np.median(e4[i]) - np.median(e3[i]))
    res["boot_M4_minus_M3"] = (np.percentile(bs, 2.5), np.percentile(bs, 97.5))
    res["n_nohist"] = int(nh.sum())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    grid = [(hb, cb, ph) for hb in (0.0, 0.01, 0.03) for cb in (0.0,) for ph in (0.6, 0.85)]
    if args.quick:
        grid = [(0.01, 0.0, 0.85)]
    rows = []
    for hb, cb, ph in grid:
        for s in range(args.seeds):
            t0 = time.time()
            r = run_once(s, hb, cb, ph)
            for m in ("M0", "M1", "M2", "M3", "M4"):
                rows.append(dict(heat_beta=hb, p_hist=ph, seed=s, method=m, **r[m]))
            lo, hi = r["boot_M4_minus_M3"]
            rows.append(dict(heat_beta=hb, p_hist=ph, seed=s, method="M4-M3 95%CI",
                             mae_med=np.nan, within100=np.nan, r2=np.nan, mae_med_nohist=np.nan, ci=f"[{lo:+.1f}, {hi:+.1f}]",
                             n_nohist=r["n_nohist"]))
            print(f"heat={hb} p_hist={ph} seed={s} done in {time.time()-t0:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    summ = (df[df.method != "M4-M3 95%CI"].groupby(["heat_beta", "p_hist", "method"])
            [["mae_med", "within100", "r2", "mae_med_nohist"]].mean().round(3))
    print(summ.to_string())
    print(df[df.method == "M4-M3 95%CI"][["heat_beta", "p_hist", "seed", "ci", "n_nohist"]].to_string(index=False))
    df.to_csv("sim/sim_results.csv", index=False)


if __name__ == "__main__":
    main()
