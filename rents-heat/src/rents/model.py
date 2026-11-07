"""Floor estimate + "one flat, one vote" same-building comps (SPEC.md Method, v4 / round 6).

fit() sees only training adverts (asserted <= train_end). Everything it builds - repeat-pair suburb index, parking
premium, ladder medians, comps signals - comes from those rows. combine() turns a fit into predictions for a method:
  floor    own last advert x suburb index (own history) / ladder suburb x bed x parking -> suburb x bed -> bed
  plain    no-history flats: mean of all same-building same-bedroom adverts (index + parking adjusted), no shrink
  onevote  no-history flats: one signal per neighbouring flat, target excluded, shrunk to the floor by k/(k+n_flats)
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, diags, vstack
from scipy.sparse.linalg import lsqr

from rents.load import assert_no_leak


def mnum(d):
    return (d.dt.year * 12 + d.dt.month - 1).astype(int)


# ------------------------------------------------------------------ repeat-pair suburb index
def _solve(m, m0, y, n_m, smooth, ridge=0.0):
    """Monthly log index I (I[0]=0) from pairs y = I[m]-I[m0], random-walk smoothing rows, optional ridge."""
    if n_m <= 1:
        return np.zeros(max(n_m, 1))
    k = len(y)
    r = np.arange(k)
    X = coo_matrix((np.r_[np.ones(k), -np.ones(k)], (np.r_[r, r], np.r_[m, m0])), shape=(k, n_m))
    D = diags([-np.ones(n_m - 1), np.ones(n_m - 1)], [0, 1], shape=(n_m - 1, n_m)) * smooth
    A = vstack([X, D]).tocsr()[:, 1:]
    b = np.r_[y, np.zeros(n_m - 1)]
    return np.r_[0.0, lsqr(A, b, damp=ridge, atol=1e-10, btol=1e-10)[0]]


class RentIndex:
    def __init__(self, m_lo, m_hi, city, sub_dev):
        self.m_lo, self.m_hi, self.city, self.sub_dev = m_lo, m_hi, city, sub_dev

    def log_at(self, suburb, m):
        mi = np.clip(np.asarray(m, dtype=int), self.m_lo, self.m_hi) - self.m_lo
        out = self.city[mi].astype(float)
        sub = np.asarray(suburb, dtype=object)
        for s, dev in self.sub_dev.items():
            sel = sub == s
            if sel.any():
                out[sel] += dev[mi[sel]]
        return out

    def log_to_now(self, suburb, m):
        """log(index at train_end month / index at month m), per suburb."""
        return self.log_at(suburb, np.full(len(np.atleast_1d(m)), self.m_hi)) - self.log_at(suburb, m)


def episodes(adv, gap_days):
    """Collapse each flat's adverts into lease episodes (gap > gap_days starts a new one); keep the last price."""
    a = adv.sort_values(["gnaf_id", "date"])
    gap = a.groupby("gnaf_id")["date"].diff().dt.days
    a = a.assign(ep=(gap.isna() | (gap > gap_days)).cumsum().values)
    return a.groupby("ep", sort=False).agg(gnaf_id=("gnaf_id", "first"), date=("date", "last"),
                                           price=("price", "last"), suburb_name=("suburb_name", "first"))


def build_index(train, m_hi, mc):
    e = episodes(train, mc["episode_gap_days"]).sort_values(["gnaf_id", "date"])
    e["m"] = mnum(e["date"])
    prev = e.groupby("gnaf_id")[["m", "price"]].shift(1)
    pr = e.assign(m0=prev["m"], p0=prev["price"]).dropna(subset=["m0"])
    pr = pr[pr["m"] > pr["m0"]]
    m_lo = int(min(e["m"].min(), m_hi - 1)) if len(e) else m_hi - 1
    n_m = m_hi - m_lo + 1
    if not len(pr):
        return RentIndex(m_lo, m_hi, np.zeros(n_m), {}), 0
    m, m0 = pr["m"].values - m_lo, pr["m0"].values.astype(int) - m_lo
    y = np.log(pr["price"].values) - np.log(pr["p0"].values)
    city = _solve(m, m0, y, n_m, mc["index_smooth"])
    resid = y - (city[m] - city[m0])
    dev = {}
    for s, idx in pd.Series(np.arange(len(pr)), index=pr["suburb_name"].values).groupby(level=0):
        ii = idx.values
        if len(ii) >= 5:
            dev[s] = _solve(m[ii], m0[ii], resid[ii], n_m, mc["suburb_index_smooth"], mc["suburb_index_ridge"])
    return RentIndex(m_lo, m_hi, city, dev), int(len(pr))


# ------------------------------------------------------------------ helpers
def advert_parking(car_spaces, park_in):
    """1/0 from the advert's car_spaces, else the flat's input parking, else NaN."""
    pa = (car_spaces > 0).astype(float).where(car_spaces.notna())
    return pa.fillna(park_in)


def _median_table(df, keys, val, min_n):
    g = df.groupby(keys)[val].agg(["median", "size"])
    return g[g["size"] >= min_n]["median"]


def _lookup(table, frame, keys):
    if not len(table):
        return np.full(len(frame), np.nan)
    t = table.rename("v").reset_index()
    return frame[keys].merge(t, on=keys, how="left")["v"].values


class Fit:
    pass


def fit(flats, train, train_end, mc):
    """flats: read_input() frame. train: canonical adverts already restricted to <= train_end."""
    assert_no_leak(train, train_end)
    F = flats[["gnaf_id", "suburb_name", "bed", "bkey", "park_in"]]
    a = train.merge(F, on="gnaf_id", how="inner")
    m_hi = int(mnum(pd.Series([train_end]))[0])
    idx, n_pairs = build_index(a, m_hi, mc)
    a["m"] = mnum(a["date"])
    a["lp_now"] = np.log(a["price"].values) + idx.log_to_now(a["suburb_name"].values, a["m"].values)
    a["pa"] = advert_parking(a["car_spaces"], a["park_in"])
    lb = a[a["m"] > m_hi - mc["ladder_lookback_months"]]

    # parking premium (log): suburb x bed (n>=min each side) -> bed -> citywide; clipped
    def gap(df, keys):
        d = df[df["pa"].notna()]
        g = d.groupby(keys + ["pa"])["lp_now"].agg(["median", "size"]).reset_index()
        w = g[g["pa"] == 1].merge(g[g["pa"] == 0], on=keys, suffixes=("1", "0"))
        w = w[(w["size1"] >= mc["premium_min_n"]) & (w["size0"] >= mc["premium_min_n"])]
        return w.set_index(keys)["median1"] - w.set_index(keys)["median0"]
    d_all = lb[lb["pa"].notna()]
    glob = (d_all[d_all["pa"] == 1]["lp_now"].median() - d_all[d_all["pa"] == 0]["lp_now"].median()) if len(d_all) else 0.08
    glob = 0.08 if not np.isfinite(glob) else glob
    lo, hi = mc["premium_clip"]

    def prem_for(frame):
        p = _lookup(gap(lb, ["suburb_name", "bed"]), frame, ["suburb_name", "bed"])
        p = np.where(np.isnan(p), _lookup(gap(lb.assign(_c=0), ["bed"]), frame, ["bed"]), p)
        return np.clip(np.where(np.isnan(p), glob, p), lo, hi)

    a["prem"] = prem_for(a)
    lb = a[a["m"] > m_hi - mc["ladder_lookback_months"]]
    lb = lb.assign(lp0=lb["lp_now"] - lb["prem"] * lb["pa"].fillna(0))

    # ladder (official example): suburb x bed x parking -> suburb x bed -> bed -> all; n >= ladder_min_n
    out = F.copy()
    out["prem"] = prem_for(out)
    n = mc["ladder_min_n"]
    lb1 = lb[lb["pa"].notna()].assign(p_now=np.exp(lb["lp_now"]))
    t1 = _median_table(lb1, ["suburb_name", "bed", "pa"], "p_now", n)
    lb0 = lb.assign(p0=np.exp(lb["lp0"]))
    t2 = _median_table(lb0, ["suburb_name", "bed"], "p0", n)
    t3 = _median_table(lb0, ["bed"], "p0", n)
    t4 = float(lb0["p0"].median()) if len(lb0) else 600.0
    base0 = _lookup(t2, out, ["suburb_name", "bed"])
    level = np.where(np.isnan(base0), "bed", "suburb_bed")
    b3 = _lookup(t3, out, ["bed"])
    level = np.where(np.isnan(base0) & np.isnan(b3), "all", level)
    base0 = np.where(np.isnan(base0), b3, base0)
    base0 = np.where(np.isnan(base0), t4, base0)
    w0 = _lookup(t1, out.assign(pa=0.0), ["suburb_name", "bed", "pa"])
    w1 = _lookup(t1, out.assign(pa=1.0), ["suburb_name", "bed", "pa"])
    lad_without = np.where(np.isnan(w0), base0, w0)
    lad_with = np.where(np.isnan(w1), base0 * np.exp(out["prem"].values), w1)
    lad_with = np.maximum(lad_with, lad_without)
    out["ladder_level"] = np.where(~np.isnan(w0) | ~np.isnan(w1), "suburb_bed_parking", level)

    # own history: latest own advert <= train_end WITH THE SAME BEDROOMS as the input row (real API data: 9% of
    # repeat adverts change bedrooms - whole-building ids, split lettings - and their errors are 2.4x larger),
    # index-adjusted to train_end. Flats whose only adverts have other bedroom counts take the no-history path.
    bed_adv = a["bedrooms"].clip(upper=mc["bed_cap"])
    same_bed = bed_adv.isna() | (bed_adv == a["bed"])
    last = a[same_bed].sort_values("date").groupby("gnaf_id").tail(1).set_index("gnaf_id")
    adj = np.exp(last["lp_now"])
    pa_last = last["pa"].fillna(0)
    own_with = np.where(pa_last == 1, adj, adj * np.exp(last["prem"]))
    own_without = np.where(pa_last == 1, adj / np.exp(last["prem"]), adj)
    out["own"] = out["gnaf_id"].isin(last.index)
    out["own_any"] = out["gnaf_id"].isin(a["gnaf_id"])
    out["own_age_m"] = out["gnaf_id"].map(m_hi - last["m"]).values  # months since that advert
    out["floor_with"] = lad_with
    out["floor_without"] = lad_without
    out["ladder_with"] = lad_with
    out["ladder_without"] = lad_without
    sel = out["own"].values
    gid = out.loc[sel, "gnaf_id"]
    out.loc[sel, "floor_with"] = pd.Series(own_with, index=last.index).reindex(gid).values
    out.loc[sel, "floor_without"] = pd.Series(own_without, index=last.index).reindex(gid).values

    # comps: signals on the without-parking log scale, lookback window, same building key x same bedrooms
    cw = a[a["m"] > m_hi - mc["comps_lookback_months"]].copy()
    cw["s"] = cw["lp_now"] - cw["prem"] * cw["pa"].fillna(0)
    keys = ["bkey", "bed"]
    # one flat, one vote: latest date-eligible advert per neighbour flat
    one = cw.sort_values("date").groupby("gnaf_id").tail(1)[["gnaf_id", "bkey", "bed", "s"]]
    g1 = one.groupby(keys)["s"].agg(S1="sum", N1="size").reset_index()
    own1 = one.set_index("gnaf_id")["s"]
    # plain: every advert counts
    gp = cw.groupby(keys)["s"].agg(SP="sum", NP="size").reset_index()
    ownp = cw.groupby("gnaf_id")["s"].agg(["sum", "size"])
    o = out[["gnaf_id"] + keys].merge(g1, on=keys, how="left").merge(gp, on=keys, how="left")
    s_self = o["gnaf_id"].map(own1)
    n1 = o["N1"].fillna(0) - s_self.notna().astype(int)
    o["n_comp_flats"] = n1.values
    o["comp_onevote"] = np.where(n1 > 0, (o["S1"].fillna(0) - s_self.fillna(0)) / n1.replace(0, np.nan), np.nan)
    sp_self, np_self = o["gnaf_id"].map(ownp["sum"]).fillna(0), o["gnaf_id"].map(ownp["size"]).fillna(0)
    nP = o["NP"].fillna(0) - np_self
    o["n_comp_adverts"] = nP.values
    o["comp_plain"] = np.where(nP > 0, (o["SP"].fillna(0) - sp_self) / nP.replace(0, np.nan), np.nan)
    for c in ["n_comp_flats", "comp_onevote", "n_comp_adverts", "comp_plain"]:
        out[c] = o[c].values

    f = Fit()
    f.table, f.index, f.train_end = out.reset_index(drop=True), idx, train_end
    f.info = dict(train_end=str(train_end.date()), n_train_adverts=int(len(train)),
                  n_train_adverts_input_flats=int(len(a)), n_repeat_pairs=n_pairs,
                  n_flats_own_history=int(out["own"].sum()), premium_citywide=round(float(glob), 4),
                  index_change_last_12m=round(float(idx.city[-1] - idx.city[max(0, len(idx.city) - 13)]), 4))
    return f


def combine(f, method="floor", k=5.0, stale_m=None, stale_w=0.0, snap=None):
    """Predictions for one rule.
    method  floor | plain | onevote: the no-history estimate (ladder, or same-building same-bedroom comps where available)
    k       one-vote shrinkage toward the ladder
    stale_m own-history rows whose last matching advert is older than this many months are treated as stale:
            their estimate becomes stale_w * own + (1 - stale_w) * no-history estimate (log scale); None = never
    snap    round both columns to the nearest $snap (asking rents cluster on $10 / $50 steps); None = no snapping"""
    t = f.table
    own = t["own"].values
    prem = np.exp(t["prem"].values)
    # no-history estimate for EVERY flat (own flats excluded from their own comps in fit())
    nh_wo, nh_w = t["ladder_without"].values.copy(), t["ladder_with"].values.copy()
    ok = np.zeros(len(t), bool)
    if method == "plain":
        ok = np.isfinite(t["comp_plain"].values)
        nh_wo[ok] = np.exp(t["comp_plain"].values[ok])
    elif method == "onevote":
        ok = np.isfinite(t["comp_onevote"].values)
        n = t["n_comp_flats"].values[ok]
        nh_wo[ok] = np.exp((n * t["comp_onevote"].values[ok] + k * np.log(nh_wo[ok])) / (n + k))
    elif method != "floor":
        raise ValueError(method)
    nh_w[ok] = nh_wo[ok] * prem[ok]
    w, wo = nh_w.copy(), nh_wo.copy()
    used = ok & ~own
    # own-history rows: own estimate (floor_* holds it for own rows), blended toward no-history when stale
    ow, owo = t["floor_with"].values, t["floor_without"].values
    stale = (own & (t["own_age_m"].values > stale_m)) if stale_m is not None else np.zeros(len(t), bool)
    fresh = own & ~stale
    w[fresh], wo[fresh] = ow[fresh], owo[fresh]
    if stale.any():
        sw = float(stale_w)
        w[stale] = np.exp(sw * np.log(ow[stale]) + (1 - sw) * np.log(nh_w[stale]))
        wo[stale] = np.exp(sw * np.log(owo[stale]) + (1 - sw) * np.log(nh_wo[stale]))
        used |= stale & ok
    if snap:
        w, wo = np.round(w / snap) * snap, np.round(wo / snap) * snap
    return pd.DataFrame({"gnaf_id": t["gnaf_id"].values, "rent_with_car_space": w, "rent_without_car_space": wo,
                         "own": own, "used_comps": used, "stale_own": stale, "bkey": t["bkey"].values})


def combine_rule(f, fr):
    """combine() from a frozen-rule dict (keys method, k, stale_m, stale_w, snap; missing keys = defaults)."""
    return combine(f, fr["method"], fr.get("k") or 0.0, fr.get("stale_m"), fr.get("stale_w") or 0.0, fr.get("snap"))


# ------------------------------------------------------------------ evaluation
def metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(yhat)
    n = len(y)
    if n == 0:
        return dict(n=0, medae=None, within100=None, r2=None, coverage=None)
    e = np.abs(y[ok] - yhat[ok])
    yy = y[ok]
    sst = np.sum((yy - yy.mean()) ** 2)
    return dict(n=int(n), medae=round(float(np.median(e)), 2) if ok.any() else None,
                within100=round(float(np.mean(e <= 100)), 4) if ok.any() else None,
                r2=round(float(1 - np.sum((yy - yhat[ok]) ** 2) / sst), 4) if sst > 0 else None,
                coverage=round(float(ok.mean()), 4))


def score(pred, flats, test):
    """Score test adverts: column chosen by advert parking (car_spaces), else input parking (else without)."""
    t = test[test["gnaf_id"].isin(flats["gnaf_id"])].merge(
        flats[["gnaf_id", "park_in"]], on="gnaf_id", how="left").merge(pred, on="gnaf_id", how="left")
    pa = advert_parking(t["car_spaces"], t["park_in"]).fillna(0).values
    yhat = np.where(pa == 1, t["rent_with_car_space"].values, t["rent_without_car_space"].values)
    y = t["price"].values
    own = t["own"].fillna(False).values.astype(bool)
    res = {"own_history": metrics(y[own], yhat[own]), "no_history": metrics(y[~own], yhat[~own]),
           "all": metrics(y, yhat)}
    u = t["used_comps"].fillna(False).values.astype(bool)
    res["comps_applied"] = dict(test_adverts=int(u.sum()), test_flats=int(t.loc[u, "gnaf_id"].nunique()),
                                test_buildings=int(t.loc[u, "bkey"].nunique()))
    res["n_test_adverts_not_in_input"] = int((~test["gnaf_id"].isin(flats["gnaf_id"])).sum())
    return res


def round_half(x):
    return np.round(np.asarray(x, float) * 2) / 2
