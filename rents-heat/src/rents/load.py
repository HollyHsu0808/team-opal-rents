"""Loaders, column mapping, building-key normaliser and per-window date filtering.

Canonical columns come from config/columns.json (canonical name -> source column). Every training frame
passes through `train_slice`, which hard-asserts that no row is dated after the window's train_end.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # rents-heat/
sys.path.insert(0, os.path.join(ROOT, "src"))
from stack.parse_unit import parse_unit  # noqa: E402  (reused, not rewritten)


def load_json(name):
    with open(os.path.join(ROOT, "config", name), encoding="utf-8") as f:
        return json.load(f)


def cfg():
    return dict(columns=load_json("columns.json"), windows=load_json("windows.json"),
                model=load_json("model.json"), paths=load_json("paths.json"))


def path_for(c, data, key):
    return os.path.normpath(os.path.join(ROOT, c["paths"][data][key]))


# ---------------------------------------------------------------- building key
_STREET_ABBR = {"STREET": "ST", "AVENUE": "AVE", "ROAD": "RD", "PLACE": "PL", "LANE": "LN", "DRIVE": "DR",
                "PARADE": "PDE", "CRESCENT": "CRES", "TERRACE": "TCE", "HIGHWAY": "HWY", "BOULEVARD": "BVD",
                "BOULEVARDE": "BVD", "SQUARE": "SQ", "CLOSE": "CL", "COURT": "CT", "ESPLANADE": "ESP",
                "CIRCUIT": "CCT", "GROVE": "GR", "WALK": "WLK", "AV": "AVE"}
_UNIT_PREFIX = re.compile(
    r"^\s*(?:(?:UNIT|APARTMENT|APT|FLAT|SUITE|STUDIO|SHOP|LOT|U|PH|PENTHOUSE)\.?\s*#?\s*[A-Z0-9-]+\s*,?\s+"
    r"|[A-Z]{0,2}\d{1,5}[A-Z]?\s*/\s*(?=\d))", re.I)


def normalise_building_key(s):
    """'18 Gadigal Avenue, Waterloo NSW 2017' and '18 GADIGAL AVENUE WATERLOO 2017' -> '18 GADIGAL AVE WATERLOO 2017'.
    Upper-case, drop commas / '(NSW)' / 'NSW', contract street types (contracting is safe for 'ST PETERS')."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    t = str(s).upper().replace(",", " ").replace("(NSW)", " ")
    t = re.sub(r"[^\w\s/-]", " ", t)
    toks = [_STREET_ABBR.get(w, w) for w in t.split() if w != "NSW"]
    out = " ".join(toks).strip()
    return out or None


def strip_unit(address):
    """Remove the unit part: 'UNIT 106, 18 GADIGAL AVENUE, WATERLOO NSW 2017' -> '18 GADIGAL AVENUE, WATERLOO NSW 2017';
    '506/20 Dunkerley Pl' -> '20 Dunkerley Pl'. Addresses without a unit come back unchanged."""
    if address is None or (isinstance(address, float) and np.isnan(address)):
        return None
    s = str(address).strip()
    if parse_unit(s).unit_raw is None:
        return s
    return _UNIT_PREFIX.sub("", s, count=1).strip()


def building_key(building_address, address):
    """Building key: normalised building_address; fallback = address with the unit part stripped."""
    k = normalise_building_key(building_address)
    if k:
        return k
    return normalise_building_key(strip_unit(address))


def building_keys(building_address: pd.Series, address: pd.Series) -> pd.Series:
    ba = building_address.astype(object).where(building_address.notna(), None)
    ad = address.astype(object).where(address.notna(), None)
    return pd.Series([building_key(b, a) for b, a in zip(ba, ad)], index=address.index, dtype=object)


# ---------------------------------------------------------------- readers
def _rename(df, cmap, source_name):
    cmap = {k: v for k, v in cmap.items() if not k.startswith("_")}
    missing = [v for v in cmap.values() if v not in df.columns]
    present = {v: k for k, v in cmap.items() if v in df.columns}
    out = df.rename(columns=present)
    return out, missing


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def clean_suburb(s: pd.Series) -> pd.Series:
    """'Waterloo (NSW)' -> 'Waterloo'; 'The Rocks (Sydney - NSW)' -> 'The Rocks'; plain names unchanged."""
    return s.astype(str).str.replace(r"\s*\((?:[^()]*-\s*)?NSW\)\s*$", "", regex=True).str.strip()


def read_input(path, c=None):
    c = c or cfg()
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    df, missing = _rename(raw, c["columns"]["input"], "input")
    if missing:
        raise ValueError(f"input file is missing columns {missing}; fix config/columns.json")
    for col in ["latitude", "longitude", "bedrooms", "bathrooms", "car_space", "car_spaces", "floor_level",
                "flats_in_building", "block_new_homes_2021_2026", "block_homes_2026", "ever_advertised_for_lease"]:
        df[col] = _num(df[col])
    nod = c["columns"]["input_no_date_value"]
    for col in c["columns"]["input_date_columns"]:
        df[col] = pd.to_datetime(df[col].where(df[col] != nod), errors="coerce")
    df["gnaf_id"] = df["gnaf_id"].str.strip()
    df["suburb_name"] = clean_suburb(df["suburb"])
    # the official building_address has no suburb ('1 WILLIAM STREET'): add it so equal street numbers in different
    # suburbs are different buildings (a suburb already inside the key is harmless)
    bk = building_keys(df["building_address"], df["address"])
    df["bkey"] = (bk.fillna("?") + " | " + df["suburb_name"].str.upper()).where(bk.notna(), None)
    # flat parking: car_space 1/0, else car_spaces > 0, else unknown
    park = df["car_space"].where(df["car_space"].isin([0, 1]))
    park = park.fillna((df["car_spaces"] > 0).astype(float).where(df["car_spaces"].notna()))
    df["park_in"] = park
    df["bed"] = df["bedrooms"].fillna(1).clip(0, c["model"]["bed_cap"]).astype(int)
    return df


def read_past_leases(path, c=None):
    c = c or cfg()
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    df, missing = _rename(raw, c["columns"]["past_leases"], "past_leases")
    for m in missing:
        if m in ("gnaf_id", "date", "price"):
            raise ValueError(f"past-leases file lacks required column '{m}'; fix config/columns.json")
    return _finish_adverts(df, "past_leases")


def read_api_rent_history(path, c=None):
    c = c or cfg()
    raw = pd.read_parquet(path)
    df, missing = _rename(raw, c["columns"]["api_rent_history"], "api")
    for m in missing:
        if m in ("gnaf_id", "date", "price"):
            raise ValueError(f"API rent-history parquet lacks required column '{m}'")
    return _finish_adverts(df, "api")


def _finish_adverts(df, source):
    for col in ["gnaf_id", "address", "building_address", "car_spaces", "bedrooms", "bathrooms", "leased"]:
        if col not in df.columns:
            df[col] = np.nan
    df = df.copy()
    df["gnaf_id"] = df["gnaf_id"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = _num(df["price"])
    df["car_spaces"] = _num(df["car_spaces"])
    df["bedrooms"] = _num(df["bedrooms"])
    df["source"] = source
    df["advert_bkey"] = building_keys(df["building_address"], df["address"])
    keep = ["gnaf_id", "date", "price", "car_spaces", "bedrooms", "bathrooms", "leased", "address", "advert_bkey", "source"]
    return df[keep]


def load_adverts(c, data, source="auto", hard_cutoff=True):
    """Adverts for training. source: auto (past-leases file if present, else API parquet) | past_leases | api | both.
    Rows without a date or price are dropped; rows after the task cut-off are dropped (count reported)."""
    pl, ap = path_for(c, data, "past_leases"), path_for(c, data, "api_rent_history")
    frames = []
    if source in ("auto", "past_leases", "both") and os.path.exists(pl):
        frames.append(read_past_leases(pl, c))
    if source in ("api", "both") or (source == "auto" and not frames):
        if os.path.exists(ap):
            frames.append(read_api_rent_history(ap, c))
    if not frames:
        raise FileNotFoundError(f"no advert file found ({pl} / {ap})")
    a = pd.concat(frames, ignore_index=True)
    n0 = len(a)
    a = a[a["date"].notna() & a["price"].notna()]
    a = a[(a["price"] >= c["model"]["price_min"]) & (a["price"] <= c["model"]["price_max"])]
    a = a.drop_duplicates(["gnaf_id", "date", "price"])
    info = dict(n_raw=n0, n_valid=len(a))
    if hard_cutoff:
        cut = pd.Timestamp(c["windows"]["hard_cutoff"])
        info["n_after_cutoff_dropped"] = int((a["date"] > cut).sum())
        a = a[a["date"] <= cut]
    return a.reset_index(drop=True), info


def read_scoring(path):
    """Synthetic-only Aug-Sep 2026 adverts (gnaf_id, date, price, car_spaces)."""
    s = pd.read_csv(path, dtype={"gnaf_id": str})
    s["date"] = pd.to_datetime(s["date"])
    s["price"] = _num(s["price"])
    s["car_spaces"] = _num(s["car_spaces"])
    return s[s["price"].notna()].reset_index(drop=True)


# ---------------------------------------------------------------- windows
def window(c, name):
    w = c["windows"][name]
    out = dict(name=name, train_end=pd.Timestamp(w["train_end"]),
               test_start=pd.Timestamp(w["test_start"]) if w.get("test_start") else None,
               test_end=pd.Timestamp(w["test_end"]) if w.get("test_end") else None)
    assert out["train_end"] <= pd.Timestamp(c["windows"]["hard_cutoff"]), "train_end after the task cut-off"
    return out


def assert_no_leak(df, cutoff, col="date", what="training data"):
    if len(df) and df[col].max() > pd.Timestamp(cutoff):
        raise AssertionError(f"LEAKAGE: {what} has rows dated {df[col].max().date()} > cut-off {pd.Timestamp(cutoff).date()}")


def train_slice(adverts, train_end):
    tr = adverts[adverts["date"] <= train_end].copy()
    assert_no_leak(tr, train_end)
    return tr


def test_slice(adverts, start, end):
    return adverts[(adverts["date"] >= start) & (adverts["date"] <= end)].copy()
