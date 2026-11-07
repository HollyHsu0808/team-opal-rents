"""Build a proxy input file from the API pull, for dry-running A/B on real adverts before the official input arrives.

Flats = every gnaf_pid with at least one advert in data_private/api/listings_rent_history.parquet whose latest
advert is not a House (the task is flats). Attributes come from the latest advert (bedrooms, bathrooms, car_spaces,
lat/lng, property_type) and from data_private/api/properties.parquet (mb, floor, flats_in_building) when present;
floor_level falls back to a high-confidence unit-number parse. Columns mirror the official 19-column input so the
same loaders run unchanged. Output: data_private/apiproxy/rents-input-flats-apiproxy.csv

Not a substitute for the official file: the real input has 65,777 flats over 33 suburbs (including never-advertised
flats and 4 suburbs the API does not serve); this proxy only has advertised flats in the 29 API suburbs.
"""
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from rents.load import ROOT, strip_unit  # noqa: E402
from stack.parse_unit import parse_unit  # noqa: E402

API = os.path.join(ROOT, "data_private", "api")
OUT = os.path.join(ROOT, "data_private", "apiproxy", "rents-input-flats-apiproxy.csv")
COLS = ["gnaf_id", "address", "building_address", "suburb", "mb", "latitude", "longitude", "property_type", "bedrooms",
        "bathrooms", "car_space", "car_spaces", "floor_level", "flats_in_building", "block_new_homes_2021_2026",
        "block_homes_2026", "ever_advertised_for_lease", "first_ever_listed", "attrs_last_seen"]


def main():
    r = pd.read_parquet(os.path.join(API, "listings_rent_history.parquet"))
    p = pd.read_parquet(os.path.join(API, "properties.parquet"),
                        columns=["gnaf_pid", "mb", "floor", "flats_in_building", "dwelling_type"]).drop_duplicates("gnaf_pid")
    r["date"] = pd.to_datetime(r["listing_date"], errors="coerce")
    r = r.sort_values("date")
    last = r.groupby("gnaf_pid").tail(1).copy()
    first = r.groupby("gnaf_pid")["date"].min()
    last = last[last["property_type"].ne("House")]
    last = last.merge(p, on="gnaf_pid", how="left")
    sub = last["suburb_query"].astype(str)
    sub_plain = sub.str.replace(r"\s*\((Sydney - )?NSW\)$", "", regex=True).str.upper()
    street = last["address"].map(strip_unit)
    unit_floor = last["address"].map(lambda a: (lambda u: u.floor_guess if u.confidence == "high" else None)(parse_unit(a)))
    out = pd.DataFrame({
        "gnaf_id": last["gnaf_pid"].values,
        "address": (last["address"].astype(str) + " " + sub_plain).values,
        "building_address": (street.astype(str) + " " + sub_plain).values,
        "suburb": sub.values,
        "mb": last["mb"].values,
        "latitude": last["lat"].values, "longitude": last["lng"].values,
        "property_type": last["property_type"].values,
        "bedrooms": last["bedrooms"].values, "bathrooms": last["bathrooms"].values,
        "car_space": (last["car_spaces"] > 0).astype(float).where(last["car_spaces"].notna()).values,
        "car_spaces": last["car_spaces"].values,
        "floor_level": last["floor"].fillna(pd.Series(unit_floor.values, index=last.index)).values,
        "flats_in_building": last["flats_in_building"].values,
        "block_new_homes_2021_2026": None, "block_homes_2026": None,
        "ever_advertised_for_lease": 1,
        "first_ever_listed": last["gnaf_pid"].map(first).dt.strftime("%Y-%m-%d").values,
        "attrs_last_seen": last["date"].dt.strftime("%Y-%m-%d").values,
    })[COLS]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(out):,} flats, {out['building_address'].nunique():,} building addresses, "
          f"floor_level known {out['floor_level'].notna().mean():.1%} (properties {last['floor'].notna().mean():.1%}), "
          f"car_space blank {out['car_space'].isna().mean():.1%}, suburbs {out['suburb'].nunique()}")
    print("property_type:", out["property_type"].value_counts().to_dict())
    print("sample:", out.iloc[0][["gnaf_id", "address", "building_address", "suburb", "bedrooms", "floor_level"]].to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
