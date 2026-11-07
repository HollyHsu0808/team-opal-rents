"""Probe: can official flats (gnaf_pid, address, lat/lng rounded to 4 dp) be joined to our OSM building table?
Uses the two example records in the organiser OpenAPI spec."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point, Polygon

ROOT = Path(__file__).resolve().parents[1]
b = pd.read_parquet(ROOT / "_derived" / "buildings_features_v2.parquet")
osm = json.load(open(ROOT / "osm" / "osm_buildings_bbox.json", encoding="utf-8"))["elements"]
polys = {}
for e in osm:
    if e.get("type") == "way" and len(e.get("geometry", [])) >= 4:
        polys[("way", e["id"])] = Polygon([(p["lon"], p["lat"]) for p in e["geometry"]])
fp = gpd.GeoDataFrame({"key": list(polys)}, geometry=list(polys.values()), crs=4326).to_crs(7856)
fp["osm_type"] = [k[0] for k in fp.key]; fp["osm_id"] = [k[1] for k in fp.key]
fp = fp.merge(b[["osm_type", "osm_id", "addr_housenumber", "addr_street", "addr_suburb", "levels", "h_p90_2020"]],
              on=["osm_type", "osm_id"], how="left", indicator=True)
fp["in_table"] = fp._merge == "both"

ex = [("GANSW703871190", "UNIT 1, 2 ARCHIBALD AVENUE WATERLOO 2017", -33.9007, 151.2068, "properties example"),
      ("GANSW720004116", "UNIT 106, 18 GADIGAL AVENUE, WATERLOO NSW 2017", -33.9026, 151.2131, "rent-history example")]
for gid, addr, lat, lng, src in ex:
    p = gpd.GeoSeries([Point(lng, lat)], crs=4326).to_crs(7856).iloc[0]
    fp["d"] = fp.distance(p)
    near = fp.nsmallest(4, "d")
    base = re.sub(r"^UNIT\s+\S+,?\s*", "", addr)
    print(f"\n{gid} | {addr} | {src}\n  building key from address: '{base}'")
    print(near[["osm_type", "osm_id", "d", "in_table", "addr_housenumber", "addr_street", "levels", "h_p90_2020"]].round(1).to_string(index=False))
print("\nlat/lng given to 4 dp -> grid step ~ 11 m N-S, ~9 m E-W")
