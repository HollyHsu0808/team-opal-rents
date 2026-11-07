"""Shared helpers for the public-data fetch scripts (Microburbs rents hackathon prep).

All scripts are re-runnable: python public-data/_scripts/<script>.py
Area: City of Sydney LGA + neighbours, lat -33.925..-33.855, lon 151.170..151.235.
"""
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]          # public-data/
BBOX = dict(s=-33.925, w=151.170, n=-33.855, e=151.235)
ENVELOPE = f"{BBOX['w']},{BBOX['s']},{BBOX['e']},{BBOX['n']}"
UA = "MicroburbsHackathonPrep/1.0 (research; contact via hackathon team)"
S = requests.Session()
S.headers["User-Agent"] = UA


def get_json(url, params=None, method="GET", retries=4, timeout=180):
    for k in range(retries):
        try:
            if method == "GET":
                r = S.get(url, params=params, timeout=timeout)
            else:
                r = S.post(url, data=params, timeout=timeout)
            r.raise_for_status()
            d = r.json()
            if isinstance(d, dict) and "error" in d:
                raise RuntimeError(f"ArcGIS error {d['error']}")
            return d
        except Exception as ex:  # noqa: BLE001
            if k == retries - 1:
                raise
            print(f"  retry {k+1} after {ex}", flush=True)
            time.sleep(5 * (k + 1))


def arcgis_query_geojson(layer_url, where="1=1", out_fields="*", envelope=ENVELOPE, chunk=200, extra=None):
    """Fetch all features intersecting the envelope via returnIdsOnly + objectIds batches (works even when
    resultOffset pagination fails). Returns a GeoJSON FeatureCollection dict in EPSG:4326."""
    base = dict(where=where, geometry=envelope, geometryType="esriGeometryEnvelope", inSR=4326,
                spatialRel="esriSpatialRelIntersects", f="json")
    ids = get_json(layer_url + "/query", dict(base, returnIdsOnly="true"))
    oids = sorted(ids.get("objectIds") or [])
    oid_field = ids.get("objectIdFieldName", "OBJECTID")
    print(f"  {len(oids)} ids from {layer_url}", flush=True)
    feats = []
    for i in range(0, len(oids), chunk):
        p = dict(objectIds=",".join(map(str, oids[i:i + chunk])), outFields=out_fields, returnGeometry="true",
                 outSR=4326, f="geojson")
        if extra:
            p.update(extra)
        d = get_json(layer_url + "/query", p, method="POST")
        feats.extend(d.get("features", []))
        time.sleep(0.3)
    return dict(type="FeatureCollection", features=feats, oid_field=oid_field)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    print(f"  saved {path} ({path.stat().st_size/1e6:.2f} MB)", flush=True)
    return path
