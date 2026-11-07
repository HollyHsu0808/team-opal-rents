"""OpenStreetMap via Overpass API (ODbL 1.0, (c) OpenStreetMap contributors).
Outputs in public-data/osm/:
  osm_buildings_bbox.json    buildings (ways + relations) with geometry and tags
  osm_stations_bbox.json     railway=station|halt|tram_stop|subway_entrance? (no), public_transport=station
  osm_water_bbox.json        natural=water|coastline|bay, place=sea|bay, water=*, waterway=riverbank (bbox extended)
  osm_history_check.json     building counts at [date:"2026-04-30T00:00:00Z"] vs now
"""
import json
import time

import requests

from common import BBOX, ROOT, UA, save_json

ENDPOINTS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
bb = f"{BBOX['s']},{BBOX['w']},{BBOX['n']},{BBOX['e']}"
# water: extend bbox by ~3 km so harbour / bay polygons and coastline segments near the edge are present
bbw = f"{BBOX['s']-0.03},{BBOX['w']-0.03},{BBOX['n']+0.03},{BBOX['e']+0.03}"

Q = {
    "buildings": f"""[out:json][timeout:300];
(way["building"]({bb});relation["building"]({bb}););
out geom tags;""",
    "stations": f"""[out:json][timeout:120];
(nwr["railway"~"^(station|halt|tram_stop)$"]({bb});nwr["public_transport"="station"]({bb}););
out center tags;""",
    "water": f"""[out:json][timeout:300];
(way["natural"="coastline"]({bbw});
 nwr["natural"~"^(water|bay)$"]({bbw});
 nwr["place"~"^(sea|bay)$"]({bbw});
 way["waterway"="riverbank"]({bbw}););
out geom;""",
    "history_now": f"""[out:json][timeout:120];
(way["building"]({bb});relation["building"]({bb}););
out count;""",
    "history_2026_04_30": f"""[out:json][timeout:120][date:"2026-04-30T00:00:00Z"];
(way["building"]({bb});relation["building"]({bb}););
out count;""",
}


def run(q):
    last = None
    for ep in ENDPOINTS:
        for k in range(3):
            try:
                r = requests.post(ep, data={"data": q}, headers={"User-Agent": UA}, timeout=400)
                if r.status_code in (429, 504):
                    print(f"  {ep} {r.status_code}, sleeping", flush=True)
                    time.sleep(30 * (k + 1))
                    continue
                r.raise_for_status()
                d = r.json()
                d["_endpoint"] = ep
                return d
            except Exception as ex:  # noqa: BLE001
                last = ex
                print(f"  {ep} failed: {ex}", flush=True)
                time.sleep(15)
    raise RuntimeError(last)


if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or ["buildings", "stations", "water", "history_now", "history_2026_04_30"]
    out = {}
    for name in names:
        t0 = time.time()
        d = run(Q[name])
        d["_query"] = Q[name]
        d["_fetched"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        ts = d.get("osm3s", {}).get("timestamp_osm_base")
        print(f"{name}: {len(d.get('elements', []))} elements, osm_base={ts}, {time.time()-t0:.0f}s", flush=True)
        if name.startswith("history"):
            out[name] = dict(elements=d.get("elements"), osm3s=d.get("osm3s"), endpoint=d["_endpoint"],
                             query=Q[name], fetched=d["_fetched"])
        else:
            save_json(d, ROOT / "osm" / f"osm_{name}_bbox.json")
        time.sleep(10)
    if out:
        save_json(out, ROOT / "osm" / "osm_history_check.json")
    print(json.dumps({k: v["elements"] for k, v in out.items()}, indent=1))
