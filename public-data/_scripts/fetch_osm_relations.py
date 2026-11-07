"""Building multipolygon relations with member geometry (the main buildings query used `out geom tags;`,
which returns relation tags/bounds but no member geometry). Output: public-data/osm/osm_building_relations_bbox.json"""
import time

from common import ROOT, save_json
from fetch_osm import bb, run

Q = f"""[out:json][timeout:180];
relation["building"]({bb});
out geom;"""

if __name__ == "__main__":
    d = run(Q)
    d["_query"] = Q
    d["_fetched"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print(len(d["elements"]), "relations, osm_base", d.get("osm3s", {}).get("timestamp_osm_base"))
    save_json(d, ROOT / "osm" / "osm_building_relations_bbox.json")
