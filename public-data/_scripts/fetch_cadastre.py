"""NSW Digital Cadastre (Spatial Services) - strata plan (SP) lots in the bbox.
Source: https://maps.six.nsw.gov.au/arcgis/rest/services/public/NSW_Cadastre/MapServer/9 (layer "Lot")
Output: public-data/cadastre/sp_lots_bbox.geojson
"""
import time

from common import ROOT, arcgis_query_geojson, save_json

URL = "https://maps.six.nsw.gov.au/arcgis/rest/services/public/NSW_Cadastre/MapServer/9"
FIELDS = ("cadid,planlabel,plannumber,lotnumber,sectionnumber,createdate,modifieddate,startdate,enddate,lastupdate,"
          "planlotarea,planlotareaunits,classsubtype,stratumlevel,hasstratum,itstitlestatus,lotidstring,urbanity")

if __name__ == "__main__":
    t0 = time.time()
    fc = arcgis_query_geojson(URL, where="planlabel LIKE 'SP%'", out_fields=FIELDS)
    fc["query"] = dict(url=URL + "/query", where="planlabel LIKE 'SP%'", envelope_4326="151.170,-33.925,151.235,-33.855",
                       fetched=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    save_json(fc, ROOT / "cadastre" / "sp_lots_bbox.geojson")
    print(f"done {len(fc['features'])} features in {time.time()-t0:.0f}s")
