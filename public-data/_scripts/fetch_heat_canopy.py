"""NSW DCCEEW Urban Heat & Green Cover (ArcGIS REST) for the bbox.
- Tree canopy 2022 per modified mesh block (MMB): UHGC/Greater_Syd_Tree_Canopy_Cover_2022/MapServer/0
- Urban heat island 2016 per mesh block: UHGC/UHGC/MapServer/0 (field UHI_16_m)
- Heat Vulnerability Index 2022 per SA1 (includes mean LST field): UHGC/Heat_Vulnerability_Index_2022/MapServer/0
Outputs in public-data/canopy/ and public-data/heat/.
"""
import time

from common import ROOT, arcgis_query_geojson, save_json

B = "https://mapprod2.environment.nsw.gov.au/arcgis/rest/services/UHGC"
JOBS = [
    (f"{B}/Greater_Syd_Tree_Canopy_Cover_2022/MapServer/0",
     "MB_CODE16,MMB_CODE,MMB_CAT,MMB_TYPE,MMB_AREA,SA2_NAME16,AREA_TREE,PCT_TREE,Canopy_Percentage,All_Veg_Percentage,VERDATE",
     ROOT / "canopy" / "canopy2022_mmb_bbox.geojson"),
    (f"{B}/UHGC/MapServer/0", "MB_CODE16,MB_CAT16,SA1_MAIN16,SA1_7DIG16,SA2_MAIN16,UHI_16_m,LGA",
     ROOT / "heat" / "uhi2016_mb_bbox.geojson"),
    (f"{B}/Heat_Vulnerability_Index_2022/MapServer/0",
     "SA1_2021_AUST_GDA2020_SA1_CODE2,SA1_CODE21,SA2_CODE21,SA2_NAME21,LGA_NAME_2022,LST,Per_Tree,Per_Road,Population_Density,Heat_Exposure_Index,"
     "Heat_Vulnerability_Index,VERDATE",
     ROOT / "heat" / "hvi2022_sa1_bbox.geojson"),
]

if __name__ == "__main__":
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else ""  # substring filter on output name, e.g. hvi
    for url, fields, out in JOBS:
        if only and only not in out.name:
            continue
        t0 = time.time()
        fc = arcgis_query_geojson(url, out_fields=fields, chunk=100)
        fc["query"] = dict(url=url + "/query", envelope_4326="151.170,-33.925,151.235,-33.855", out_fields=fields,
                           fetched=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        save_json(fc, out)
        print(f"{out.name}: {len(fc['features'])} features in {time.time()-t0:.0f}s", flush=True)
