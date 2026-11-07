# Public data manifest (Microburbs rents hackathon prep)

Fetched 2026-11-07 00:39–00:48 AEDT (2026-11-06 13:39–13:48 UTC) by the unattended prep agent. Area: lat −33.925..−33.855,
lon 151.170..151.235 (water layer: bbox + 0.03° padding; LST: bbox + 0.01° padding). All fetch/build scripts are in
`public-data/_scripts/` and can be re-run (`python <script>`; `fetch_lst.py` and `build_features.py` need
`rents-heat/.venv/Scripts/python.exe`, where I installed geopandas, scipy, requests, pyarrow and tabulate).
Logs: `_manifest/log_*.txt`. Raw licence metadata from the data.nsw CKAN API: `_manifest/licence_records_datansw.json`.

## Licences (quoted text and where I read it)

| Source | Licence quote | Page read |
|---|---|---|
| NSW Digital Cadastre (Spatial Services, DCS) | data.nsw dataset "NSW Cadastre Web Service": licence **"Creative Commons Attribution 3.0"** (CKAN `license_url` https://creativecommons.org/licenses/by/3.0/au/), organisation "Spatial Services (DCS)". The older record "Spatial Services - NSW Cadastre" (author "Spatial Services (DFSI)") is listed as "Creative Commons Attribution". | https://data.nsw.gov.au/data/dataset/nsw-cadastre-web-service and CKAN API package_show (`nsw-cadastre-web-service`, `spatial-services-nsw-cadastre`) |
| OpenStreetMap | "OpenStreetMap is open data, licensed under the Open Data Commons Open Database License (ODbL)" … "Where you use OpenStreetMap data, you are required to do the following two things:" (credit "© OpenStreetMap contributors" and make clear the data is under ODbL). The Overpass responses also say: "The data included in this document is from www.openstreetmap.org. The data is made available under ODbL." | https://www.openstreetmap.org/copyright |
| Tree canopy 2022 per MMB (NSW DPHI) | data.nsw "Greater Sydney Region Tree Canopy to Modified Mesh Block 2022": licence **"Creative Commons Attribution"** (cc-by). | CKAN API package_show `urban-tree-canopy-cover-for-greater-sydney-2022` |
| Urban Heat Island 2016 per mesh block (NSW DCCEEW) | data.nsw "NSW Urban Heat Island to Modified Mesh Block 2016": licence **"Creative Commons Attribution"** (cc-by). | CKAN API package_show `nsw-urban-heat-island-to-modified-mesh-block-2016` |
| 2022 Heat Vulnerability Index and the LST summer 2022-23 GeoTIFF (NSW DPHI) | SEED record 08c39277-…: licence **"Creative Commons Attribution 4.0"** (https://creativecommons.org/licenses/by/4.0/); the data.nsw copy says "Creative Commons Attribution". | CKAN API package_show on datasets.seed.nsw.gov.au (`08c39277-8b70-4c4d-905e-5c931c2eef08`) and data.nsw (`2022-heat-vulnerability-index-for-the-greater-sydney-region`) |
| ABS Census 2021 (Data API) | "The Commonwealth owns the copyright in all material produced by the Australian Bureau of Statistics (ABS)." "All material presented on this website is provided under a Creative Commons Attribution 4.0 International licence". "Material obtained from this website is to be attributed to this department." | https://www.abs.gov.au/website-privacy-copyright-and-disclaimer |
| NSW rental bond data (already present; not re-downloaded) | See R4: the page shows only "Copyright"; an older data.nsw record says CC BY (unverified for current files). | — |

The ArcGIS REST service pages themselves have an empty `copyrightText`. The licences above come from the matching catalogue records.

## Files

| File | Source endpoint + query | Fetched (UTC) | Data date / version | Licence | Size | Notes |
|---|---|---|---|---|---|---|
| cadastre/sp_lots_bbox.geojson | https://maps.six.nsw.gov.au/arcgis/rest/services/public/NSW_Cadastre/MapServer/9/query; `where=planlabel LIKE 'SP%'`, envelope 151.170,-33.925,151.235,-33.855 inSR 4326, returnIdsOnly, then objectIds batches of 200, outSR 4326, f=geojson; fields cadid, planlabel, plannumber, lotnumber, sectionnumber, createdate, modifieddate, startdate, enddate, lastupdate, planlotarea, planlotareaunits, classsubtype, stratumlevel, hasstratum, itstitlestatus, lotidstring, urbanity | 2026-11-06 13:39 | live DCDB (lastupdate up to 2026) | CC BY 3.0 AU | 4.04 MB | 3,357 features = 3,357 distinct SP plans (one polygon per plan, classsubtype 3). `resultOffset` pagination returned HTTP 400 on this server, so I used objectIds batching. lotnumber and planlotarea are empty for SP polygons. DP lots were not fetched (not needed). |
| canopy/canopy2022_mmb_bbox.geojson | https://mapprod2.environment.nsw.gov.au/arcgis/rest/services/UHGC/Greater_Syd_Tree_Canopy_Cover_2022/MapServer/0/query; same envelope; fields MB_CODE16, MMB_CODE, MMB_CAT, MMB_TYPE, MMB_AREA, SA2_NAME16, AREA_TREE, PCT_TREE, Canopy_Percentage, All_Veg_Percentage, VERDATE | 2026-11-06 13:40 | canopy 2022; VERDATE 2021-11-25 (MMB geometry) | CC BY | 22.2 MB | 12,612 modified mesh blocks |
| heat/uhi2016_mb_bbox.geojson | …/UHGC/UHGC/MapServer/0/query (Urban Heat Island 2016); fields MB_CODE16, MB_CAT16, SA1_MAIN16, SA1_7DIG16, SA2_MAIN16, UHI_16_m, LGA | 2026-11-06 13:41 | summer 2015-16 | CC BY | 2.95 MB | 3,404 mesh blocks (2016). UHI_16_m is in °C above a vegetated reference. The service is in GDA94 (4283); the output was requested in 4326. |
| heat/hvi2022_sa1_bbox.geojson | …/UHGC/Heat_Vulnerability_Index_2022/MapServer/0/query; fields SA1_2021_AUST_GDA2020_SA1_CODE2, SA1_CODE21, SA2_CODE21, SA2_NAME21, LGA_NAME_2022, LST, Per_Tree, Per_Road, Population_Density, Heat_Exposure_Index, Heat_Vulnerability_Index, VERDATE | 2026-11-06 13:42 | HVI 2022 (VERDATE 2024-10-23) | CC BY 4.0 | 1.31 MB | 757 SA1s. Field SA1_CODE21 is 0 in the service, so use SA1_2021_AUST_GDA2020_SA1_CODE2. LST here is the SA1 mean. |
| heat/lst_sydney_2022-2023_summer_bbox.tif | https://datasets.seed.nsw.gov.au/dataset/08c39277-8b70-4c4d-905e-5c931c2eef08/resource/b081a8de-ca2e-4693-b961-706cd9f2c6ad/download/lst_sydney_2022-2023_summer.tif → 302 to a signed S3 URL; read with GDAL `/vsicurl/` range requests, bbox window ±0.01° only | 2026-11-06 13:41 | summer 2022-23 | CC BY 4.0 | 0.64 MB | The source is 5567×7430 float64, EPSG:4326, ~30 m pixels, 256² tiles (~309 MB). Only the window (315×334 px) was read; the full file was never downloaded. NaN = no data. |
| osm/osm_buildings_bbox.json | Overpass https://overpass-api.de/api/interpreter: `(way["building"](bbox);relation["building"](bbox);); out geom tags;` | 2026-11-06 13:39 | timestamp_osm_base 2026-11-06T13:39:05Z | ODbL 1.0 | 13.3 MB | 21,704 elements (21,635 ways, 69 relations). Relations came without member geometry, so I added the next file. |
| osm/osm_building_relations_bbox.json | Overpass: `relation["building"](bbox); out geom;` | 2026-11-06 13:44 | osm_base 2026-11-06T13:44:18Z | ODbL 1.0 | 0.14 MB | 69 multipolygon building relations with member geometry (68 polygonised) |
| osm/osm_stations_bbox.json | Overpass: `nwr["railway"~"^(station\|halt\|tram_stop)$"](bbox); nwr["public_transport"="station"](bbox); out center tags;` | 2026-11-06 13:40 | osm_base 2026-11-06T13:40:06Z | ODbL 1.0 | 0.03 MB | 75 elements (heavy rail, metro, light rail) |
| osm/osm_water_bbox.json | Overpass, bbox padded 0.03°: `way["natural"="coastline"]; nwr["natural"~"^(water\|bay)$"]; nwr["place"~"^(sea\|bay)$"]; way["waterway"="riverbank"]; out geom;` | 2026-11-06 13:45 | osm_base 2026-11-06T13:45:18Z | ODbL 1.0 | 2.13 MB | 402 elements. The first run used `out geom tags` (no relation members) and was re-run. Sydney Harbour / Port Jackson are water=harbour relations. |
| osm/osm_history_check.json | Overpass `out count;` for buildings now and with `[date:"2026-04-30T00:00:00Z"]` | 2026-11-06 13:42–13:43 | osm_base 13:42:07Z / 13:43:13Z | ODbL 1.0 | 1 KB | Historical snapshots work: 21,704 building elements now vs 20,482 at 2026-04-30 (+6%, mostly new mapping, not new buildings). The first attempt got a 502 and the retry worked. |
| osm/osm_buildings_tags_probe_2026-07-24.json | (pre-existing, not created by this run) | — | — | ODbL | 3.0 MB | left untouched |
| abs/c21_g02_sa2_median_rent_bbox.csv | ABS Data API https://data.api.abs.gov.au/rest/data/ABS,C21_G02_SA2,1.0.0/6.<34 SA2 codes>.SA2.1?format=csvfilewithlabels (MEDAVG 6 = Median rent $/weekly) | 2026-11-06 13:42 | Census 2021 | CC BY 4.0 | 23 KB | 34 SA2s intersecting the bbox (codes taken from the HVI SA1 layer) |
| abs/c21_g02_poa_median_rent_bbox.csv | same, C21_G02_POA, 21 postcodes (CoS postcodes + neighbours) | 2026-11-06 13:42 | Census 2021 | CC BY 4.0 | 8 KB | 20 POAs returned (2006 has no rent) |
| abs/sa1_2021_bbox_from_hvi.csv | derived from heat/hvi2022_sa1_bbox.geojson | — | ASGS 2021 | CC BY 4.0 | 59 KB | SA1→SA2 allocation and rough centroid for the 757 SA1s. **SA1-level median rent skipped**: the ABS Data API offers G02 only at SA2+, LGA, POA, SAL, CED, SED, SUA, UCL and RA. SA1 needs the GCP DataPack (a large zip, >100 MB for NSW). The mesh-block→SA1 link is available through UHI field SA1_MAIN16 (2016 ASGS). |
| _derived/buildings_features.parquet / .csv | `_scripts/build_features.py` (inputs above) | built 2026-11-06 13:47 | — | derived (ODbL for OSM-derived parts: share-alike applies to the database; CC BY attributions for NSW/ABS) | 0.54 / 1.78 MB | 4,399 residential buildings. Rule and columns: see `_derived/buildings_features_summary.md` |
| _derived/buildings_features_{coverage,corr_spearman,describe}.csv, buildings_features_summary.md | same script | — | — | derived | small | coverage %, Spearman correlations |
| _derived/sp_plans.csv | `_scripts/sp_year_calibration.py` | — | — | derived (CC BY 3.0 AU) | 0.38 MB | one row per SP plan: plannumber, createdate ISO, centroid, approximate area, bulk-date flag |
| _derived/sp_year_calibration.{json,png}, sp_year_lookup.csv | same | — | — | derived | small | SP number → year fit (see report) |
| bond/*.xlsx | pre-existing (R4) | — | 2025, Jan–Sep 2026 | see above | — | not re-downloaded |

The `__pycache__/` folder in `_scripts/` is a Python artefact and can be deleted.

## Pending: LiDAR order (ELVIS, 2026-11-07 ~02:15 AEDT)
- Ordered via https://elevation.fsdf.org.au/ (ICSM ELVIS), extent N −33.854 E 151.235 S −33.924 W 151.170; email link to the user's inbox.
- 80 datasets, ~5.25 GB: NSW Spatial Services 1 m DEM (Sydney201304 + Sydney202005, 40 GeoTIFF tiles) and LiDAR point clouds C3 AHD (Sydney201304 + Sydney202005, 40 LAZ tiles; 2020 tiles ~200 MB each). Tiles 2 km, eastings 330–336 km, northings 6244–6252 km (MGA56).
- Licence shown per file on ELVIS: Creative Commons Attribution (CC BY). Cost: none ("Data can be downloaded at no charge").
- Limits: May 2020 is the latest survey — buildings completed after mid-2020 have no height. Two epochs (2013, 2020) allow detecting buildings built between them.
- When the email arrives: download into public-data/lidar/ and record files here.
