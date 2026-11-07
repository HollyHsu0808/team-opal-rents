# Data dictionary — LiDAR, view and aircraft features (public data only)

Built by `public-data/_scripts/lidar_precompute.py` on 2026-11-07. Join key to the existing table: `osm_type` + `osm_id` (one row per OSM residential building in `buildings_features.csv`, 4,399 buildings). On the day, attach flats to buildings by point-in-footprint or nearest building centroid (lat/lng from the input file).

Sources (credit in README): LiDAR point clouds + 1 m DEM — © State of New South Wales (Spatial Services, DCS NSW), via ELVIS (ICSM), CC BY; surveys `Sydney201304` (April 2013) and `Sydney202005` (May 2020). OSM footprints, water, parks, runways — © OpenStreetMap contributors, ODbL.

Date rule: both surveys predate every training window (A/B/final), so all LiDAR features are allowed. Buildings completed after May 2020 have no 2020 height (they show as near-zero or ground); flag them via `new_2013_2020` = False and low `h_p90_2020` while OSM `levels` is high.

## Files
| file | rows | content |
|---|---|---|
| `lidar_buildings.csv/.parquet` | 1 per building | ground level, heights 2013 and 2020, new-build flag |
| `lidar_floor_view.csv/.parquet` | 1 per building × floor | eye height, roofline margin, water / park view |
| `aircraft_proxy.csv/.parquet` | 1 per building | distance to Sydney Airport runway centrelines |
| `buildings_features_v2.csv/.parquet` | 1 per building | `buildings_features.csv` + all of the above (building-level summaries of the floor table) |
| `lidar_rasters/*.tif` | 1 m grids (MGA zone 56) | `dem_*` bare earth, `dsm_*` highest return (buildings + trees), `ndsm_*` = DSM − DEM, `bldmax_*` highest building-class return |

## Columns
**lidar_buildings**
- `ground_z_{2013,2020}` — median DEM (m AHD) inside the footprint.
- `h_p90_{year}` — 90th percentile of DSM − DEM inside the footprint (m) ≈ roof height; `h_max_{year}` maximum; `h_bldclass_p90_{year}` using building-classified points only.
- `n_px_{year}` — footprint pixels (≈ m²) used.
- `new_2013_2020` — True when `h_p90_2013` < 3 m and `h_p90_2020` ≥ 9 m (built between the surveys).
- `height_change_2013_2020_m` — `h_p90_2020 − h_p90_2013`.
- `levels_from_lidar_2020` — `h_p90_2020` / 3.1 m, rounded (storey height is an assumption).
- `footprint_src` — `osm` footprint polygon, or `box16m` (16 m square around the centroid when OSM had no way geometry).

**lidar_floor_view** (floor 0 = ground floor; eye = ground + 3.1 m × floor + 1.5 m)
- `eye_z_ahd` — eye height (m AHD).
- `roof_max_100m_ahd` — highest DSM within 100 m (beyond the own footprint) along 36 rays; `roof_margin_m` = eye − that (> 0 means the floor clears every roof/tree within 100 m).
- `water_view_frac` — share of 36 directions (10° each) in which some water within 1.5 km is in line of sight; `water_nearest_visible_m` nearest visible water distance.
- `park_view_frac`, `park_nearest_visible_m` — same for OSM parks/gardens/golf courses/reserves.
- Line of sight uses the 2020 DSM (trees and buildings block), 5 m steps, eye placed at the footprint edge. Orientation of the actual unit is unknown, so these are upper bounds for the floor ("could a flat on this floor see water in some direction").

**aircraft_proxy** (Sydney Airport runways from OSM; 16/34 axes oriented north, i.e. toward the City of Sydney)
- `rwy_{16R_34L,16L_34R,07_25}_along_m` — distance along the extended centreline from the northern (or eastern for 07/25) runway end; positive = beyond the runway on the City side.
- `rwy_*_lateral_m` — perpendicular distance from the extended centreline.
- `rwy_16_34_min_lateral_m`, `rwy_16_34_along_m` — nearest of the two parallel runways.
- `under_16_34_path_1km` — within 1 km laterally and 0–15 km along a 16/34 path (arrivals/departures over the inner west). A geometric proxy, not a noise measurement; compare with Microburbs `shapes/noise-heatmap` and `risks/property-hazard-counts` on the day.

**buildings_features_v2 extra summaries**
- `first_floor_water_view` — lowest floor with any water view (NaN = none).
- `first_floor_above_roofline` — lowest floor with `roof_margin_m` > 0.
- `top_floor_water_view_frac`, `top_floor_park_view_frac`, `top_floor_roof_margin_m`.

## How to use on the day
- For a flat with known `floor_level`: join building → pick that floor row in `lidar_floor_view` → features `roof_margin_m`, `water_view_frac`, `park_view_frac`.
- Unknown floor: use building-level summaries, or the building median floor.
- These vary within a building, so they are the only public features that same-building comps cannot absorb — test them on window A for no-history rows and as a floor adjustment to building comps.
