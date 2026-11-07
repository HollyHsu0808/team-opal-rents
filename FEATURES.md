# Feature engineering and the LiDAR dataset

## The LiDAR dataset

Two free government airborne laser surveys of the City of Sydney (April 2013 and May 2020) give a 1 m height surface of every roof and tree, which we turn into building heights and a view test for every floor.

| item | value |
|---|---|
| source | NSW Spatial Services (DCS), ordered through the ELVIS portal; licence CC BY 4.0, no charge |
| surveys | Sydney201304 (Apr 2013, GDA94 / MGA 56) and Sydney202005 (May 2020, GDA2020 / MGA 56) |
| files | 80 files, 5.6 GB: per survey 20 point-cloud tiles (LAZ, 2 km × 2 km) + 20 bare-earth 1 m DEM tiles (GeoTIFF) |
| density | 2013: 164 million points in total (~2 points/m²); 2020: about 50 million points per tile |
| coverage | 100% of the 33 input suburbs (checked against NSW suburb boundaries) |
| outputs in this repo | `public-data/_derived/lidar_buildings` (4,399 buildings), `lidar_floor_view` (29,970 building × floor rows), `aircraft_proxy`, `buildings_features_v2` — CSV + Parquet; columns in `DATA_DICTIONARY_lidar.md` |

From points to features (`public-data/_scripts/lidar_precompute.py`, `lidar_views_v2.py`):

1. **Height surfaces (1 m grid).** DEM = bare earth (supplied). DSM = highest return in each cell, noise classes 7 and 18 removed. nDSM = DSM − DEM = height above ground.
2. **Building height.** For each of 4,399 residential OpenStreetMap footprints: roof height = 90th percentile of nDSM inside the footprint (ignores antennas). Levels = height ÷ 3.1 m. Check: 99.9% of buildings measured; LiDAR levels match OSM levels within 1 floor (median), correlation 0.72.
3. **New builds.** Height below 3 m in 2013 and at least 9 m in 2020 → built between the surveys: 79 buildings.
4. **View per floor.** Eye height = ground + 3.1 m × floor + 1.5 m. From the footprint edge, 36 directions (every 10°), a sample every 5 m out to 1.5 km. A sample is visible when its angle above the eye is higher than every sample before it. Water = OSM water plus flat cells at sea level (the harbour); parks = OSM parks. The building's own roof never blocks.
5. **Three features per floor.** `water_view_frac` and `park_view_frac` = share of the 36 directions with water / park in sight; `roofline_clear_frac` = share of directions where the floor is above every roof and tree within 100 m.
6. **Join to flats.** Flats are grouped by `building_address` + suburb; the group's mean point is matched to the nearest footprint within 15 m (if two are close, the one whose height best fits the highest known floor). Flats with `floor_level` (52%) take that floor's row; the rest take the building's median over floors.

Limits: the unit's orientation is unknown, so a floor's view is an upper bound for any flat on it; buildings finished after May 2020 have no 2020 height; 3.1 m per storey is an assumption.

**What it showed:** within the same building and bedroom count (fixed effects, 22,593 flats in 2,901 groups), a floor that sees water in every direction asks +16.3% (window A), +16.8% (B), +18.7% (final training) more than one that sees none; park +4.5–6.2%; clear of every roof within 100 m +4.8–5.3%. The floor number alone adds about +1.2% per ten floors. It is the view, not the floor.

## Feature engineering

Every feature is rebuilt for each validation window from adverts dated on or before that window's cut-off. In the order the estimate is built (code: `rents-heat/src/rents/model.py`, `public_within.py`):

| # | feature | built from | how | effect / status |
|---|---|---|---|---|
| 1 | advert table | past-leases file + API rent-history | 153,040 adverts after de-duplicating on (flat, date, price); $80–$20,000; nothing after 31 Jul 2026 | used |
| 2 | building key | input `building_address` + suburb | upper case, commas and NSW removed, street types shortened (AVENUE → AVE), suburb appended: 3,671 buildings | used |
| 3 | parking flag | `car_space`, else `car_spaces` > 0; advert parking first | 1 / 0 / unknown | used |
| 4 | repeat-rent index | pairs of adverts of the same flat | adverts within 60 days merged into one letting; monthly log index solved from all pairs with smoothing; suburb deviation shrunk to zero | used |
| 5 | rent moved to today | 1 + 4 | log rent + index(cut-off) − index(advert month) | used |
| 6 | parking premium | 5, last 12 months | median gap with vs without parking: suburb × bedrooms (≥10 each side) → bedrooms → city; clipped to 0–35%; median 8% | used |
| 7 | suburb ladder | 5, last 12 months | median of suburb × bedrooms × parking (≥10) → suburb × bedrooms → bedrooms → city | used |
| 8 | own history | 5 | latest advert with the same bedroom count; its age in months | used |
| 9 | stale flag | 8 | own advert older than 36 months → treated as no history | used: $53 → $46.5 on window A |
| 10 | one-vote comparables | 2 + 5 + 6 | same building and bedrooms, last 36 months, one latest advert per neighbour flat, target excluded, parking removed; weight n ÷ (n + 0.5) against the ladder | used: no-history $99 → $60 on window B |
| 11 | floor-view difference | LiDAR features + 10 | target's view minus the mean view of its comparables, priced with within-building coefficients | used: R² 0.877 → 0.880 on window B |
| 12 | price grid | output | round to the nearest $5, then $0.50 | used: $46.5 → $45 on window A |

Tested on window A and dropped: plain comparables (every advert counts), other shrinkage values, stale thresholds of 24 or 60 months, a 50% stale blend, $10 rounding, the cross-sectional public adjustment (water, park, height, flight path, new build, strata age — $45 → $50, because the neighbours' rents already contain them), the same adjustment on ladder-only flats, and the floor number in place of LiDAR. Kept out on purpose: `first_ever_listed` and `attrs_last_seen` (may be dated after the cut-off), the API `leased` flag (today's status, not the advert's) and `rent-map` (dated after the cut-off).
