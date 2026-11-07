# rents-heat — build spec (prepared 2026-11-06, used on 2026-11-07)

Microburbs hackathon, Task 4 "What will this flat rent for?". Tonight: public data + scripts on synthetic data. Tomorrow 11:00: swap in the real files and API, then validate.

## Official contract (from https://www.microburbs.com.au/p/hackathon-task-rents, read 2026-11-06)
- Input `rents-input-flats-city-of-sydney.csv`, 65,777 rows, columns:
  `gnaf_id, address, building_address, suburb, mb, latitude, longitude, property_type, bedrooms, bathrooms, car_space, car_spaces, floor_level, flats_in_building, block_new_homes_2021_2026, block_homes_2026, ever_advertised_for_lease, first_ever_listed, attrs_last_seen`
  - suburb looks like `Waterloo (NSW)`; 33 suburbs; mb = ABS mesh block code (e.g. 11205183500)
  - bedrooms 0–15; car_space 1/0/blank (5,793 blank); floor_level known for 34,396; ever_advertised_for_lease 0 for 16,094
  - first_ever_listed / attrs_last_seen are dates; `0001-01-01` = no known date
  - 4,246 rows absent from City address/block files: blank block columns, must still be predicted
- Output `rents_<team>.csv`: `gnaf_id, rent_with_car_space, rent_without_car_space`; all 65,777 IDs, no extras, no duplicates; weekly dollars, may end in .50; no blanks.
- Scoring: median absolute $ error on Aug–Sep 2026 adverts, column chosen by advert parking (else input parking); tie-break share within $100.
- report.html must: lead with insights; show held-out median abs error, share ≤$100 and R²; explain the date split; build only on earlier adverts, test on later adverts dated ≤ 31 Jul 2026; show data and method; credit every external source.
- Cut-off: nothing dated after 31 Jul 2026, from Microburbs or elsewhere.
- `rents-past-leases-city-of-sydney.csv` schema is NOT published. Assumed (map in `config/columns.json`):
  `gnaf_id, building_address, date, price, bedrooms, bathrooms, car_spaces` (one row per lease advert; price = weekly $). API shape for comparison: `GET /v1/properties/{gnaf_id}/rent-history` → `{"data":[{"date":"2025-07-10","price":850,"type":"For Rent","address":"Unit 506, 20 Dunkerley Place"}]}`.
- `blocks-homes-and-people-2021-2026.csv`: schema unknown; not needed by the core pipeline.
- Repo must never contain: API key, supplied files, API output. All of those live in `data_private/` (gitignored).

## Method (v4, see research/docs/.../T/v4-topics.md)
- Windows on advert date (official example report used ≤30 Apr 2026 train / 1 May–31 Jul 2026 test). Ours, all configurable in `config/windows.json`:
  - A: train ≤ 2026-04-30, test 2026-05-01..2026-05-31 (select rules)
  - B: train ≤ 2026-05-31, test 2026-06-01..2026-07-31 (evaluate once, frozen)
  - final: train ≤ 2026-07-31, predict all 65,777
- Floor estimate per flat and parking variant:
  1. own history: latest own advert ≤ train cut-off × (suburb rent index now / index at that advert month); index rebuilt from repeat pairs in training adverts only.
  2. no own history: median of comparable group (suburb, bedrooms, parking) with ≥10 adverts → (suburb, bedrooms) → (bedrooms) — the official example's ladder.
  - parking premium: median with-vs-without gap per (suburb, bedrooms) group, fallback citywide; used to produce the other column.
- One correction (no-own-history rows), per round 6: "one flat, one vote" same-building same-bedroom comps — collapse each neighbouring flat's adverts to one date-eligible, index-adjusted signal; exclude the target flat; adjust parking; shrink toward the floor by the number of DISTINCT flats. A compares floor vs plain same-building comps vs one-vote comps (overall MedAE must improve and within-$100 must not drop; report affected flats/buildings); freeze; B once. No stacking of recency / SP-wing / stack comps / shift calibration.
- Public data: LST + canopy map for report and pitch; optional pre-specified A ablation, else "incremental value untested". SP table diagnostic only. Bond only if licence and publish date pass.
- Metrics per slice (own-history / no-history / all): n, median abs error, share ≤$100, R², coverage.

## Public feature contract (produced by src/publicdata, consumed by src/rents)
`join_heat.py --in points.csv --out features.csv` where points.csv has `gnaf_id, latitude, longitude`. Output columns (one row per input id, NaN when no coverage):
- `gnaf_id`
- `lst_point_c` — summer 2022-23 land surface temperature at the point (°C)
- `lst_100m_c` — mean LST within 100 m of the point (°C)
- `canopy_mmb_pct` — tree canopy % of the containing 2022 modified mesh block
- `uhi16_mmb_c` — 2016 urban heat island value of the containing modified mesh block
- `mmb_id_2016`, `mmb_id_2022` — containing modified mesh block ids (diagnostic)
Sources (all CC BY, credited in ATTRIBUTION.md): NSW DPHI LST Greater Sydney summer 2022-23; NSW DPHI tree canopy to modified mesh block 2022 (GDA2020); NSW DCCEEW urban heat island to modified mesh block 2016.

## Layout and ownership
- `src/publicdata/`, `data_public/`, `ATTRIBUTION.md`, `requirements-geo.txt`, `.venv/` — public-data owner
- `src/rents/`, `synthetic/`, `tests/`, `config/`, `run_all.py`, `requirements.txt` — pipeline owner
- `SPEC.md`, `README.md`, `.gitignore` — integrator
