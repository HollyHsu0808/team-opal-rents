# Team-Opal — What will this flat rent for? (Microburbs hackathon, Task 4)

**Would you let one flat vote ten times?** We estimate the weekly asking rent of all 65,777 City of Sydney flats, with and without a car space, from the flat's **own past rent** and from **its neighbours in the same building** — counting each neighbour once, and only trusting evidence that is still fresh.

| held-out check (window B: trained on adverts ≤ 31 May 2026, tested on 2,898 adverts dated 1 Jun – 31 Jul 2026) | median absolute error | within $100 | R² |
|---|---|---|---|
| NSW rental bond data, postcode × bedrooms (government baseline) | $100.0 | 51.3% | 0.50 |
| suburb × bedrooms × parking medians (simple estimate) | $97.2 | 52.5% | 0.59 |
| + the flat's own past rent × repeat-rent index | $56.1 | 68.7% | 0.77 |
| **ours** (+ same-building comps, one flat one vote, stale history out, floor-view difference, $5 grid) | **$45.0** | **77.1%** | **0.88** |

The organisers' reference points on the scored Aug–Sep adverts are $100 (simple estimate), $93 (with parking) and $86 (Microburbs rental valuation, 55% within $100) — a different set of adverts, so only indicative.

## Hand-in files
- `rents_team-opal.csv` — 65,777 rows: `gnaf_id, rent_with_car_space, rent_without_car_space` (weekly $, $0.50 steps; contract-checked)
- `report.html` — insights first, data, method, held-out results, date split, what worked and what did not, sources
- `pitch_slides.pdf` — the 8-slide story behind the pitch video
- `submission/frozen_rule.json` — the rule chosen on window A; `submission/baselines.md` — the baseline ladder on windows A and B
- `FEATURES.md` — every engineered feature (how it is built, what it did) and the LiDAR dataset (surveys, processing, per-floor view test)

## Who it helps
Tenants can check whether an asking rent is high for *this* floor of *this* building; landlords can set a rent that fits the flats around them rather than the suburb average; buyers and investors can compare likely rent before they buy.

## The clever step: deciding which evidence to trust
1. **Repeat-rent index.** Pairs of adverts of the same flat measure the market move (location and quality cancel out); held flat after July.
2. **Own history only when it is the same flat and still fresh.** The latest advert with the same bedroom count, moved to the cut-off with the index. A changed bedroom count multiplied the error by ~2.4; adverts older than 36 months were *worse than the neighbours' current rents* ($53 → $46.5 on window A), so they are replaced.
3. **Same-building comparables, one flat one vote.** Other flats in the building with the same bedrooms, each counted once however often it relisted, the target excluded, shrunk toward suburb medians when neighbours are few.
4. **Floor view inside the building.** NSW LiDAR line-of-sight for every floor (36 directions, 1.5 km): within the same building and bedroom count, a floor that sees water in every direction asks **+16–19%** more. We price only the *difference* between the flat and its comparable neighbours, with coefficients estimated within buildings. **It is the view, not the floor:** within the same building, the floor number alone is worth only about +1.2% per ten floors (≈ $1 a week per floor), and once water view is in the model it stays at +14–15% while the floor effect stays small. LiDAR views also cover the 48% of flats with no floor number (22,593 vs 15,516 flats with a usable value), and gave the higher R² on both windows.
5. **The asking-rent grid.** 84% of asking rents end in $0 and 52% are multiples of $50; predictions are snapped to $5 ($46.5 → $45 on window A).

**What did not work (and why):** building- or area-level layers — water view against suburb medians (+25%), strata age (−6% per decade), flight path (−2%), new build, heat, canopy — *raised* the error when added on top of the comparables, because the neighbours' rents already contain them. They stay in the report as insights. A ladder-only public adjustment and recency blends were also rejected on window A.

## Validation (never random)
- Window **A**: train ≤ 30 Apr 2026, test May → choose and freeze the rule (comps method, shrinkage, stale threshold, snapping, public layers).
- Window **B**: train ≤ 31 May, test Jun–Jul → evaluated with the frozen rule (the floor-view step was chosen on A after the base rule froze; both B results are in `report.html`).
- **Final**: train on everything ≤ 31 Jul 2026, predict all 65,777 flats.
- For every window the index, medians, parking premium and comparables are rebuilt from that window's training adverts only; the code asserts that no training row is dated after the cut-off. Nothing dated after 31 Jul 2026 is used anywhere (the API `rent-map` layer is dated after the cut-off and is not used).

## Data
- **Supplied (not in this repo):** `rents-input-flats-city-of-sydney.csv`, `rents-past-leases-city-of-sydney.csv`; Microburbs API `listings/rent-history` and `properties` for the 29 City of Sydney suburbs, `to_date=2026-07-31`.
- **Public (in `public-data/`, see `public-data/_manifest/MANIFEST.md` for endpoints, dates and licences):**
  - NSW Spatial Services LiDAR Sydney201304 + Sydney202005 via ELVIS (CC BY) → `public-data/_derived/lidar_buildings`, `lidar_floor_view` (29,970 building × floor rows), `buildings_features_v2` — see `public-data/_derived/DATA_DICTIONARY_lidar.md`
  - OpenStreetMap buildings, water, parks, stations, Sydney Airport runways (© OpenStreetMap contributors, ODbL)
  - NSW cadastre strata plans, suburb and LGA boundaries (NSW Spatial Services, CC BY)
  - NSW DPHI land surface temperature 2022-23, heat vulnerability 2022, tree canopy 2022 (CC BY 4.0); NSW DCCEEW urban heat island 2016 (CC BY)
  - ABS Census 2021 (CC BY 4.0)
  - NSW Fair Trading rental bond lodgements (baseline only; files not redistributed here — download from NSW Fair Trading)

## How to run
```bash
python -m venv rents-heat/.venv
rents-heat/.venv/Scripts/python -m pip install -r requirements-freeze.txt      # Windows; use rents-heat/.venv/bin/python elsewhere
rents-heat/.venv/Scripts/python run_everything.py --zip <organiser zip with the 3 CSVs> --key <team API key> --team team-opal
```
`run_everything.py` places the supplied files and key in `rents-heat/data_private/` (gitignored), pulls the API adverts up to 31 Jul 2026, runs window A (select + freeze), window B (held-out check), the final 65,777-flat CSV, the public-layer insight table and `report.html`, and copies the outputs to `submission/`. About 4–5 minutes on a laptop; peak memory ≈ 0.3 GB. Individual steps: `rents-heat/README.md`. Tests (synthetic data, no network): `rents-heat/.venv/Scripts/python -m pytest rents-heat/tests -q`.

The LiDAR rasters (1 m DEM/DSM, ~0.8 GB) are not needed to run: the per-building and per-floor tables in `public-data/_derived/` are included. Rebuild them with `public-data/_scripts/lidar_precompute.py` after ordering the two surveys from ELVIS.

## Licence
Code: MIT. Derived public-data tables keep the licences of their sources (OSM-derived tables under ODbL; NSW and ABS layers CC BY with attribution as above).
