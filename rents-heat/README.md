# rents-heat

Microburbs hackathon Task 4, "What will this flat rent for?". The contract and method are in `SPEC.md`.

## Layout
- `config/`: `columns.json` (input, past-leases and API column maps), `windows.json` (A / B / final), `model.json`, `paths.json`
- `src/rents/load.py`: loaders, building-key normaliser, window slicing (hard leakage asserts)
- `src/rents/api_fetch.py`: API client for suburb rent-history and properties (paging, about 9 req/s, 429 backoff, 403 recorded)
- `src/rents/model.py`: floor (own last advert × repeat-pair suburb index; ladder suburb×bed×parking → suburb×bed → bed), parking premium, "one flat, one vote" building comps, metrics
- `src/rents/public_features.py`: flat → OSM building join, LiDAR floor views, windowed nDSM, aircraft proxy (optional window-A ablation)
- `synthetic/make_synthetic.py`: synthetic files shaped like the official ones
- `run_all.py`: runs the chain. `tests/`: pytest, no network

Everything private (organiser files, API output, key, outputs) lives in `data_private/`, which is gitignored.

## Tonight (synthetic)
```
.venv/Scripts/python.exe synthetic/make_synthetic.py --n-flats 65777
.venv/Scripts/python.exe run_all.py --data synthetic --window A
.venv/Scripts/python.exe run_all.py --data synthetic --window B
.venv/Scripts/python.exe run_all.py --data synthetic --window final --team synth
.venv/Scripts/python.exe -m pytest tests -q
```
Synthetic outputs go to `data_private/out/synthetic/`.

## Day of (real data, 11:00)
1. Copy the supplied files to `data_private/real/`, keeping the names `rents-input-flats-city-of-sydney.csv` and `rents-past-leases-city-of-sydney.csv`. If the headers differ, edit the right-hand side of `config/columns.json`.
2. Put the key in `data_private/.env` as `MICROBURBS_API_KEY=...`. Do not paste it into a shell command or commit it.
3. Fetch the API data (to_date=2026-07-31 is applied by the server and checked again in the client):
   ```
   .venv/Scripts/python.exe src/rents/api_fetch.py --dry-run
   .venv/Scripts/python.exe src/rents/api_fetch.py --limit-suburbs 1
   .venv/Scripts/python.exe src/rents/api_fetch.py
   ```
4. Run the windows in order. A selects and freezes the rule (best of plain / one-vote k grid × stale-own-history
   handling, then snapping; accepted only if it beats the floor on MedAE without losing share<=$100), B evaluates it
   once, final writes the submission; then the insight table and the report:
   ```
   .venv/Scripts/python.exe run_all.py --data real --window A --ablation
   .venv/Scripts/python.exe run_all.py --data real --window B
   .venv/Scripts/python.exe run_all.py --data real --window final --team <TEAM>
   .venv/Scripts/python.exe src/rents/public_insights.py --data real --window B
   .venv/Scripts/python.exe src/rents/make_report.py --data real --team <TEAM>     -> data_private/out/report.html
   ```
   Before the official input file arrives, the same chain runs on the API pull alone with `--data apiproxy`
   (`src/rents/build_api_proxy_input.py` first); its outputs live in `data_private/out/apiproxy/` and are not a submission.
   `--window A --ablation` also fits the public-data adjustment (LiDAR floor view, roofline, height, flight path,
   new build, strata age) on A training rows and tests it on A; it is written into `frozen_rule.json` with
   `apply: true` only if no-history MedAE improves, overall MedAE does not worsen and share<=$100 does not drop.
   B and final then apply it automatically (printed as "public adjustment applied").
   Report insight (does the market price the public layers?): `src/rents/public_insights.py --data real --window B`
   writes `data_private/out/public_insights_B.csv/.md` (residual by water view / roofline / flight path / new build /
   height / heat / canopy / strata age, plus the largest within-building rent spreads).
   12:00 fallback (no complete A run yet): `run_all.py --data real --window final --team <TEAM> --force-floor` writes a
   floor-only submission without window-A selection (also works for `--window B`). Rerun `--window A` later to replace it.
   Adverts come from the past-leases file when it exists, otherwise from the API parquet (`--adverts api|both` overrides this). Real outputs go to `data_private/out/` as `metrics_<window>.json`, `frozen_rule.json` and `rents_<TEAM>.csv`. The final run fails if the contract check fails.
