# Validation of every rule, and comparison with regression and ML

## Our AVM vs regression vs gradient-boosted trees (window B: same 2,898 held-out Jun-Jul 2026 adverts)

Learned models are trained leak-free: features built at origins 2025-09-30, 2025-11-30, 2026-01-31, 2026-03-31 from adverts on or before each origin, labels = the next two months (9,747 rows); then applied with features built at 2026-05-31. Code: `rents-heat/src/rents/ml_benchmark.py`.

| model | median abs error | within $100 | R² |
|---|---|---|---|
| R1 hedonic regression (no history) | $82.8 | 58.2% | 0.714 |
| R2 regression on our engineered evidence | $68.8 | 65.8% | 0.784 |
| ML gradient-boosted trees (L1) on all features | $46.1 | 77.6% | 0.819 |
| OURS evidence-weighting AVM (submitted rule) | $45.0 | 77.1% | 0.880 |

Reading: a regression cannot exploit per-flat evidence; gradient-boosted trees catch up only when fed our engineered evidence (own history, one-vote building comps). Our evidence-weighting AVM matches the trees on accuracy, has the best R², and every estimate can name its evidence.

## Walk-forward validation of every rule (code: `rents-heat/src/rents/validate_rules.py`)


Train on adverts <= origin, test on the next two months. 'better' = the submitted setting has lower MedAE (tie: higher share within $100) on that origin.


## Structural rules (6 origins, 2025-03 .. 2026-05)

| variant | origins | mean MedAE | mean within $100 | mean R² | submitted better on | submitted worse on | verdict |
|---|---|---|---|---|---|---|---|
| submitted rule | 6 | $50.00 | 75.96% | 0.820 | 0 | 0 | reference |
| floor only (no comps, no stale rule) | 6 | $62.50 | 66.68% | 0.726 | 6 | 0 | submitted setting supported |
| no building comps | 6 | $57.50 | 69.38% | 0.753 | 6 | 0 | submitted setting supported |
| plain comps instead of one vote | 6 | $50.00 | 75.64% | 0.821 | 6 | 0 | submitted setting supported |
| no stale rule | 6 | $55.83 | 70.41% | 0.776 | 6 | 0 | submitted setting supported |
| stale threshold 24 months | 6 | $48.33 | 76.09% | 0.819 | 2 | 3 | ALTERNATIVE BETTER - review |
| stale threshold 60 months | 6 | $55.00 | 72.57% | 0.802 | 6 | 0 | submitted setting supported |
| stale blend 50% instead of replace | 6 | $50.83 | 74.46% | 0.818 | 6 | 0 | submitted setting supported |
| one-vote k = 2 | 6 | $50.00 | 75.63% | 0.815 | 6 | 0 | submitted setting supported |
| one-vote k = 5 | 6 | $51.08 | 74.77% | 0.804 | 6 | 0 | submitted setting supported |
| no LiDAR floor-view (P1) | 6 | $50.00 | 75.88% | 0.819 | 3 | 2 | submitted setting supported |
| no snapping | 6 | $49.53 | 75.22% | 0.820 | 2 | 4 | ALTERNATIVE BETTER - review |
| snap to $10 | 6 | $50.00 | 76.56% | 0.820 | 1 | 5 | mixed |
| extrapolate 6-month trend after cut-off | 6 | $51.67 | 75.60% | 0.820 | 4 | 2 | submitted setting supported |

## Fixed constants, group 1 (4 origins)

| variant | origins | mean MedAE | mean within $100 | mean R² | submitted better on | submitted worse on | verdict |
|---|---|---|---|---|---|---|---|
| base | 4 | $48.75 | 75.92% | 0.831 | 0 | 0 | reference |
| episode_gap_days = 30 | 4 | $48.75 | 75.92% | 0.831 | 2 | 1 | submitted setting supported |
| episode_gap_days = 90 | 4 | $48.75 | 75.95% | 0.831 | 1 | 2 | mixed |
| ladder_min_n = 5 | 4 | $48.75 | 75.87% | 0.832 | 2 | 1 | submitted setting supported |
| ladder_min_n = 20 | 4 | $48.75 | 75.88% | 0.826 | 3 | 0 | submitted setting supported |
| ladder_lookback_months = 6 | 4 | $48.75 | 75.83% | 0.828 | 2 | 1 | submitted setting supported |
| ladder_lookback_months = 24 | 4 | $48.75 | 76.10% | 0.832 | 0 | 4 | mixed |

## Fixed constants, group 2 (4 origins)

| variant | origins | mean MedAE | mean within $100 | mean R² | submitted better on | submitted worse on | verdict |
|---|---|---|---|---|---|---|---|
| base | 4 | $48.75 | 75.92% | 0.831 | 0 | 0 | reference |
| comps_lookback_months = 24 | 4 | $48.75 | 75.78% | 0.830 | 3 | 0 | submitted setting supported |
| comps_lookback_months = 60 | 4 | $48.75 | 75.13% | 0.829 | 3 | 1 | submitted setting supported |
| premium_min_n = 5 | 4 | $48.75 | 75.86% | 0.831 | 2 | 0 | submitted setting supported |
| premium_min_n = 20 | 4 | $48.75 | 75.89% | 0.831 | 3 | 1 | submitted setting supported |
| premium_clip = [0.0, 0.25] | 4 | $48.75 | 75.97% | 0.831 | 1 | 3 | mixed |
| premium_clip = [0.0, 0.5] | 4 | $48.75 | 75.92% | 0.831 | 0 | 0 | submitted setting supported |

## Fixed constants, group 3 (4 origins)

| variant | origins | mean MedAE | mean within $100 | mean R² | submitted better on | submitted worse on | verdict |
|---|---|---|---|---|---|---|---|
| base | 4 | $48.75 | 75.92% | 0.831 | 0 | 0 | reference |
| index_smooth = 1.0 | 4 | $48.75 | 75.95% | 0.831 | 1 | 3 | mixed |
| index_smooth = 10.0 | 4 | $48.75 | 75.97% | 0.831 | 1 | 2 | mixed |
| suburb_index_ridge = 0.3 | 4 | $48.75 | 75.87% | 0.830 | 3 | 1 | submitted setting supported |
| suburb_index_ridge = 3.0 | 4 | $47.50 | 76.32% | 0.832 | 0 | 4 | ALTERNATIVE BETTER - review |
| bed_cap = 3 | 4 | $48.75 | 75.90% | 0.827 | 2 | 1 | submitted setting supported |
| bed_cap = 5 | 4 | $48.75 | 75.92% | 0.817 | 1 | 0 | submitted setting supported |

## Fixed constants, group 4 (4 origins)

| variant | origins | mean MedAE | mean within $100 | mean R² | submitted better on | submitted worse on | verdict |
|---|---|---|---|---|---|---|---|
| base | 4 | $48.75 | 75.92% | 0.831 | 0 | 0 | reference |
| own_same_bed = False | 4 | $48.75 | 75.91% | 0.827 | 3 | 1 | submitted setting supported |
| price_bounds = [150, 5000] | 4 | $47.50 | 76.09% | 0.874 | 1 | 3 | ALTERNATIVE BETTER - review |

## When each rule was fixed
Base constants committed 2026-11-07 09:12-09:19, before the API key (11:40) and the official files (12:15). Bedroom match, stale threshold, snapping and comps method were chosen on window A at 11:56 (June-July API adverts had been seen by then; disclosed). Walk-forward suggests 24-month staleness, suburb ridge 3.0 and a $150-$5,000 price filter would be about $1-2 better; not applied because they were found after the rule was frozen.
