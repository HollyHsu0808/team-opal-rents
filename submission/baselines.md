# Baseline ladder on identical test adverts


## Window A: train <= 2026-04-30, test 2026-05-01..2026-05-31 (1,318 adverts)

| step | median abs error | within $100 | R2 |
|---|---|---|---|
| 1. government bond data (postcode x bedrooms) | $100.0 | 52.1% | 0.569 |
| 2. suburb x bedrooms x parking medians (organiser-style simple estimate) | $92.2 | 54.1% | 0.642 |
| 3. + the flat's own past rent x repeat-rent index | $57.3 | 69.3% | 0.797 |
| 4. ours: + same-building comps (one flat, one vote), stale history out, $5 grid | $45.0 | 79.1% | 0.862 |

bond files used (published before the cut-off): rentalbond_lodgements_february_2026.xlsx, rentalbond_lodgements_january_2026.xlsx, rentalbond_lodgements_march_2026.xlsx, rentalbond_lodgements_year_2025.xlsx; 29,642 flat lodgements in the last 12 months

## Window B: train <= 2026-05-31, test 2026-06-01..2026-07-31 (2,898 adverts)

| step | median abs error | within $100 | R2 |
|---|---|---|---|
| 1. government bond data (postcode x bedrooms) | $100.0 | 51.3% | 0.504 |
| 2. suburb x bedrooms x parking medians (organiser-style simple estimate) | $97.2 | 52.5% | 0.589 |
| 3. + the flat's own past rent x repeat-rent index | $56.1 | 68.7% | 0.773 |
| 4. ours: + same-building comps (one flat, one vote), stale history out, $5 grid | $45.0 | 77.1% | 0.877 |

bond files used (published before the cut-off): rentalbond_lodgements_april_2026.xlsx, rentalbond_lodgements_february_2026.xlsx, rentalbond_lodgements_january_2026.xlsx, rentalbond_lodgements_march_2026.xlsx, rentalbond_lodgements_year_2025.xlsx; 29,137 flat lodgements in the last 12 months