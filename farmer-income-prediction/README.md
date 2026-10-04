# Farmer Income Prediction: L&T Finance Challenge

**3rd place out of 126 participants** (team *Katin Katin*, team of 2) in the L&T Finance farmer income prediction challenge.

The task: predict each farmer's total annual income from 105 raw columns (demographics, land, soil, climate, village-level socio-economic indicators, credit bureau history). 47,970 training rows, 9,986 test rows. Metric: **MAPE**.

**Our 5-fold out-of-fold MAPE: 18.74%** (a slightly optimistic estimate; see [Limitations](#limitations)).

## Approach

1. **Exploit the structure of the target.** Total income equals non-agricultural income plus agricultural income exactly (checked on all training rows). Non-agricultural income is given in the test set, so one of our models predicts only the agricultural part and adds the known part back. Final predictions are also floored at the known non-agricultural income.
2. **Domain-driven features.** Land × state income tier (the second most useful feature by gain), haversine distance to the state capital and to the nearest major metro, rainfall variability, market access (road density vs. mandi distance), debt per hectare, groundwater balance, and more. Roughly 200 features in total.
3. **Out-of-fold target encoding** for high-cardinality geography (district, city, village, zipcode, nearest mandi), with count-based smoothing so rows never see their own target.
4. **Light external data:** a 5-tier state income ranking from the RBI *Handbook of Statistics on Indian Economy* (Table 10, per capita NSDP). MSP data was evaluated and rejected because the dataset has no crop-type column to join on.
5. **Blend of three LightGBM formulations** (MAE objective, 5-fold CV), then a small MAPE-optimal shrink factor:

| Formulation | Predicts | OOF MAPE |
|---|---|---|
| A | log(total income) | 19.39 |
| B | log1p(agricultural income) + known non-agri income | 19.16 |
| C | log(income per hectare) × landholding | 19.47 |
| **Blend** | weighted average × shrink | **18.74** |

The model also reproduces the inverse farm-size/productivity relationship on test predictions without being told to. Full write-up, including the error breakdown by income decile: [`docs/INSIGHTS.md`](docs/INSIGHTS.md). Competition slides: [`docs/presentation.pdf`](docs/presentation.pdf).

## Repo layout

```
src/features.py        feature engineering + out-of-fold target encoding
src/solution.py        end-to-end pipeline: features -> 3 models -> blend -> submission CSV
docs/INSIGHTS.md       methodology and findings
docs/presentation.pdf  final presentation
submission/            our submitted predictions
data/                  put the competition files here (not committed)
```

## Reproduce

```bash
pip install -r requirements.txt
# put the competition files in data/ first (see data/README.md)
python src/solution.py \
  --data data/LTF_Farmer_Income_data_with_dictionary_For_Share_KGP_v2.xlsx \
  --sample data/sample_submission_file.csv \
  --out submission.csv
```

Seeded (`SEED = 42`), so results should be stable for a given library version. Package versions are not pinned.

## Limitations

- Blend weights and the shrink factor are tuned on the same out-of-fold predictions they are scored on, so 18.74% is a best-case estimate. Nested or repeated CV would be more honest.
- No systematic hyperparameter search (manual tuning only).
- Error is U-shaped across income deciles: about 12-14% in the middle, over 30% for the poorest and richest deciles.

## Team

Shreya Harikumar and Divya, NITK Surathkal.
