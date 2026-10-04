# L&T Finance Farmer Income Prediction — Methodology & Insights
Team: katinnitk

## 1. Headline result
5-fold cross-validated MAPE: **18.74%** on the 47,970-row training set,
after blending three target formulations and applying an MAPE-optimal
shrinkage factor. Full breakdown and rationale below.

## 2. Key structural insight: the target decomposes exactly
`Target_Variable/Total Income = Non_Agriculture_Income + Agricultural Income`,
verified across all 47,970 training rows: the residual
`target − Non_Agriculture_Income` is never negative and is zero for only
0.5% of farmers. `Non_Agriculture_Income` is provided in the test set
too, so a meaningful share of the answer is already known and only the
agricultural component needs modelling. This is the single most
consequential fact about the dataset, and it shapes our whole approach:
our best-performing formulation (the largest weight in the final blend) predicts
`log1p(agricultural income)` and adds the known non-agricultural income
back on, rather than predicting total income directly.

## 3. Most useful engineered features
**Land × regional prosperity.** `log(landholding) × state-income-tier`
ranked **second in feature gain, ahead of the raw Non_Agriculture_Income
column itself**. The state tier alone is almost useless on its own
(negligible gain) — its value only appears in interaction with
landholding. The economic reading: the same number of hectares is worth
very different amounts of income depending on the regional economy the
farmer sits in (crop prices, market access, labour costs). Land size is
a necessary but not sufficient predictor; land size *combined with*
where the farmer is located is far stronger.

**Distance to the state capital.** Parsing the raw `Location`
lat/lon and computing a haversine distance to each farmer's state
capital landed at **rank 17 of 204 features** — a genuine, non-trivial
signal, not noise. A similarly-computed distance to the nearest of eight
major national metros (independent of state boundaries) also contributed
meaningfully. Both are remoteness/market-access proxies the raw lat/lon
columns alone don't expose to a tree model directly, since "distance to
a fixed point" is a nonlinear combination of latitude and longitude that
a tree can only approximate through many splits.

## 4. External data used, and why we stopped there
We added one external variable: a 5-tier state-level income ranking built
from the RBI *Handbook of Statistics on Indian Economy*, Table 10 (Per
Capita Net State Domestic Product, constant prices). We could cleanly
retrieve a single consistent 2021-22 vintage for 13 of the dataset's 17
states; for the remaining four (Telangana, Uttar Pradesh, West Bengal,
Chandigarh) the publicly available tables use incompatible price bases
across releases, so we placed them by well-corroborated general standing
(Telangana and Chandigarh top-tier, UP bottom-tier, West Bengal mid-tier,
consistent across multiple independent sources) rather than force a
number that might mix incompatible bases. We chose this honest, coarser
route deliberately: a wrong precise number is worse than an admittedly
approximate tier, and the tier still captures the signal that matters
(see §3).

We evaluated but did not pursue MSP (minimum support price) data, because
the dataset has no crop-type column to join it against — without knowing
what a farmer grows, a district-level MSP figure can't be attached
meaningfully. This is a natural next step if crop data becomes available.

## 5. Modelling approach
Three LightGBM (MAE objective) models are trained per formulation, 5-fold
CV, then blended:

| Formulation | What it predicts | OOF MAPE | Blend weight |
|---|---|---|---|
| A | log(total income) directly | 19.39 | 0.20 |
| B | log1p(agricultural income) + known non-agri income | 19.16 | 0.58 |
| C | log(income per hectare) × landholding | 19.47 | 0.22 |
| **Blend** | weighted average, then × 0.9675 shrink | **18.74** | — |

B dominates the blend, consistent with §2. The blend still beats B alone
(18.74 vs 19.16) — A and C each contribute a different bias/variance
trade-off (A ignores the land normalisation, C ignores non-agri income
directly), so averaging them cancels out some of each one's error.

The 0.9675 shrinkage factor is chosen by grid search on out-of-fold
predictions and is not a hack: MAPE's error-minimising point is a
weighted median rather than a mean, and our models' outputs (trained
under MAE, on a log scale) sit slightly above that point, so a small,
uniform downward correction reduces error. We floor every prediction at
the known `Non_Agriculture_Income` value, since total income cannot be
lower than its own non-agricultural component.

Two other modelling choices worth flagging:
- **Target encoding, not raw categories, for geography.** DISTRICT (405
  levels), CITY (2,721), VILLAGE (5,650) and Zipcode (4,520) are
  out-of-fold target-encoded on log(income) with count-based smoothing
  (prior weight 20), rather than passed as raw high-cardinality
  categoricals — this avoids both overfitting and an unwieldy one-hot
  space. `te_DISTRICT` is the single highest-gain feature in the model.
- **Village-level columns are duplicated across farmers in the same
  village** (soil type, rainfall, groundwater, agro-ecological zone,
  census-style living-index percentages). We treat these as context
  rather than farmer-level signal, which is why farmer-specific columns
  (land, non-agri income, bureau history, marital status) carry
  disproportionate weight in the final model relative to their small
  share of the 105 raw columns.

## 6. Sanity check on test predictions: the inverse farm-size relationship
Beyond cross-validated MAPE, we checked whether the actual test-set
predictions behave sensibly. Median predicted agricultural income per
hectare falls monotonically as landholding size increases:

| Land size bucket | Median agri income / hectare |
|---|---|
| Smallest 20% | ₹148,952 |
| Small 20% | ₹109,261 |
| Mid 20% | ₹79,007 |
| Large 20% | ₹63,979 |
| Largest 20% | ₹52,628 |

This is not something we engineered or targeted — it emerged from the
model on its own. It also matches a well-documented phenomenon in
agricultural economics, the **inverse farm size–productivity
relationship**: smallholders typically farm more intensively per hectare
(more labour input relative to land, more attention per plot, often
higher-value crops), while larger holdings tend toward lower per-hectare
intensity even as total output rises. Reproducing this pattern without
being told about it is, to us, more reassuring evidence of genuine
structure than the aggregate MAPE number alone — it's a check a judge
can also verify directly against the submitted predictions. We also
confirmed no predicted test income falls below its corresponding known
`Non_Agriculture_Income` (0 violations across all 9,986 test rows), and
no zero or negative predictions.

## 7. Where the model struggles — and why that's expected, not a bug
Per-decile OOF MAPE is a clear U-shape:

| Income decile | Median income (₹) | MAPE |
|---|---|---|
| 0 (poorest) | 550,000 | 31.2% |
| 1–6 (middle) | 680,000–1,150,000 | 12–14% |
| 7 | 1,300,000 | 17.4% |
| 8 | 1,586,000 | 22.6% |
| 9 (richest) | 2,400,000+ | 33.8% |

This pattern is close to inherent to the metric, not a sign the model is
missing something fixable: MAPE weights every rupee of absolute error at
the poorest decile far more heavily (as a percentage) than the same
rupee error at the median, while the richest decile is both thin (fewer
training examples above ₹6.3M — the 99th percentile in raw counts) and
genuinely more heterogeneous (a handful of farmers report incomes up to
₹8 crore). We would flag this explicitly to the panel rather than
present a single blended MAPE as if performance were uniform.

## 8. Honest limitations / next steps
- No hyperparameter search beyond manual tuning of learning rate,
  num_leaves and regularisation — a proper Optuna/Bayesian sweep would
  likely buy another fraction of a point.
- The blend weights and shrinkage factor are fit on the same 5-fold OOF
  predictions they are then evaluated against; a nested/repeated CV
  would give a less optimistic estimate of the true holdout MAPE (18.74%
  should be read as a slightly best-case number for this reason).
- Crop-type-linked external data (MSP, crop yield) is not usable without
  a crop-type column in the given data — worth requesting from the
  organizers if pursuing this further.
