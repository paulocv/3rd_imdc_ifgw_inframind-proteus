# Outbreak-features component

Statistical predictions of the **macroscopic outbreak features** of a Brazilian dengue season,
per `(UF, year)`, for the 3rd Infodengue-Mosqlimate Dengue Challenge. This component produces
stochastic samples of three features that the outbreak-dynamics (renewal) model consumes to
reconstruct the full season.

The three delivered features (see `export_contract.md`):

| Contract feature   | Meaning                         | Model target (rate) → conversion        |
|--------------------|---------------------------------|-----------------------------------------|
| `case_attack_rate` | total cases in the season       | `size_attack_rate` /100k × pop / 100k   |
| `peak_amplitude`   | cases in the peak week          | `size_peak_incidence` /100k × pop / 100k|
| `peak_week`        | week of the peak, `1` = EW41    | `peak_timing_week` (identical)          |

## How to run

```bash
# full deliverable: 500 samples per (UF, year, feature) for the validation years
uv run run-outbreak-features --n-samples 500
# or, equivalently
uv run python src/scripts/run_outbreak_features.py --n-samples 500
```

Writes one CSV per feature to `predictions/` (`case_attack_rate.csv`, `peak_amplitude.csv`,
`peak_week.csv`), each with columns `location_id, year, i_sample, <feature>`. Options:
`--years`, `--level` (default `state` = UF), `--seed`, `--output-dir`, `--samples-out` (also
save the intermediate rate/week samples).

### The 2026-2027 forecast (the final-round deliverable)

```bash
uv run run-outbreak-features --n-samples 500 --years 2026 \
    --output-dir predictions/forecast_2026 \
    --samples-out predictions/forecast_2026/samples_rate_week_2026.csv
```

Already generated and validated in **`predictions/forecast_2026/`**: 500 samples x 27 UFs x 3
features, `year = 2026` (season EW41 2026 -> EW40 2027), issued from `t0 = EW202625`. This is the
input the outbreak-dynamics model consumes for the final submission.

Sanity figures: national season total **median 1.92M cases, 90% PI [1.08M, 2.97M]** (historical
range 0.50M-6.86M); peak-week medians follow the expected spatial gradient (North earliest ~EW04,
Southeast ~EW12-15, Northeast latest ~EW23).

`--years 2026` works because the year is absent from the IMDC fold columns: the pipeline falls
back to `season_t0(2026)` and `_augment_labels` adds NaN-target placeholder rows so the feature
pipeline still builds a row per UF.

## Required information

### 1. Team and contributors
Alberto Aleta (BIFI–Unizar, Spain), Marco Fernandez (BIFI–Unizar, Spain), Paulo Cesar Ventura
(IFGW–UNICAMP, Brazil).

### 2. Repository structure
This component lives in `src/inframind_proteus/outbreak_features/`:

| File | Role |
|---|---|
| `config.py` | constants & paths (targets, season rules, model hyperparameters) |
| `data.py` | `DataRepository`: weekly panel (cases + population → incidence), spatial aggregation, IMDC fold decoding |
| `labels.py` | the three macroscopic targets per `(unit, season)` |
| `season_features.py` | one feature vector per `(unit, season)` (epi history, static env/demography, climate, ocean) + `FeatureAssembler` |
| `features.py` | population-weighted ERA5 climate & Copernicus forecast aggregation |
| `base_model.py` | `BaseModel` contract + held-out permutation importance |
| `tabular.py` | `CatBoostModel` + season baselines |
| `residual.py` | `CatBoostAnomModel` — climatology-anomaly wrapper (fit the ratio to climatology) |
| `sequence.py` | `LSTMModel` (available for evaluation; not used in the delivered export) |
| `sarimax.py` | `SarimaxModel` — per-unit ARIMA + Fourier seasonality, weekly forecast |
| `baselines.py` | seasonal-naive / climatology weekly baselines |
| `evaluate.py` | derive macro targets from a weekly curve + metrics |
| `run.py` / `run_season.py` | evaluation harnesses (weekly-curve / season-matrix tracks) |
| `export.py` | **orchestrator**: rolling-origin fit + stochastic sampling per `(UF, year)` |
| `contract.py` | rate→count conversion, UF-acronym mapping, contract CSV writer |

The CLI is `src/scripts/run_outbreak_features.py` (console script `run-outbreak-features`).

### 3. Libraries and dependencies
`catboost` (size model), `statsmodels` (SARIMAX), `numpy`, `pandas`, `scikit-learn`, `scipy`,
`epiweeks`. `torch` (CPU) is only needed for the optional `LSTMModel`, not for the delivered
export. All are declared in `pyproject.toml`.

### 4. Data and variables
Datasets (from `data/data_imdc_2026/`): dengue weekly cases (`dengue.csv.gz`), DataSUS
population, ERA5 climate (`climate.csv.gz`), Copernicus seasonal forecast
(`forecasting_climate.csv.gz`), ocean teleconnections, chikungunya (cross-disease signal), and
the regional-health crosswalk. We model **rates** (cases /100k), converting to counts only at
export using the UF population of the season start year.

The feature matrix (one row per `(unit, season)`) is built in `season_features.py`:
pre-season epidemiological history and immunity proxies (`p5__…`), static environment and
demography (`p6__…`), observed ERA5 summaries (`p2__…`), Copernicus forecast features
(`p3__…`), and ocean teleconnections (`p4__…`). Feature construction and the leakage cutoff are
all in `season_features.py` / `features.py`.

### 5. Model training and forecasts
For each validation year `Y`, a rolling-origin fit trains on the complete seasons strictly
before `Y` and predicts `Y` from features observed up to the issue point (EW25 of `Y`):
- **size** (`case_attack_rate`, `peak_amplitude`): `CatBoostAnomModel` — CatBoost regressing the
  multiplicative anomaly to climatology, with per-quantile models for predictive intervals.
  CatBoost uses early stopping on the latest training season (leakage-safe).
- **timing** (`peak_week`): `SarimaxModel` — per-unit ARIMA(1,1,1) + annual Fourier terms on
  `log1p` weekly incidence, simulating stochastic trajectories over the full season horizon.

Training + forecast generation is `export.py`, driven by the CLI above. (At state level
`CatBoostAnomModel` outperforms the `LSTMModel` on every validation year and both size targets,
so the size samples are drawn from it alone.)

### 6. Data usage restriction (EW25)
The issue point is fixed at `t0 = EW25` of the season start year. Observed features use only data
with `epiweek <= t0`; every fitted transform (climatology baselines, scalers, early-stopping
splits) is fit on training seasons only. SARIMAX fits on weeks `<= t0` and forecasts EW41→EW40 of
the next year. For the 2025 target, whose data ends mid-season, the timing forecast uses a
synthetic full-season week grid so the peak is not truncated.

### 7. Predictive uncertainty
Size: each feature is sampled from `CatBoostAnomModel`'s predictive quantile ladder by
piecewise-linear inverse-CDF (with linear tail extrapolation). Timing: the argmax week of each
SARIMAX stochastic trajectory is one sample. Samples are produced per `(UF, year, feature)`;
counts are rounded to non-negative integers only after sampling.

The two size features share **one uniform draw per `(UF, i_sample)`**, so sample *i* sits at the
same quantile of both marginals (comonotonic coupling). Drawing them independently produced
physically impossible samples with `peak_amplitude > case_attack_rate`; the coupling removes them
(0/13500 in the 2026 forecast) while leaving each marginal unchanged. It is empirically justified:
the historical within-UF Spearman correlation between the two size targets is 0.968.

> **Known issue in the validation deliverable.** `predictions/*.csv` (years 2022-2025) were
> generated *before* this fix and still contain 3.69% incoherent samples. They are kept
> byte-identical to what was submitted for the validation round. Regenerate with the command in
> "How to run" if the renewal model's calibration would benefit.

### 7b. Data vintage
The organizers ship the EW01-EW25 2026 extension as separate `*_update_2026.csv.gz` files that
**overlap and revise** the base snapshot (the overlapping dengue weeks drop ~9.6% in total cases),
so `vintage.py` assembles both and lets the newer vintage win on the natural key. It also
normalizes the Copernicus column rename (`umid_med` -> `rel_umid_med`), drops the `epiweek` column
the refreshed ocean files carry, and carries DataSUS population forward to 2026 (the file stops at
2025). Re-scoring folds 1-3 under a simulated provisional vintage showed the effect on predictions
is ~1% with mixed sign -- well below model seed noise -- at **state** grain; this is untested and
plausibly material at municipality grain.

### 8. References
Moving Epidemic Method (MEM) and the Infodengue-Mosqlimate Dengue Challenge:
https://sprint.mosqlimate.org/
