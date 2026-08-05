# Outbreak-dynamics component

Renewal-equation projections of weekly dengue cases per `(UF, year)` for the 3rd Infodengue-Mosqlimate Dengue Challenge. This component calibrates a stochastic mechanistic model from historical UF time series, updates multi-year parameter distributions with outbreak-feature predictions, and exports forecast intervals for EW41 to EW40.

## Reproducing the pipeline

```bash
# 1) Calibrate outbreak dynamics (stages 1/2/3)
uv run calibrate-3rd-imdc

# 2) For each location-year calibration folder, compute outbreak-feature stats from posteriors
uv run calc-outbreak-features-from-posteriors --output-dir outputs/validation_round_calibration/<UF>_<YEAR>

# 3) Combine calibration years and apply outbreak-feature Bayesian update
uv run process-data-for-projections

# 4) Run final projections and export submission-ready predictive intervals
uv run project-3rd-imdc
```

# Required information

### 1. Team and contributors
Paulo Cesar Ventura (IFGW-UNICAMP, Brazil), Alberto Aleta (BIFI-Unizar, Spain), Marco Fernandez (BIFI-Unizar, Spain).

### 2. Repository structure
Core dynamics code is in `src/inframind_proteus/outbreak_dynamics/`.

| File | Role |
|---|---|
| `simulator.py` | `RenewalSimulator`, simulation config/data classes, renewal execution, case-beam and scoring flow |
| `rt_models.py` | Reproduction-number trajectories (`logistic`, `enveloped_logistic`) |
| `generation_time.py` | Generation-time PMF models (`ConstantGammaGT`) |
| `sampling.py` | Sobol/LHS/given sampling and priors for calibration parameters |
| `scoring.py` | NB loglikelihood, WIS, coverage metrics, Cornish-Fisher NB quantiles |
| `initial_infections.py` | Warm-up infection seeding |
| `outbreak_features.py` | Outbreak-feature extraction from trajectories and prediction cache utilities |
| `utils.py` | Miscellaneous helpers (time, YAML, parallel) |

Pipeline scripts:

| Script | Role |
|---|---|
| `src/scripts/prepare_dengue_time_series.py` | Aggregate municipality dengue notifications to UF weekly series |
| `src/make_uf_table.py` | Build UF metadata + yearly population table |
| `src/scripts/calibrate_3rd_imdc/` | Three-stage calibration procedure (`calibrate-3rd-imdc`) |
| `src/scripts/calc_outbreak_features_from_posteriors.py` | Derive outbreak-feature distributions from calibrated trajectories |
| `src/scripts/process_data_for_projections.py` | Multi-year prior + outbreak-feature Bayesian update |
| `src/scripts/project_3rd_imdc.py` | Final projection runs and interval export |

### 3. Libraries and dependencies
Core dependencies are declared in `pyproject.toml`: `numpy`, `pandas`, `scipy`, `numba`, `scikit-learn`, `statsmodels`, `epiweeks`, `matplotlib`, and project tooling via `uv`.

### 4. Data and variables
The Outbreak Dynamics component uses only population and past disease cases data. The following files from the IMDC repository were used:

- IMDC dengue notifications: `data/data_imdc_2026/dengue.csv.gz`
- IMDC population: `data/data_imdc_2026/datasus_population_2001_2025.csv.gz`
- Regional crosswalk for UF metadata: `data/data_imdc_2026/map_regional_health.csv`
- Outbreak-feature predicted samples (from the other component): `predictions/case_attack_rate.csv`, `predictions/peak_amplitude.csv`, `predictions/peak_week.csv`

Prepared intermediate data (processed from IMDC files):

- UF weekly dengue notifications: `data/disease/dengue_cases_uf_weekly/dengue_<UF>.csv`
- UF metadata/population table: `data/demographic/uf_table.csv`

Note: Population data from 2025 was repeated for 2026 and 2027 when needed.

No external data sources were used.


### 5. Model training and forecasts
Calibration was achieved through a multistage Bayesian approach, aimed first at reproducing past seasons, then using predictions from the Outbreak Features component to narrow down projections. 

#### 5.1 Yearwise 3-stage calibration
The initial 3-stage calibration runs independently for each location and season, orchestrated by the `src/scripts/calibrate_3rd_imdc.py` file (`calibrate-3rd-imdc` entry point).

Each simulation starts at EW26 of reference year `Y` and runs 69 weekly steps. The calibration scoring window runs from EW41 of year `Y` to EW25 of `Y+1`, ensuring all years use the same range of epidemiologic weeks while respecting data availability.


Stage summary:

1. Stage 1: broad Sobol exploration (overdispersion removed), with negative-binomial loglikelihood scoring, retain max-likelihood parameters. Use gamma-shaped prior for `notif_relative_scale` to match recent numbers of cases; other parameters have uniform priors.
2. Stage 2: free parameters reduced to `rt_logist_r_high` and `rt_logist_start`; nuisance parameters fixed to stage-1 maximum likelihood; weighted posterior samples regularized and converted to Gaussian KDE.
3. Stage 3: sample (`rt_logist_r_high`, `rt_logist_start`) from stage-2 KDE and sample inverse overdispersion uniformly within configured bounds; score by coverage loglikelihood (50% and 95% intervals), regularize weights, and export posterior samples.

Main stage-3 artifacts per location-year are exported under:

- `outputs/validation_round_calibration/<UF>_<YEAR>/stage3_posterior_samples.csv.gz`
- `outputs/validation_round_calibration/<UF>_<YEAR>/stage3_mean_cases.csv.gz`
- `outputs/validation_round_calibration/<UF>_<YEAR>/stage3_case_stats_trajectories.csv`

Then `calc-outbreak-features-from-posteriors` computes feature stats from stochastic trajectories (`outbreak_feature_stats.csv.gz`).

#### 5.2. Update with outbreak feature predictions

All available calibration years are combined into a single distribution, which is updated to follow predictions of total number of cases, peak size and peak week.

The `src/scripts/process_data_for_projections.py` script merges available calibration years (`year < projection_year`, 
respecting the data usage restriction), reweights by year so all years are equally likely, then applies outbreak-feature likelihoods. 
It then samples `N=5000` projection parameter sets (`projection_parameter_samples.csv`) from the updated posterior distributions.

After visual inspection, 3 location/year pairs were excluded from the calibrated posterior due to poor model fit: AL_2015, PB_2021, SE_2018. 
Exclusion done by setting `exclude_years_by_location` in `process_data_for_projections_default.yaml`. 

### 5.3 Forecasting 
`src/scripts/project_3rd_imdc.py` runs projection trajectories and exports `imdc_submission.csv` per location and projection-year.

### 6. Data usage restriction (EW25)
Restriction is enforced by ensuring the following two conditions:

- calibration windows end at EW25 of `Y+1` for each calibrated year - applied to keep it standard;
- projection priors only use calibration years strictly before the projection year (`year < projection_year`);

### 7. Predictive uncertainty
Uncertainty of the final projections encompasses both uncertainty in model parameters and stochastic fluctuations.

- During projections, one stochastic trajectory is sampled per posterior parameter sample. Predictive intervals are calculated as the empirical quantiles across all obtained trajectories at each week (median plus required 50/80/90/95% intervals).

Obs: Stochastic fluctuations are mostly encoded in the negative-binomial overdispersion, adjusted in stage 3 of the multistage calibration procedure to match 50% and 95% interval coverages.

### 8. References


- Generation-time context used in model notes: https://www.sciencedirect.com/science/article/pii/S1755436517300907