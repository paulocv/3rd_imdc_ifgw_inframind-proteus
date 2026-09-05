"""Project-wide constants and paths for the outbreak-features component."""
from pathlib import Path

# config.py lives at <repo>/src/inframind_proteus/outbreak_features/config.py
#   parents[0]=outbreak_features  [1]=inframind_proteus  [2]=src  [3]=<repo root>
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "data_imdc_2026"
RESULTS_DIR = REPO_ROOT / "outputs" / "outbreak_features"

DENGUE_FILE = DATA_DIR / "dengue.csv.gz"
CHIK_FILE = DATA_DIR / "chikungunya.csv.gz"
ENVIRON_FILE = DATA_DIR / "environ_vars.csv.gz"
OCEAN_FILE = DATA_DIR / "ocean_climate_oscillations.csv.gz"
POP_FILE = DATA_DIR / "datasus_population_2001_2025.csv.gz"
CROSSWALK_FILE = DATA_DIR / "map_regional_health.csv"
CLIMATE_FILE = DATA_DIR / "climate.csv.gz"
FORECAST_FILE = DATA_DIR / "forecasting_climate.csv.gz"
CACHE_DIR = REPO_ROOT / ".cache"

# 2026-2027 forecast phase
DENGUE_UPDATE_FILES = [DATA_DIR / "dengue_update_2026.csv.gz"]
CHIK_UPDATE_FILES = [DATA_DIR / "chikungunya_update_2026.csv.gz"]
CLIMATE_UPDATE_FILES = [DATA_DIR / "climate_update_2026.csv.gz"]
FORECAST_UPDATE_FILES = [DATA_DIR / "forecasting_climate_update_2026.csv.gz"]
OCEAN_UPDATE_FILES = [DATA_DIR / "ocean_climate_oscillations_update_2026.csv.gz"]
OCEAN_REFRESHED_FILE = DATA_DIR / "ocean_climate_oscillations_refreshed.csv.gz"
POP_EXTEND_TO_YEAR = 2026          # DATASUS stops at 2025; carry denominators forward

# Spatial level -> the key column (present in both dengue.csv.gz and the crosswalk).
SPATIAL_LEVELS = {
    "municipality": "geocode",
    "regional": "regional_geocode",
    "macroregional": "macroregional_geocode",
    "state": "uf_code",
}

# Season runs EW41 -> EW40 of the next year, labeled by its start year.
SEASON_START_WEEK = 41
MIN_SEASON_WEEKS = 50          # a season needs >= this many weeks to be labeled (complete)
INCIDENCE_SCALE = 100_000      # cases per 100k

# SARIMAX: per-unit ARIMA on log1p(incidence) + Fourier annual seasonality, no exog by default.
ANNUAL_PERIOD = 365.25 / 7     # ~52.18 weeks per year (for Fourier terms)
SARIMAX_ORDER = (1, 1, 1)      # (p, d, q); no seasonal order (Fourier handles seasonality)
SARIMAX_FOURIER_K = 2          # number of Fourier harmonic pairs
SARIMAX_MIN_TRAIN_WEEKS = 60       # below this many training weeks, fall back to baseline
SARIMAX_MIN_NONZERO_WEEKS = 52     # sparsity guard: below this many nonzero-incidence training
                                   # weeks, fall back (zero-inflated small units)

# Simulation-based predictive distributions for the macro targets.
SARIMAX_N_SIMS = 500           # sample paths per (unit, fold); 0 disables simulation
SARIMAX_SIM_CAP_MULT = 3.0     # cap simulated weekly incidence at this x the unit's historical max
                               # (tames the explosive I(1) log-scale tail at the ~67-week horizon)

# Climate variables aggregated into the season-matrix climate blocks (population-weighted ERA5).
CLIMATE_VARS = ["temp_med", "precip_med", "rel_humid_med"]

# ---- Tabular direct-regression track (season-grain) -----------------------
# Features are one vector per (unit, season); each row uses only data <= t0(s) = EW25 of the
# season's start year (so feature rows are season-intrinsic & leakage-safe).
SEASON_ISSUE_WEEK = 25         # t0 within-year epiweek (EW25 ~ late June); season_t0(s) = s*100+25

# Targets trained on log1p scale (right-skewed rates); reported on the natural scale.
LOG1P_TARGETS = ("size_peak_incidence", "size_attack_rate")

# P5 epidemiological-history knobs.
P5_TARGET_LAGS = (1, 2, 3)         # season lags of the targets used as features
P5_IMMUNITY_WINDOW = 3             # seasons in the cumulative-attack-rate immunity proxy
P5_PRESEASON_WEEKS = 12            # recent weeks <= t0 summarized as the pre-season "tail"
P5_INCLUDE_CHIK = True             # add chikungunya pre-season incidence (cross-disease)

# Climate/ocean feature blocks. All season-intrinsic & leakage-safe: features for season s use
# only data <= t0(s); anomaly baselines use each row's own <=t0 history (so the feature matrix
# stays fold-independent). Toggles for ablation.
INCLUDE_P2_CLIMATE = True          # observed ERA5 pre-season window summaries
INCLUDE_P3_FORECAST = True         # Copernicus seasonal-forecast features (the in-season signal)
INCLUDE_P4_OCEAN = True            # ENSO/IOD/PDO teleconnections (global -> broadcast by season)
SEASON_ISSUE_FCST_MONTH = 6        # forecast reference month at t0 ~ EW25 (late June)
P2_PRESEASON_WINDOWS = (12, 26)    # ERA5 lead-up windows (weeks before t0)

# CatBoost hyperparameters (shallow + regularized for the small state/regional panels;
# early stopping on a held-out latest train season — see CatBoostModel).
CATBOOST_PARAMS = dict(
    iterations=2000, learning_rate=0.03, depth=4, l2_leaf_reg=6.0,
    loss_function="RMSE", random_seed=0, allow_writing_files=False, verbose=False,
)
CATBOOST_EARLY_STOPPING = 100      # od_wait rounds; uses the latest train season as eval set
CATBOOST_QUANTILES = (0.05, 0.5, 0.95)   # per-quantile models -> predictive intervals

# Held-out permutation importance: a model-agnostic driver basis (MAE rise when a raw feature
# column is shuffled) computed on the target-season fold the model never trained on, pooled
# across folds (an out-of-sample driver attribution).
HELDOUT_PERM = True
HELDOUT_PERM_MODELS = ("catboost", "catboost_anom")
HELDOUT_PERM_REPEATS = 5           # shuffles averaged per feature
HELDOUT_PERM_MAX_ROWS = 4000       # subsample target-season rows (each repeat re-predicts all rows)

# ---- LSTM (sequence -> scalar, hybrid: weekly history + tabular season features) ----
# One model per target; encodes the leakage-safe pre-season weekly incidence sequence
# (<= t0 = EW25) and concatenates a tabular head over the season matrix.
LSTM_LOOKBACK = 104            # weeks of pre-t0 weekly history fed to the recurrent encoder (~2 yr)
LSTM_HIDDEN = 64               # LSTM hidden size = tabular-head width
LSTM_LAYERS = 1                # recurrent layers (few seasons -> keep shallow)
LSTM_DROPOUT = 0.3             # dropout (also reused for MC-dropout intervals)
LSTM_LR = 1e-3
LSTM_WEIGHT_DECAY = 1e-4       # L2 (data-hungry net, few seasons -> regularize)
LSTM_BATCH = 256
LSTM_MAX_EPOCHS = 200
LSTM_PATIENCE = 20             # early-stopping patience on the held-out latest train season
LSTM_CONFORMAL = True          # split-conformal intervals (fixes MC-dropout over-confidence).
                               # Honest residuals from a shadow net trained on the earlier seasons
                               # and calibrated on the last K held-out seasons; falls back to
                               # single-season then MC-dropout if too few seasons.
LSTM_CONFORMAL_SEASONS = 3     # calibrate on the last K train seasons (shadow-model split-conformal);
                               # bigger K -> steadier quantiles but a shorter shadow-train window.
LSTM_MC_SAMPLES = 50           # stochastic forward passes for the MC-dropout fallback
LSTM_QUANTILES = (0.05, 0.5, 0.95)
LSTM_SEED = 0
LSTM_IMPORTANCE_MAXROWS = 4000 # cap rows for block-permutation importance (keep it cheap)

# The three macroscopic targets delivered by this component (duration dropped — see __init__).
TARGETS = [
    "size_peak_incidence",   # max weekly incidence /100k  -> peak_amplitude (count)
    "size_attack_rate",      # sum of weekly incidence /100k over the season -> case_attack_rate (count)
    "peak_timing_week",      # within-season week index (1 = EW41) of the peak -> peak_week
]
