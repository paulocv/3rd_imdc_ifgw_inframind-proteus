"""Orchestrator: produce stochastic samples of the macroscopic outbreak features per
(unit, season) for the validation years.

For each validation year Y, a rolling-origin fit trains on the complete seasons strictly
before Y and predicts Y from features observed up to the issue point t0 = EW25 of Y:

  * size features (`size_peak_incidence`, `size_attack_rate`, rates /100k) -- `catboost_anom`,
    whose predictive quantile ladder is sampled by piecewise-linear inverse-CDF. (At state level
    `catboost_anom` beats `lstm` on every validation year and both size targets, so the size
    samples are drawn from it alone rather than a pooled ensemble.)
  * timing (`peak_timing_week`, within-season index, 1 = EW41) -- SARIMAX simulates stochastic
    weekly trajectories over the full season horizon; the argmax week of each trajectory is one
    sample.

Output is long-form and stays in **rate / week space**; the rate->count conversion and the
UF-acronym mapping live downstream. Sample size need not match across features/units/years.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from epiweeks import Week

from . import config
from .data import DataRepository
from .labels import build_labels
from .season_features import FeatureAssembler, season_t0
from .residual import CatBoostAnomModel
from .sarimax import SarimaxModel

SIZE_TARGETS = ("size_peak_incidence", "size_attack_rate")
TIMING_TARGET = "peak_timing_week"
EXPORT_QUANTILES = (0.05, 0.5, 0.95)   # quantile ladder sampled for the size features
DEFAULT_YEARS = (2022, 2023, 2024, 2025)


# ---- sampling helper -------------------------------------------------------
def sample_from_quantiles(qlevels, qvals, n: int, rng, clip_min: float = 0.0,
                          u: np.ndarray | None = None) -> np.ndarray:
    """Draw n samples from a distribution defined by a quantile ladder.

    Piecewise-linear inverse-CDF between the given quantiles; the tails are linearly
    extrapolated from the outermost segments. `qvals` must be sorted ascending (monotone).

    Pass `u` to supply the uniform draws instead of taking fresh ones from `rng`; sharing one
    `u` across several targets couples their samples comonotonically (see `_size_samples`).
    """
    qlevels = np.asarray(qlevels, float)
    qvals = np.asarray(qvals, float)
    u = rng.uniform(0.0, 1.0, n) if u is None else np.asarray(u, float)
    s = np.interp(u, qlevels, qvals)                       # clamps to endpoints outside the ladder
    lo_slope = (qvals[1] - qvals[0]) / (qlevels[1] - qlevels[0])
    hi_slope = (qvals[-1] - qvals[-2]) / (qlevels[-1] - qlevels[-2])
    left, right = u < qlevels[0], u > qlevels[-1]
    s[left] = qvals[0] - (qlevels[0] - u[left]) * lo_slope
    s[right] = qvals[-1] + (u[right] - qlevels[-1]) * hi_slope
    return np.clip(s, clip_min, None)


# ---- full-season scaffold (for timing trajectories) ------------------------
def season_epiweeks(season_start_year: int) -> list[int]:
    """All epiweeks of a season EW41(start year) .. EW40(start year + 1), as YYYYWW ints."""
    cur, end = Week(season_start_year, config.SEASON_START_WEEK), Week(season_start_year + 1, 40)
    out = []
    while True:
        out.append(cur.year * 100 + cur.week)
        if cur == end:
            return out
        cur = cur + 1


def _full_season_udf(udf: pd.DataFrame, unit_col, unit, season: int, t0: int) -> pd.DataFrame:
    """Ensure the unit panel covers a contiguous week grid from t0+1 through EW40 of the next
    year, synthesizing any missing future weeks (NaN incidence) so SARIMAX can forecast the
    *whole* season even when the data ends mid-season (e.g. the 2025 target)."""
    seq = season_epiweeks(season)
    pos = {ew: i + 1 for i, ew in enumerate(seq)}           # epiweek -> week_in_season (1-based)
    end_ew = seq[-1]
    have = set(udf["epiweek"].astype(int))
    cur = Week(t0 // 100, t0 % 100) + 1                     # first week strictly after t0
    rows = []
    while True:
        ew = cur.year * 100 + cur.week
        if ew not in have:
            wk = ew % 100
            rows.append({unit_col: unit, "epiweek": ew, "year": ew // 100,
                         "season": ew // 100 if wk >= config.SEASON_START_WEEK else ew // 100 - 1,
                         "week_in_season": pos.get(ew, np.nan), "incidence": np.nan,
                         "casos": np.nan, "population": np.nan})
        if ew == end_ew:
            break
        cur = cur + 1
    if not rows:
        return udf
    add = pd.DataFrame(rows)
    return pd.concat([udf, add], ignore_index=True).sort_values("epiweek").reset_index(drop=True)


# ---- orchestrator ----------------------------------------------------------
def _augment_labels(labels: pd.DataFrame, units, years) -> pd.DataFrame:
    """Add placeholder (unit, year) rows (NaN targets) for target years not present, so the
    feature pipeline builds feature rows for them (e.g. the incomplete 2025 season)."""
    have = set(map(tuple, labels[["unit", "season"]].to_numpy()))
    extra = [{"unit": u, "season": int(Y)} for Y in years for u in units if (u, int(Y)) not in have]
    if not extra:
        return labels
    add = pd.DataFrame(extra)
    for c in labels.columns:
        if c not in add.columns:
            add[c] = np.nan
    return pd.concat([labels, add[labels.columns]], ignore_index=True)


def _size_samples(repo, asm, lab_idx, X_tr, X_ts, target, n_samples, rng, u=None):
    """Empirical size samples (rate space) from catboost_anom's predictive quantile ladder.

    `u` is a (units x n_samples) matrix of uniform draws shared across the size targets, so
    sample i of a unit sits at the same quantile of every size marginal. The two size targets
    are marginals of one season, and drawing them independently produced physically impossible
    samples with `peak_amplitude > case_attack_rate` (3.7% of the validation deliverable).
    Comonotonic coupling makes a big season draw a big peak; it does not *guarantee* the
    ordering, since the two marginals are predicted separately, so the caller checks.
    """
    y_tr = lab_idx.reindex(X_tr.index)[target]
    units = X_ts.index.get_level_values("unit").to_numpy()
    model = CatBoostAnomModel(target, cat_features=asm.cat_features, repo=repo,
                             quantiles=EXPORT_QUANTILES)
    model.fit(X_tr, y_tr, cat_features=asm.cat_features)
    q = model.predict_quantiles(X_ts)                      # (units x quantiles), natural scale
    cols = list(q.columns)
    draws = np.empty((len(units), n_samples))
    for i in range(len(units)):
        draws[i] = sample_from_quantiles(cols, np.sort(q.iloc[i].to_numpy()), n_samples, rng,
                                         u=None if u is None else u[i])
    return units, draws                                    # (units, n_samples)


def _timing_samples(repo, year, t0, n_samples, rng):
    """peak_timing_week samples per unit from SARIMAX stochastic trajectories (full season).

    Trajectories that simulate no epidemic (zero total incidence) have no meaningful peak, so
    their week is resampled from the epidemic-bearing trajectories instead of defaulting to the
    first week; a unit with no epidemic-bearing trajectory at all falls back to the point curve.
    """
    panel = repo.panel()
    unit_col = repo.unit_col
    wk = np.arange(1, len(season_epiweeks(year)) + 1)      # within-season indices, 1 = EW41
    meta = {"target_season": year, "issue_epiweek": t0}
    sar = SarimaxModel(n_sims=n_samples)
    units, samples, n_fallback = [], [], 0
    for u, g in panel.groupby(unit_col):
        udf = _full_season_udf(g.sort_values("epiweek"), unit_col, u, year, t0)
        curve = sar.predict_curve(udf, meta, wk.tolist())
        peak = np.full(n_samples, np.nan)
        paths = sar.last_paths                              # (n_sims, W) or None
        if paths is not None:
            epi = np.nansum(paths, axis=1) > 0
            if epi.any():
                peak[epi] = wk[np.nanargmax(paths[epi], axis=1)]
                if not epi.all():
                    peak[~epi] = rng.choice(peak[epi], size=int((~epi).sum()))
        if np.isnan(peak).any():                           # no epidemic-bearing trajectory
            n_fallback += 1
            arr = curve.to_numpy(float)
            peak[np.isnan(peak)] = wk[int(np.nanargmax(arr))] if np.nansum(arr) > 0 else wk[0]
        units.append(u)
        samples.append(peak)
    return np.array(units), np.vstack(samples), n_fallback  # (units,), (units, n_samples)


def build_features_export(years=DEFAULT_YEARS, n_samples: int = 500, level: str = "state",
                          seed: int = 0) -> pd.DataFrame:
    """Return long-form rate/week samples: columns [unit, year, target, i_sample, value]."""
    rng = np.random.default_rng(seed)
    repo = DataRepository(level)
    labels = build_labels(repo)
    units_all = sorted(repo.panel()[repo.unit_col].unique().tolist())
    labels_plus = _augment_labels(labels, units_all, years)
    lab_idx = labels.set_index(["unit", "season"]).sort_index()
    asm = FeatureAssembler(repo, labels_plus)
    folds = {m["target_season"]: m for m in repo.folds().values()}

    out = []
    for Y in years:
        Y = int(Y)
        t0 = folds[Y]["issue_epiweek"] if Y in folds else season_t0(Y)
        train_seasons = sorted(s for s in labels["season"].unique() if s < Y)
        asm.fit(train_seasons)
        X_tr, X_ts = asm.transform(train_seasons), asm.transform([Y])
        print(f"[year {Y}] t0={t0} | train {len(X_tr)} rows / {len(train_seasons)} seasons "
              f"| target {len(X_ts)} units")

        # one uniform matrix shared by both size targets -> comonotonic coupling (see _size_samples)
        U = rng.uniform(0.0, 1.0, (len(X_ts), n_samples))
        size_draws = {}
        for target in SIZE_TARGETS:
            units, draws = _size_samples(repo, asm, lab_idx, X_tr, X_ts, target, n_samples, rng, u=U)
            size_draws[target] = draws
            for i, u in enumerate(units):
                out.append(pd.DataFrame({"unit": u, "year": Y, "target": target,
                                         "i_sample": np.arange(draws.shape[1]), "value": draws[i]}))

        # a season's peak week cannot exceed its total; coupling makes this rare, not impossible
        bad = size_draws["size_peak_incidence"] > size_draws["size_attack_rate"]
        if bad.any():
            print(f"    [size] {bad.sum()}/{bad.size} samples ({100 * bad.mean():.2f}%) have "
                  f"peak > total after coupling")

        units, draws, n_fb = _timing_samples(repo, Y, t0, n_samples, rng)
        if n_fb:
            print(f"    [timing] {n_fb}/{len(units)} units fell back to baseline peak")
        for i, u in enumerate(units):
            out.append(pd.DataFrame({"unit": u, "year": Y, "target": TIMING_TARGET,
                                     "i_sample": np.arange(draws.shape[1]), "value": draws[i]}))

    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    df = build_features_export(n_samples=n)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / "feature_samples_state.csv"
    df.to_csv(path, index=False)
    print(f"\nsaved {len(df)} sample rows -> {path}")
    print(df.groupby(["target"])["value"].describe().round(2).to_string())
