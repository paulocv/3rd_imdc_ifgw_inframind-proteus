"""Scoring functions for evaluating simulation outputs against observations.

Primary method
--------------
``wis_score_vectorized`` — Weighted Interval Score (WIS) applied to
deterministic case beam quantiles.

Fallback metrics (individual trajectories)
------------------------------------------
``rmse_vectorized``, ``smape_vectorized`` — used when case beam scoring
is not applicable (e.g. single-trajectory calibration).

Helper
------
``nbinom_ppf_cf`` — fast Cornish-Fisher approximation to the Negative
Binomial PPF, used to build case beam quantiles without calling
``scipy.stats.nbinom.ppf`` for every simulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats


# ---------------------------------------------------------------------------
# Negative Binomial PPF (Cornish-Fisher approximation)
# ---------------------------------------------------------------------------

def nbinom_ppf_cf(
    q: float | np.ndarray,
    n: np.ndarray,
    p: np.ndarray,
    continuity: bool = True,
) -> np.ndarray:
    """Cornish-Fisher approximation to the Negative Binomial PPF.

    Fast vectorised alternative to ``scipy.stats.nbinom.ppf`` for computing
    case beam quantiles across many simulations simultaneously.

    Uses the SciPy parameterisation: X ~ nbinom(n, p) counts failures
    before n successes.

    Parameters
    ----------
    q:
        Quantile(s) in ``(0, 1)``.
    n:
        Number-of-successes parameter (> 0).
    p:
        Success-probability parameter in ``(0, 1)``.
    continuity:
        Apply a +0.5 continuity correction before returning.

    Returns
    -------
    np.ndarray
        Approximate quantile values (continuous, not rounded to integer).
    """
    from scipy.stats import norm

    n = np.asarray(n, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    z = norm.ppf(q)
    z2 = z * z
    z3 = z2 * z

    # When p == 1 (zero expected cases), the CF expansion is undefined but the
    # correct answer is 0; errstate suppresses divide-by-zero/invalid warnings
    # and np.maximum clips any NaN/negative results to 0 afterwards.
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = n * (1.0 - p) / p
        sigma = np.sqrt(n * (1.0 - p)) / p

        # Third-order Cornish-Fisher skewness/kurtosis terms
        gamma1 = (2.0 - p) / np.sqrt(n * (1.0 - p))
        gamma2 = (p * p - 6.0 * p + 6.0) / (n * (1.0 - p))

        cf = (
            z
            + (gamma1 / 6.0) * (z2 - 1.0)
            + (gamma2 / 24.0) * (z3 - 3.0 * z)
            - (gamma1 * gamma1 / 36.0) * (2.0 * z3 - 5.0 * z)
        )

        x = mu + sigma * cf
    if continuity:
        x += 0.5

    return np.maximum(x, 0.0)


# ---------------------------------------------------------------------------
# Primary scoring: WIS over case beam quantiles
# ---------------------------------------------------------------------------

def wis_score_vectorized(
    simulations_df: pd.DataFrame,
    observations_sr: pd.Series,
    alphas: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    weight_of_median: float = 0.5,
) -> np.ndarray:
    """Compute the Weighted Interval Score (WIS) for multiple simulations.

    Scores deterministic case beam quantiles (prediction intervals) against
    observed case counts.

    Precondition: All simulations must have the same sets of quantiles in `simulations_df`.

    Parameters
    ----------
    simulations_df:
        DataFrame with MultiIndex ``(quantile, i_simulation)`` and columns
        corresponding to time steps.  Must include the 0.5 (median) quantile.
    observations_sr:
        Observed case counts indexed by time step.  Must match
        ``simulations_df.columns``.
    alphas:
        Interval levels (e.g. ``[0.05, 0.5]`` for 95 % and 50 % PIs).
        Inferred from available quantiles when ``None``.
    weights:
        Per-interval weights.  Defaults to ``alphas / 2``.
    weight_of_median:
        Weight assigned to the median absolute error component.

    Returns
    -------
    np.ndarray
        Shape ``(num_simulations, num_time_steps)``.
    """
    assert simulations_df.columns.equals(observations_sr.index), (
        "Time steps in simulations and observations must match"
    )

    available_q = simulations_df.index.get_level_values("quantile").unique().values
    available_q.sort()

    assert 0.5 in available_q, "Median (0.5) quantile must be present for WIS calculation"

    if alphas is None:
        use_q = [q for q in available_q if q < 0.5]
        alphas = np.array([2 * q for q in use_q])

    if weights is None:
        weights = alphas / 2.0

    obs_vec = observations_sr.values[np.newaxis, :]  # (1, num_time_steps)

    wis_components = []
    for alpha in alphas:
        q_low = alpha / 2.0
        q_high = 1.0 - alpha / 2.0

        if q_low not in available_q or q_high not in available_q:
            raise ValueError(
                f"Quantiles {q_low} and {q_high} must be present in simulations_df "
                f"for alpha={alpha}"
            )

        pred_low = simulations_df.xs(q_low, level="quantile").values
        pred_high = simulations_df.xs(q_high, level="quantile").values
        # Shape: (num_simulations, num_time_steps)

        sharpness = pred_high - pred_low
        calibration = (2.0 / alpha) * (
            np.maximum(0, obs_vec - pred_high) + np.maximum(0, pred_low - obs_vec)
        )
        wis_components.append(sharpness + calibration)

    pred_median = simulations_df.xs(0.5, level="quantile").values
    wis_median = np.abs(pred_median - obs_vec)  # (num_simulations, num_time_steps)

    wis = wis_median * weight_of_median
    for i, alpha in enumerate(alphas):
        wis += wis_components[i] * weights[i]

    wis /= len(alphas) + 0.5

    return wis


def coverages_vectorized(
    simulations_df: pd.DataFrame,
    observations_sr: pd.Series,
    alphas: np.ndarray | None = None,
    calculate_coverage_likelihood: bool = True,
) -> pd.DataFrame:
    """Compute empirical coverage of case beam quantiles.

    Parameters
    ----------
    simulations_df:
        DataFrame with MultiIndex ``(quantile, i_simulation)`` and columns
        corresponding to time steps.  Must include the 0.5 (median) quantile.
    observations_sr:
        Observed case counts indexed by time step.  Must match ``simulations_df.columns``.
    alphas:
        Prediction interval levels to evaluate coverages for.
        Defaults to  ``[0.05, 0.5]`` for 95% and 50% PIs.

    """
    assert simulations_df.columns.equals(observations_sr.index), (
        "Time steps in simulations and observations must match"
    )

    alphas = alphas or np.array([0.05, 0.5])

    available_q = simulations_df.index.get_level_values("quantile").unique().values
    available_q.sort()

    # ==== EXPERIMENTAL: Filter point to calculate for
    # Based on observations
    if False:
        max_obs = observations_sr.max()
        thresh = 0.05 * max_obs
        min_points = 10

        # Find time steps where observed counts are above threshold
        # or there are at least min_points (with highest values)
        above_mask: pd.Series = observations_sr >= thresh
        if above_mask.sum() < min_points:
            # If not enough points above threshold, take the top min_points
            top_indices = observations_sr.nlargest(min_points).index
            mask = observations_sr.index.isin(top_indices)
        else:
            mask = above_mask
        # Apply filter
        observations_sr = observations_sr[mask]
        simulations_df = simulations_df.loc[:, mask]


    # Greater-than and less-than masks
    sim_le_obs = simulations_df.T.le(observations_sr, axis=0).T
    sim_ge_obs = simulations_df.T.ge(observations_sr, axis=0).T

    sr_list = list()
    for alpha in alphas:
        q_low = alpha / 2.0
        q_high = 1.0 - alpha / 2.0
        pi_width = 1.0 - alpha

        if q_low not in available_q or q_high not in available_q:
            raise ValueError(
                f"Quantiles {q_low} and {q_high} must be present in simulations_df "
                f"for alpha={alpha}"
            )

        low_mask = sim_le_obs.xs(q_low, level="quantile")
        high_mask = sim_ge_obs.xs(q_high, level="quantile")

        # In this combine operation, quantiles must match
        sr = (low_mask & high_mask).mean(axis=1)
        sr.name = f"coverage_{pi_width:0.3f}"

        sr_list.append(sr)

    coverages_df = pd.concat(sr_list, axis=1)

    # ----
    if calculate_coverage_likelihood:
        num_obs = simulations_df.shape[1]
        coverage_ll_sr = pd.Series(
            0., index=coverages_df.index, name="coverage_loglikelihood"
        )

        for alpha in alphas:
            pi_width = 1.0 - alpha
            coverage_col = f"coverage_{pi_width:0.3f}"
            # Recover from mean to total number of covered points
            covered_sr = coverages_df[coverage_col] * num_obs

            # Calculate loglikelihood for this pi_width
            coverage_ll_sr += scipy.stats.binom.logpmf(
                k=covered_sr.astype(int),
                n=num_obs,
                p=pi_width,
            )

        coverages_df = pd.concat([coverages_df, coverage_ll_sr], axis=1)

    return coverages_df


# ---------------------------------------------------------------------------
# Individual trajectory metrics
# ---------------------------------------------------------------------------

def rmse_vectorized(
    simulations_df: pd.DataFrame,
    observations_sr: pd.Series,
) -> np.ndarray:
    """Root Mean Square Error for individual case trajectories.

    Parameters
    ----------
    simulations_df:
        DataFrame of shape ``(num_simulations, num_time_steps)``.
    observations_sr:
        Observed counts indexed by time step.

    Returns
    -------
    np.ndarray
        Shape ``(num_simulations,)``.
    """
    obs = observations_sr.reindex(simulations_df.columns).values  # (num_time_steps,)
    residuals = simulations_df.values - obs[np.newaxis, :]        # (num_simulations, num_time_steps)
    return np.sqrt(np.mean(residuals ** 2, axis=1))


def smape_vectorized(
    simulations_df: pd.DataFrame,
    observations_sr: pd.Series,
) -> np.ndarray:
    """Symmetric Mean Absolute Percentage Error for individual trajectories.

    Parameters
    ----------
    simulations_df:
        DataFrame of shape ``(num_simulations, num_time_steps)``.
    observations_sr:
        Observed counts indexed by time step.

    Returns
    -------
    np.ndarray
        Shape ``(num_simulations,)``.
    """
    obs = observations_sr.reindex(simulations_df.columns).values  # (num_time_steps,)
    sim = simulations_df.values                                   # (num_simulations, num_time_steps)
    denom = np.abs(sim) + np.abs(obs[np.newaxis, :])
    numerator = 2.0 * np.abs(sim - obs[np.newaxis, :])
    # Where both sim and obs are zero, the term is 0/0 → treat as 0 error.
    # Use errstate to suppress the spurious warning numpy raises before the
    # np.where mask is applied.
    with np.errstate(invalid="ignore"):
        ratio = np.where(denom == 0, 0.0, numerator / denom)
    return np.mean(ratio, axis=1)


def nb_loglikelihood_vectorized(
    simulations_df: pd.DataFrame,
    observations_sr: pd.Series,
    overdisp: np.ndarray,
) -> np.ndarray:
    """Negative Binomial log-likelihood for individual trajectories.

    Parameters
    ----------
    simulations_df:
        DataFrame of shape ``(num_simulations, num_time_steps)``.)
    observations_sr:
        Observed counts indexed by time step.
    overdisp:
        Array of shape ``(num_simulations,)`` with the negative binomial
        overdispersion ("n" parameter) to consider for each simulation.

    Returns
    -------
    np.ndarray
        Shape ``(num_simulations,)`` with the total log-likelihood of each simulation
        trajectory under a Negative Binomial model with mean given by the
        simulation and overdispersion given by `overdisp`.
    """
    obs = observations_sr.reindex(simulations_df.columns).values  # (num_time_steps,)
    pred = simulations_df.values            # (num_simulations, num_time_steps)
    n = overdisp[:, np.newaxis]             # (num_simulations, 1)

    with np.errstate(invalid="ignore"):
        ll_array = scipy.stats.nbinom.logpmf(
            k=obs[np.newaxis, :],
            n=n,
            p=n / (pred + n)
        )
        # Shape: (num_simulations, num_time_steps)

    return np.sum(ll_array, axis=1)  # Sum over time steps
