"""
Likelihood of uncertain observations under a Gaussian-KDE predictive distribution.

Each observation is assumed as a Gaussian distribution, given its mean and
standard deviation.

Setup
-----
- The predictive distribution is a Gaussian KDE: an equally-weighted mixture of M
  Gaussians, one per sample in `outb_feats_predicted_samples`, all sharing the same
  bandwidth (std) `kde_bandwidth` = sqrt(kde.covariance[0, 0]).
- Each observation is itself uncertain: a Gaussian with its own mean and std, given
  by the 'mean' and 'std' columns of `out_feats_avail_samples_df`.

For a single predictive component j and observation i, the "evidence" of the
observation under that component (marginalizing over the observation's own
uncertainty) is the convolution of two Gaussians, which is itself Gaussian:

    N(mu_obs_i; mu_pred_j, sigma_kde^2 + sigma_obs_i^2)

Since every predictive sample is equally likely, the overall likelihood is the
mean of this quantity over all M predictive samples:

    L_i = (1/M) * sum_j N(mu_obs_i; mu_pred_j, sigma_kde^2 + sigma_obs_i^2)
"""

import numpy as np
import pandas as pd
from scipy.special import logsumexp

LOG_2PI = np.log(2 * np.pi)


def kde_gaussian_log_likelihood(
    outb_feats_predicted_samples: pd.Series,
    out_feats_avail_samples_df: pd.DataFrame,
    kde_bandwidth: float,
    chunk_size: int = 5000,
) -> np.ndarray:
    """
    Compute log-likelihood of each (uncertain) observation under the KDE mixture.

    Parameters
    ----------
    outb_feats_predicted_samples : pd.Series, length M
        Means of the predictive distribution's Gaussian components (i.e. the
        points the KDE was fit on).
    out_feats_avail_samples_df : pd.DataFrame, length N
        Must have columns 'mean' and 'std' describing each observation's own
        Gaussian uncertainty.
    kde_bandwidth : float
        The (shared) std of each KDE component, e.g. sqrt(kde.covariance[0, 0]).
    chunk_size : int
        Number of observations processed per batch, to bound peak memory
        (peak usage per chunk is roughly M * chunk_size * 8 bytes).

    Returns
    -------
    np.ndarray, shape (N,)
        Natural-log likelihood for each observation. Use np.exp() only if you
        need the raw probability density and are confident it won't underflow
        (it easily can with many components / far-off observations) — for
        downstream use (comparisons, summing across observations, etc.),
        prefer staying in log space.
    """
    mu_pred = np.asarray(outb_feats_predicted_samples, dtype=np.float64)  # (M,)
    mu_obs = out_feats_avail_samples_df["mean"].to_numpy(dtype=np.float64)  # (N,)
    std_obs = out_feats_avail_samples_df["std"].to_numpy(dtype=np.float64)  # (N,)

    M = mu_pred.shape[0]
    N = mu_obs.shape[0]
    sigma1_sq = kde_bandwidth**2
    log_M = np.log(M)

    log_likelihoods = np.empty(N, dtype=np.float64)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        mu_obs_chunk = mu_obs[start:end]      # (n,)
        std_obs_chunk = std_obs[start:end]    # (n,)

        eff_var = sigma1_sq + std_obs_chunk**2          # (n,)
        diff = mu_pred[:, None] - mu_obs_chunk[None, :]  # (M, n)

        log_pdf = -0.5 * (LOG_2PI + np.log(eff_var)[None, :] + diff**2 / eff_var[None, :])

        # log( (1/M) * sum_j pdf_j ) = logsumexp(log_pdf, axis=0) - log(M)
        log_likelihoods[start:end] = logsumexp(log_pdf, axis=0) - log_M

    return log_likelihoods


def kde_gaussian_likelihood(
    outb_feats_predicted_samples: pd.Series,
    out_feats_avail_samples_df: pd.DataFrame,
    kde_bandwidth: float,
    chunk_size: int = 5000,
) -> np.ndarray:
    """Convenience wrapper returning likelihoods (not log-likelihoods)."""
    return np.exp(
        kde_gaussian_log_likelihood(
            outb_feats_predicted_samples,
            out_feats_avail_samples_df,
            kde_bandwidth,
            chunk_size
        )
    )


if __name__ == "__main__":
    # Minimal smoke test / usage example
    rng = np.random.default_rng(0)

    M, N = 500, 80_000
    predicted = pd.Series(rng.normal(loc=0.0, scale=1.0, size=M))
    obs_df = pd.DataFrame(
        {
            "mean": rng.normal(loc=0.0, scale=1.0, size=N),
            "std": np.abs(rng.normal(loc=0.1, scale=0.05, size=N)),
        }
    )
    bandwidth = 0.15  # e.g. from sqrt(kde.covariance[0, 0])

    log_lik = kde_gaussian_log_likelihood(predicted, obs_df, bandwidth)
    print("log-likelihood stats:", log_lik.min(), log_lik.mean(), log_lik.max())