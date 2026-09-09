"""Test alternatives to the Cornish-Fisher approximation of a negative
binomial PPF
"""

import numpy as np

import scipy
import scipy.stats

import plotly
import plotly.graph_objects as go
import plotly.express as px

from inframind_proteus.outbreak_dynamics.scoring import nbinom_ppf_cf
from scipy.interpolate import RegularGridInterpolator



# Goal: Test an alternative approximation of the NB PPF, a 2D linear interpolation from exact values.
# ============
quantiles = [0.025, 0.25, 0.50, 0.75, 0.975]

expectancy_grid = np.linspace(0., 50., num=200)
overdisp_grid = np.linspace(0.1, 5.0, num=200)

# Create 2D grids
_overdisp_2d = overdisp_grid[None, :]  # Shape: (1, len(overdisp_grid))
_expec_2d = expectancy_grid[:, None]   # Shape: (len(expectancy_grid), 1)

# Convert overdispersion and expectancy to n and p parameters for scipy.stats.nbinom
# Convention: p = overdisp / (overdisp + expec),  n = overdisp
# Check: mean = n*(1-p)/p = overdisp * (expec/(overdisp+expec)) / (overdisp/(overdisp+expec)) = expec ✓

_p_2d = _overdisp_2d / (_overdisp_2d + _expec_2d)   # Shape: (len(expectancy_grid), len(overdisp_grid))
_n_2d = _overdisp_2d * np.ones_like(_expec_2d)       # Shape: (len(expectancy_grid), len(overdisp_grid))

# Build lookup tables for each quantile
lookup_tables = {}

for q in quantiles:
    # Calculate exact PPF values on the grid
    ppf_values = scipy.stats.nbinom.ppf(
        q=q, n=_n_2d, p=_p_2d
    )
    lookup_tables[q] = ppf_values


def nbinom_ppf_interpolated(q, n, p):
    """
    Interpolated negative binomial PPF using precomputed lookup tables.

    This function provides the same signature as scipy.stats.nbinom.ppf:

    Parameters
    ----------
    q : float or array-like
        Quantile(s) in (0, 1).
    n : array-like
        Number of successes parameter (> 0).
        Converted from expectancy: expectancy = n(1-p)/p
    p : array-like
        Success probability parameter in (0, 1).
        Related to overdispersion: p = 1/overdispersion

    Returns
    -------
    ppf : ndarray
        Interpolated quantile values.
    """
    # Handle scalar inputs
    n_scalar = np.isscalar(n)
    p_scalar = np.isscalar(p)
    q_scalar = np.isscalar(q)

    n = np.atleast_1d(np.asarray(n, dtype=np.float64))
    p = np.atleast_1d(np.asarray(p, dtype=np.float64))

    # Convert (n, p) back to (expectancy, overdispersion)
    # Convention: n = overdisp,  p = overdisp / (overdisp + expec)
    # => expectancy = n*(1-p)/p
    # => overdispersion = n
    expectancy = n * (1.0 - p) / p
    overdispersion = n

    result = np.zeros_like(expectancy, dtype=np.float64)

    # For each unique quantile value (or handle as single), interpolate
    if not q_scalar:
        q_arr = np.atleast_1d(q)
        raise ValueError("q must be a scalar for this interpolation function")

    q = float(q)

    # Find nearest quantiles for linear interpolation between quantiles
    quantiles_sorted = sorted(quantiles)
    if q < quantiles_sorted[0] or q > quantiles_sorted[-1]:
        raise ValueError(f"Quantile q={q} is outside the supported range [{quantiles_sorted[0]}, {quantiles_sorted[-1]}]")

    # Find the two surrounding quantiles
    idx_upper = next(i for i, qt in enumerate(quantiles_sorted) if qt >= q)

    if quantiles_sorted[idx_upper] == q:
        # Exact quantile match
        ppf_grid = lookup_tables[q]
        interp = RegularGridInterpolator(
            (expectancy_grid, overdisp_grid),
            ppf_grid,
            bounds_error=False,
            fill_value=np.nan
        )
        points = np.column_stack([expectancy.ravel(), overdispersion.ravel()])
        result = interp(points).reshape(expectancy.shape)
    else:
        # Linear interpolation between two quantiles
        idx_lower = idx_upper - 1
        q_lower = quantiles_sorted[idx_lower]
        q_upper = quantiles_sorted[idx_upper]

        ppf_lower = lookup_tables[q_lower]
        ppf_upper = lookup_tables[q_upper]

        interp_lower = RegularGridInterpolator(
            (expectancy_grid, overdisp_grid),
            ppf_lower,
            bounds_error=False,
            fill_value=np.nan
        )
        interp_upper = RegularGridInterpolator(
            (expectancy_grid, overdisp_grid),
            ppf_upper,
            bounds_error=False,
            fill_value=np.nan
        )

        points = np.column_stack([expectancy.ravel(), overdispersion.ravel()])

        ppf_at_lower = interp_lower(points)
        ppf_at_upper = interp_upper(points)

        # Linear interpolation between the two PPF values
        weight_upper = (q - q_lower) / (q_upper - q_lower)
        weight_lower = 1.0 - weight_upper
        result = (weight_lower * ppf_at_lower + weight_upper * ppf_at_upper).reshape(expectancy.shape)

    # Return scalar if input was scalar
    if n_scalar and p_scalar:
        return float(result.flat[0])

    return result


print("Lookup tables created for quantiles:", quantiles)
# print("Expectancy grid: 0 to 50 (100 points)")
# print("Overdispersion grid: 0.1 to 5.0 (100 points)")
print(f"\nInterpolation function 'nbinom_ppf_interpolated' is ready to use.")
print(f"Signature: nbinom_ppf_interpolated(q, n, p)")
print(f"  q: scalar quantile in {quantiles[0]} to {quantiles[-1]}")
print(f"  n, p: arrays with same shape (scipy.stats.nbinom parameters)")


test_overdisp_grid = np.linspace(0.1, 4.8, num=85)
test_expectancy_grid = np.linspace(0.0, 49.0, num=85)

_expec_2d, _overdisp_2d = np.meshgrid(
    test_expectancy_grid, test_overdisp_grid,
    indexing="xy"
    # indexing="ij"
)
_p_2d = _overdisp_2d / (_overdisp_2d + _expec_2d)

# --- Calculate PPF exactly and approximated
for q in [0.5, 0.025, 0.25, 0.75, 0.975]:
    # q = 0.25
    exact_ppf = scipy.stats.nbinom.ppf(
        q=q, n=_overdisp_2d, p=_p_2d
    )
    approx_ppf = nbinom_ppf_interpolated(
        q=q, n=_overdisp_2d, p=_p_2d
    )

    # Approximation ratio: Approx / exact
    fig = px.imshow(
        approx_ppf / exact_ppf,
        x=test_expectancy_grid,
        y=test_overdisp_grid,
        labels={"x": "Expectancy", "y": "Overdispersion"},
        aspect="auto",
        title=f"NB {q=}: Approx / Exact",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=1.,
        range_color=[0, 2],
    )

    fig.show()
print()