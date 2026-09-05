"""Climate aggregation.

Builds population-weighted weekly ERA5 climate per spatial unit and population-weighted
Copernicus seasonal forecasts. These panels feed the season-matrix climate/forecast feature
blocks (P2/P3 in `season_features.py`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, vintage


# ---- climate / forecast aggregation ----------------------------------------
def _xwalk(unit_col: str) -> pd.DataFrame:
    return (pd.read_csv(config.CROSSWALK_FILE, usecols=["geocode", unit_col])
            .astype({"geocode": "int32", unit_col: "int32"}).drop_duplicates("geocode"))


def _pop() -> pd.DataFrame:
    return vintage.read_population(dtype={"geocode": "int32", "year": "int32", "population": "int64"})


def build_climate_panel(unit_col: str, spatial_level: str) -> pd.DataFrame:
    """Population-weighted weekly ERA5 climate per (unit, epiweek). Cached per level."""
    cache = config.CACHE_DIR / f"climate_{spatial_level}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)

    cols = ["epiweek", "geocode", "date"] + config.CLIMATE_VARS
    dt = {"epiweek": "int32", "geocode": "int32", **{v: "float32" for v in config.CLIMATE_VARS}}
    df = vintage.read_climate(usecols=cols, dtype=dt)
    df["year"] = (df["epiweek"] // 100).astype("int32")
    df["month"] = pd.to_datetime(df["date"]).dt.month.astype("int16")
    df = df.drop(columns=["date"]).merge(_xwalk(unit_col), on="geocode", how="inner")
    df = df.merge(_pop(), on=["geocode", "year"], how="left")
    df["population"] = df["population"].fillna(1.0)

    for v in config.CLIMATE_VARS:
        df[v] = df[v] * df["population"]
    agg = df.groupby([unit_col, "epiweek"], sort=False).agg(
        population=("population", "sum"), year=("year", "first"), month=("month", "first"),
        **{v: (v, "sum") for v in config.CLIMATE_VARS},
    ).reset_index()
    for v in config.CLIMATE_VARS:
        agg[v] = agg[v] / agg["population"]
    agg["woy"] = (agg["epiweek"] % 100).astype("int16")
    panel = agg[[unit_col, "epiweek", "year", "month", "woy"] + config.CLIMATE_VARS].copy()

    config.CACHE_DIR.mkdir(exist_ok=True)
    panel.to_pickle(cache)
    return panel


def build_forecast_panel(unit_col: str, spatial_level: str) -> pd.DataFrame:
    """Population-weighted Copernicus forecasts per (unit, ref, horizon, target month). Cached."""
    cache = config.CACHE_DIR / f"forecast_{spatial_level}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)

    df = vintage.read_forecast(
        usecols=["geocode", "reference_month", "forecast_months_ahead", "temp_med", "umid_med"],
        dtype={"geocode": "int32", "forecast_months_ahead": "int16",
               "temp_med": "float32", "umid_med": "float32"},
    )
    ref = pd.to_datetime(df["reference_month"], format="ISO8601")
    df["ref_year"] = ref.dt.year.astype("int32")
    df["ref_month"] = ref.dt.month.astype("int16")
    tot = ref.dt.year * 12 + (ref.dt.month - 1) + df["forecast_months_ahead"]
    df["target_year"] = (tot // 12).astype("int32")
    df["target_month"] = ((tot % 12) + 1).astype("int16")
    df = df.drop(columns=["reference_month"]).merge(_xwalk(unit_col), on="geocode", how="inner")
    pop = _pop().rename(columns={"year": "ref_year"})
    df = df.merge(pop, on=["geocode", "ref_year"], how="left")
    df["population"] = df["population"].fillna(1.0)

    for v in ("temp_med", "umid_med"):
        df[v] = df[v] * df["population"]
    keys = [unit_col, "ref_year", "ref_month", "forecast_months_ahead", "target_year", "target_month"]
    agg = df.groupby(keys, sort=False).agg(
        population=("population", "sum"), temp_med=("temp_med", "sum"), umid_med=("umid_med", "sum"),
    ).reset_index()
    agg["temp_med"] = agg["temp_med"] / agg["population"]
    agg["umid_med"] = agg["umid_med"] / agg["population"]
    panel = agg[keys + ["temp_med", "umid_med"]].copy()

    config.CACHE_DIR.mkdir(exist_ok=True)
    panel.to_pickle(cache)
    return panel
