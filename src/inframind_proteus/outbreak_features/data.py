"""Spatial/temporal alignment & aggregation. Builds the weekly panel and decodes folds."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, vintage

_FOLD_COLS = [f"{p}_{k}" for k in range(1, 5) for p in ("train", "target")]


class DataRepository:
    """Loads dengue + population, aggregates to a spatial level, exposes the weekly panel.

    The weekly panel has one row per (unit, epiweek) with: casos, population, incidence,
    season, week_in_season, and the IMDC train_k/target_k flags.
    """

    def __init__(self, spatial_level: str = "state", data_dir=config.DATA_DIR):
        if spatial_level not in config.SPATIAL_LEVELS:
            raise ValueError(f"unknown spatial_level {spatial_level!r}")
        self.spatial_level = spatial_level
        self.unit_col = config.SPATIAL_LEVELS[spatial_level]
        self.data_dir = data_dir
        self._panel: pd.DataFrame | None = None

    # ---- loading -----------------------------------------------------------
    def _load_dengue(self) -> pd.DataFrame:
        usecols = ["casos", "epiweek", self.unit_col] + _FOLD_COLS
        dtypes = {"casos": "int32", "epiweek": "int32", self.unit_col: "int32"}
        df = vintage.read_dengue(usecols=usecols, dtype=dtypes)
        for c in _FOLD_COLS:
            df[c] = df[c].astype(str).eq("True")
        return df

    def _load_population(self) -> pd.DataFrame:
        pop = vintage.read_population(
            dtype={"geocode": "int32", "year": "int32", "population": "int64"}
        )
        xwalk = pd.read_csv(
            config.CROSSWALK_FILE, usecols=["geocode", self.unit_col]
        ).astype({"geocode": "int32", self.unit_col: "int32"})
        xwalk = xwalk.drop_duplicates("geocode")
        pop = pop.merge(xwalk, on="geocode", how="inner")
        return pop.groupby([self.unit_col, "year"], as_index=False)["population"].sum()

    # ---- panel -------------------------------------------------------------
    def _build_panel(self) -> pd.DataFrame:
        df = self._load_dengue()
        df["year"] = (df["epiweek"] // 100).astype("int32")
        wk = df["epiweek"] % 100
        df["season"] = np.where(wk >= config.SEASON_START_WEEK, df["year"], df["year"] - 1).astype("int32")

        unit = self.unit_col
        agg_spec = {"casos": ("casos", "sum"), "year": ("year", "first"), "season": ("season", "first")}
        agg_spec.update({c: (c, "first") for c in _FOLD_COLS})
        panel = df.groupby([unit, "epiweek"], sort=False).agg(**agg_spec).reset_index()

        pop = self._load_population()
        panel = panel.merge(pop, on=[unit, "year"], how="left")
        panel["incidence"] = panel["casos"] / panel["population"] * config.INCIDENCE_SCALE

        panel = panel.sort_values([unit, "season", "epiweek"]).reset_index(drop=True)
        panel["week_in_season"] = panel.groupby([unit, "season"]).cumcount() + 1
        self._panel = panel
        return panel

    def panel(self) -> pd.DataFrame:
        if self._panel is None:
            self._build_panel()
        return self._panel

    def population_by_unit_year(self) -> pd.Series:
        """(unit, year) -> population (one figure per calendar year)."""
        panel = self.panel()
        return panel.groupby([self.unit_col, "year"])["population"].first()

    # ---- folds -------------------------------------------------------------
    def folds(self) -> dict[int, dict]:
        """Decode the IMDC rolling-origin folds from the train_k/target_k flags."""
        panel = self.panel()
        out: dict[int, dict] = {}
        for k in range(1, 5):
            tr, tg = f"train_{k}", f"target_{k}"
            if not panel[tg].any():
                continue
            target_season = int(panel.loc[panel[tg], "season"].mode().iloc[0])
            out[k] = dict(
                fold=k,
                target_season=target_season,
                issue_epiweek=int(panel.loc[panel[tr], "epiweek"].max()),
                train_seasons=sorted(panel.loc[panel[tr], "season"].unique().tolist()),
            )
        return out
