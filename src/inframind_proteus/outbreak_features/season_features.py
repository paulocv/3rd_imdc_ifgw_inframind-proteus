"""Season-grain feature pipelines + the FeatureAssembler.

Produces one feature vector per (unit, season) for the tabular direct-regression track.
Every row is **season-intrinsic and leakage-safe**: features for season s use only data
<= t0(s) = EW25 of s's start year and labels of seasons strictly < s, so the same matrix
serves every fold (the fold only decides which rows are train vs target). The blocks are
epidemiological history (p5), static environment/demography (p6), and the climate/ocean
blocks (p2 observed ERA5, p3 Copernicus forecast, p4 ocean teleconnections).

Feature naming: pNN__group__name[__stat].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, vintage
from .data import DataRepository
from .features import build_climate_panel, build_forecast_panel


def season_t0(season: int) -> int:
    """Issue-point epiweek for a season: EW25 of its start year (YYYYWW)."""
    return season * 100 + config.SEASON_ISSUE_WEEK


def _ols_slope(y: np.ndarray) -> float:
    """Least-squares slope of y over its (NaN-dropped) order; NaN if < 2 points."""
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]
    if len(y) < 2:
        return np.nan
    return float(np.polyfit(np.arange(len(y)), y, 1)[0])


# ---- P6: static environment & demography -----------------------------------
def _static_env_features(repo: DataRepository) -> tuple[pd.DataFrame, list[str], dict]:
    """Population-weighted koppen/biome composition + dominant class per unit (static)."""
    unit_col = repo.unit_col
    env = pd.read_csv(config.ENVIRON_FILE, usecols=["geocode", "koppen", "biome"],
                      dtype={"geocode": "int32"})
    xwalk = (pd.read_csv(config.CROSSWALK_FILE, usecols=["geocode", unit_col])
             .astype({"geocode": "int32", unit_col: "int32"}).drop_duplicates("geocode"))
    pop = vintage.read_population(dtype={"geocode": "int32", "year": "int32",
                                         "population": "int64"})
    pop = pop.sort_values("year").groupby("geocode", as_index=False)["population"].last()  # latest weight
    df = env.merge(xwalk, on="geocode", how="inner").merge(pop, on="geocode", how="left")
    df["population"] = df["population"].fillna(1.0)

    blocks, describe = [], {}
    for grp in ("koppen", "biome"):
        w = df.groupby([unit_col, grp])["population"].sum()
        share = (w / w.groupby(level=0).sum()).unstack(fill_value=0.0)
        share.columns = [f"p6__env__{grp}__share__{c}" for c in share.columns]
        dom = w.groupby(level=0).idxmax().map(lambda t: t[1]).rename(f"p6__env__{grp}__dom")
        for c in share.columns:
            describe[c] = f"population-fraction of {grp} class {c.split('__')[-1]} in the unit"
        describe[dom.name] = f"dominant (pop-weighted) {grp} class of the unit"
        blocks += [share, dom]
    out = pd.concat(blocks, axis=1)
    cat_cols = [f"p6__env__{g}__dom" for g in ("koppen", "biome")]
    return out, cat_cols, describe


# ---- P5: chikungunya cross-disease weekly incidence ------------------------
def _chik_weekly(repo: DataRepository) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """unit -> (sorted epiweeks, chikungunya incidence /100k). Empty dict if unavailable."""
    unit_col = repo.unit_col
    if not config.CHIK_FILE.exists():
        return {}
    df = pd.read_csv(config.CHIK_FILE, usecols=["casos", "epiweek", unit_col],
                     dtype={"casos": "int32", "epiweek": "int32", unit_col: "int32"})
    df["year"] = (df["epiweek"] // 100).astype("int32")
    g = df.groupby([unit_col, "epiweek"], sort=False).agg(
        casos=("casos", "sum"), year=("year", "first")).reset_index()
    g = g.merge(repo._load_population(), on=[unit_col, "year"], how="left")
    g["incidence"] = g["casos"] / g["population"] * config.INCIDENCE_SCALE
    return {u: (s.sort_values("epiweek")["epiweek"].to_numpy(),
               s.sort_values("epiweek")["incidence"].to_numpy(float))
            for u, s in g.groupby(unit_col)}


def _recent(ews: np.ndarray, vals: np.ndarray, t0: int, k: int) -> tuple[float, float, float]:
    """(sum, mean, max) of the k most-recent values at epiweeks <= t0; NaN if none."""
    sel = vals[ews <= t0]
    sel = sel[-k:]
    sel = sel[~np.isnan(sel)]
    if len(sel) == 0:
        return (np.nan, np.nan, np.nan)
    return (float(sel.sum()), float(sel.mean()), float(sel.max()))


# ---- P2: observed ERA5 pre-season summaries --------------------------------
def _p2_climate(repo: DataRepository, lab_u: dict) -> tuple[pd.DataFrame, dict]:
    """Per (unit, season) ERA5 window stats (mean/trend/anomaly), all weeks <= t0(s)."""
    clim = build_climate_panel(repo.unit_col, repo.spatial_level)
    vars_ = config.CLIMATE_VARS
    wins = config.P2_PRESEASON_WINDOWS
    by_unit = {u: g.sort_values("epiweek") for u, g in clim.groupby(repo.unit_col)}

    rows, describe = [], {}
    for u, lg in lab_u.items():
        cg = by_unit.get(u)
        if cg is None:
            continue
        ew = cg["epiweek"].to_numpy(); woy = cg["woy"].to_numpy()
        yr = cg["year"].to_numpy(); mo = cg["month"].to_numpy()
        V = {v: cg[v].to_numpy(float) for v in vars_}
        for s in lg["season"].to_numpy():
            s = int(s); t0 = season_t0(s)
            obs = np.where(ew <= t0)[0]
            if len(obs) == 0:
                rows.append({"unit": u, "season": s}); continue
            clim_woy = {v: pd.Series(V[v][obs]).groupby(woy[obs]).mean() for v in vars_}
            windows = {f"pre{w}w": obs[-w:] for w in wins}
            windows["wet"] = obs[((yr[obs] == s - 1) & (mo[obs] == 12))
                                 | ((yr[obs] == s) & (mo[obs] <= 5))]
            row = {"unit": u, "season": s}
            for wn, idx in windows.items():
                for v in vars_:
                    vals = V[v][idx]
                    row[f"p2__clim__{v}__mean__{wn}"] = float(np.nanmean(vals)) if len(vals) else np.nan
                    row[f"p2__clim__{v}__trend__{wn}"] = _ols_slope(vals)
                    anom = vals - clim_woy[v].reindex(woy[idx]).to_numpy() if len(vals) else np.array([])
                    row[f"p2__clim__{v}__anom__{wn}"] = float(np.nanmean(anom)) if len(anom) else np.nan
            rows.append(row)
    win_desc = {**{f"pre{w}w": f"the {w} weeks before t0" for w in wins},
                "wet": "the previous wet season (Dec-May before t0)"}
    for wn, wd in win_desc.items():
        for v in vars_:
            describe[f"p2__clim__{v}__mean__{wn}"] = f"mean {v} over {wd}"
            describe[f"p2__clim__{v}__trend__{wn}"] = f"OLS trend of {v} over {wd}"
            describe[f"p2__clim__{v}__anom__{wn}"] = f"{v} anomaly vs unit woy-climatology over {wd}"
    return pd.DataFrame(rows), describe


# ---- P3: Copernicus seasonal-forecast features -----------------------------
def _p3_forecast(repo: DataRepository, lab_u: dict) -> tuple[pd.DataFrame, dict]:
    """Per (unit, season) forecast issued at t0 (ref_month <= June of year s)."""
    fc = build_forecast_panel(repo.unit_col, repo.spatial_level)
    fvars = ("temp_med", "umid_med")
    im = config.SEASON_ISSUE_FCST_MONTH
    by_unit = {u: g for u, g in fc.groupby(repo.unit_col)}

    rows, describe = [], {}
    for u, lg in lab_u.items():
        fg = by_unit.get(u)
        for s in lg["season"].to_numpy():
            s = int(s); row = {"unit": u, "season": s}
            if fg is not None:
                issue = fg[(fg["ref_year"] == s) & (fg["ref_month"] <= im)]
                if len(issue):
                    cur = issue[issue["ref_month"] == issue["ref_month"].max()].sort_values("forecast_months_ahead")
                    src = fg[(fg["ref_year"] < s) | ((fg["ref_year"] == s) & (fg["ref_month"] <= im))]
                    for v in fvars:
                        vals = cur[v].to_numpy(float)
                        clim = src.groupby("target_month")[v].mean()
                        anom = vals - clim.reindex(cur["target_month"]).to_numpy()
                        row[f"p3__fcst__{v}__mean"] = float(np.nanmean(vals)) if len(vals) else np.nan
                        row[f"p3__fcst__{v}__max"] = float(np.nanmax(vals)) if len(vals) else np.nan
                        row[f"p3__fcst__{v}__trend"] = _ols_slope(vals)
                        row[f"p3__fcst__{v}__anom"] = float(np.nanmean(anom)) if len(anom) else np.nan
            rows.append(row)
    for v in fvars:
        describe[f"p3__fcst__{v}__mean"] = f"mean forecast {v} over the issued horizons (Jul-Dec)"
        describe[f"p3__fcst__{v}__max"] = f"max forecast {v} over the issued horizons"
        describe[f"p3__fcst__{v}__trend"] = f"trend of forecast {v} across horizon"
        describe[f"p3__fcst__{v}__anom"] = f"forecast {v} anomaly vs the per-month forecast climatology"
    return pd.DataFrame(rows), describe


# ---- P4: ocean teleconnections (ENSO/IOD/PDO; global -> by season) ----------
def _file_fingerprint(*paths) -> str:
    """Short hash of (path, size, mtime_ns) for each input -> content-aware cache key without
    reading 480 MB. Changing either source file invalidates the cache (no stale-cache risk)."""
    import hashlib
    h = hashlib.sha1()
    for p in paths:
        st = p.stat()
        h.update(f"{p}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()[:12]


def _ocean_by_epiweek() -> pd.DataFrame:
    """Weekly ENSO/IOD/PDO aligned to epiweek (date asof-merged to the dengue grid). Cached,
    keyed on a fingerprint of the source files so an upstream data change invalidates it."""
    fp = _file_fingerprint(config.OCEAN_FILE, config.DENGUE_FILE,
                           *[f for f in (config.OCEAN_UPDATE_FILES
                                         + config.DENGUE_UPDATE_FILES
                                         + [config.OCEAN_REFRESHED_FILE]) if f.exists()])
    cache = config.CACHE_DIR / f"ocean_epiweek_{fp}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    o = vintage.read_ocean(parse_dates=["date"]).sort_values("date")
    m = vintage.read_dengue(usecols=["date", "epiweek"], dtype={"epiweek": "int32"})
    m["date"] = pd.to_datetime(m["date"])
    m = m.drop_duplicates("date").sort_values("date")
    o = pd.merge_asof(o, m, on="date", direction="backward").dropna(subset=["epiweek"])
    o["epiweek"] = o["epiweek"].astype(int)
    out = o.groupby("epiweek")[["enso", "iod", "pdo"]].mean().reset_index().sort_values("epiweek")
    config.CACHE_DIR.mkdir(exist_ok=True)
    out.to_pickle(cache)
    return out


def _p4_ocean(all_seasons: list[int]) -> tuple[pd.DataFrame, dict]:
    """Per-season ocean features (same for every unit at a given season)."""
    ob = _ocean_by_epiweek()
    ew = ob["epiweek"].to_numpy()
    rows = []
    for s in all_seasons:
        t0 = season_t0(int(s)); m = ew <= t0
        row = {"season": int(s)}
        for v in ("enso", "iod", "pdo"):
            vv = ob[v].to_numpy()[m]
            row[f"p4__ocean__{v}__level"] = float(vv[-1]) if len(vv) else np.nan
            row[f"p4__ocean__{v}__mean26w"] = float(np.nanmean(vv[-26:])) if len(vv) else np.nan
            row[f"p4__ocean__{v}__mean52w"] = float(np.nanmean(vv[-52:])) if len(vv) else np.nan
            row[f"p4__ocean__{v}__trend26w"] = _ols_slope(vv[-26:])
        rows.append(row)
    describe = {}
    for v in ("enso", "iod", "pdo"):
        describe[f"p4__ocean__{v}__level"] = f"{v.upper()} value at t0"
        describe[f"p4__ocean__{v}__mean26w"] = f"mean {v.upper()} over the 26 weeks before t0"
        describe[f"p4__ocean__{v}__mean52w"] = f"mean {v.upper()} over the 52 weeks before t0"
        describe[f"p4__ocean__{v}__trend26w"] = f"trend of {v.upper()} over the 26 weeks before t0"
    return pd.DataFrame(rows), describe


# ---- assembly --------------------------------------------------------------
def build_season_features(repo: DataRepository,
                          labels: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    """Return (features indexed by (unit, season), categorical columns, data dictionary)."""
    unit_col = repo.unit_col
    panel = repo.panel()
    describe: dict[str, str] = {}
    lags = config.P5_TARGET_LAGS
    k = config.P5_PRESEASON_WEEKS

    # ---- P5a: lagged targets (vectorized self-join on shifted season) ----
    base = labels[["unit", "season"]].copy()
    feat = base.copy()
    for L in lags:
        src = labels[["unit", "season"] + config.TARGETS].copy()
        src["season"] = src["season"] + L          # this past row supplies features for season+L
        ren = {t: f"p5__epi__{t}__lag{L}" for t in config.TARGETS}
        for t, c in ren.items():
            describe[c] = f"{t} of the unit {L} season(s) earlier (autoregressive)"
        feat = feat.merge(src.rename(columns=ren), on=["unit", "season"], how="left")

    # ---- P5b/P5c: immunity proxies + pre-season activity (per-unit loop) ----
    pop_uy = repo.population_by_unit_year()
    panel_by_unit = {u: g.sort_values("epiweek") for u, g in panel.groupby(unit_col)}
    chik_by_unit = _chik_weekly(repo) if config.P5_INCLUDE_CHIK else {}
    lab_u = {u: g.sort_values("season") for u, g in labels.groupby("unit")}
    N = config.P5_IMMUNITY_WINDOW

    extra = []
    for u, lg in lab_u.items():
        seasons = lg["season"].to_numpy()
        attack = lg["size_attack_rate"].to_numpy(float)
        g = panel_by_unit.get(u)
        ews = g["epiweek"].to_numpy() if g is not None else np.array([])
        inc = g["incidence"].to_numpy(float) if g is not None else np.array([])
        cews, cinc = chik_by_unit.get(u, (np.array([]), np.array([])))
        for i, s in enumerate(seasons):
            prior = attack[:i]                       # strictly-prior seasons (seasons sorted asc)
            t0 = season_t0(int(s))
            r_sum, r_mean, r_max = _recent(ews, inc, t0, k)
            row = {
                "unit": u, "season": int(s),
                "p5__epi__immun_cumattack": float(np.nansum(prior[-N:])) if len(prior) else np.nan,
                "p5__epi__seasons_since_max": (
                    float(len(prior) - 1 - int(np.nanargmax(prior))) if np.any(~np.isnan(prior)) else np.nan),
                "p5__epi__preseason_inc_sum": r_sum,
                "p5__epi__preseason_inc_mean": r_mean,
                "p5__epi__preseason_inc_max": r_max,
                "p6__demo__log_pop": float(np.log1p(pop_uy.get((u, int(s)), np.nan))),
            }
            if chik_by_unit:
                cs, _, cmax = _recent(cews, cinc, t0, k)
                row["p5__epi__chik_preseason_inc_sum"] = cs
                row["p5__epi__chik_preseason_inc_max"] = cmax
            extra.append(row)
    extra = pd.DataFrame(extra)
    describe.update({
        "p5__epi__immun_cumattack": f"cumulative attack rate over the previous {N} seasons (immunity proxy)",
        "p5__epi__seasons_since_max": "seasons since the unit's largest prior attack rate",
        "p5__epi__preseason_inc_sum": f"sum of dengue incidence in the {k} weeks before t0 (pre-season tail)",
        "p5__epi__preseason_inc_mean": f"mean dengue incidence in the {k} weeks before t0",
        "p5__epi__preseason_inc_max": f"max dengue incidence in the {k} weeks before t0",
        "p6__demo__log_pop": "log1p population of the unit in the season's start year",
    })
    if chik_by_unit:
        describe["p5__epi__chik_preseason_inc_sum"] = f"sum of chikungunya incidence in the {k} weeks before t0"
        describe["p5__epi__chik_preseason_inc_max"] = f"max chikungunya incidence in the {k} weeks before t0"

    # ---- P6: static env/demography ----
    p6, cat_cols, p6_desc = _static_env_features(repo)
    describe.update(p6_desc)

    X = (feat.merge(extra, on=["unit", "season"], how="left")
             .merge(p6.reset_index().rename(columns={unit_col: "unit"}), on="unit", how="left"))

    # ---- P2/P3/P4: climate & ocean blocks (season-intrinsic, leakage-safe) ----
    if config.INCLUDE_P2_CLIMATE:
        p2, d = _p2_climate(repo, lab_u); describe.update(d)
        X = X.merge(p2, on=["unit", "season"], how="left")
    if config.INCLUDE_P3_FORECAST:
        p3, d = _p3_forecast(repo, lab_u); describe.update(d)
        X = X.merge(p3, on=["unit", "season"], how="left")
    if config.INCLUDE_P4_OCEAN:
        p4, d = _p4_ocean(sorted(labels["season"].unique().tolist())); describe.update(d)
        X = X.merge(p4, on="season", how="left")

    X = X.set_index(["unit", "season"]).sort_index()
    return X, cat_cols, describe


class FeatureAssembler:
    """Builds the season matrix once, then serves leakage-safe per-fold train/target blocks.

    The features are season-intrinsic (each row already uses only data <= its own t0 and
    labels < its own season), so `fit` is a near-no-op here; the fit/transform split exists
    so any future fold-aware transform has a place to hook in.
    """

    def __init__(self, repo: DataRepository, labels: pd.DataFrame):
        self.X, self.cat_features, self.describe = build_season_features(repo, labels)
        self._train_seasons: list[int] | None = None

    def fit(self, train_seasons: list[int]) -> "FeatureAssembler":
        self._train_seasons = list(train_seasons)
        return self

    def transform(self, seasons: list[int]) -> pd.DataFrame:
        idx = self.X.index.get_level_values("season").isin(seasons)
        return self.X.loc[idx]

    def feature_names(self) -> list[str]:
        return list(self.X.columns)
