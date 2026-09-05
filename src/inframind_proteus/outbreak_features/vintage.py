"""Data-vintage assembly: base IMDC snapshot + the `*_update_2026` files from the FTP.

For the 2026-2027 forecast phase the organizers publish the EW01-EW25 2026 extension as
separate `*_update_2026.csv.gz` files rather than reissuing the base files. The updates are
NOT purely additive: they overlap the base snapshot (dengue to EW202610, climate to EW202611)
and *revise* it -- the overlapping dengue weeks drop ~9.6% in total cases -- so on overlap the
newer vintage wins (`keep="last"` on the natural key).

The Copernicus update also renames one column (`umid_med` -> `rel_umid_med`); it is mapped
back to the base name here so downstream code sees one schema.

Population: DATASUS only publishes through 2025, so the 2026 season's rate denominators are
carried forward from each unit's last observed year. Rates are per 100k, so the effect of a
one-year-stale denominator is well under a percent, but it must be explicit rather than a
silent NaN (a NaN denominator would blank the entire pre-season run-up to t0 = EW25 2026).
"""
from __future__ import annotations

import pandas as pd

from . import config


def read_updated(base_file, update_files, key_cols: list[str], **read_kw) -> pd.DataFrame:
    """Concatenate base + update vintages; on duplicate `key_cols`, the last vintage wins.

    A caller's `usecols` may omit part of the natural key (e.g. the panel loader selects
    `uf_code`, not `geocode`). Deduplicating on a partial key would collapse whole rows, so the
    missing key columns are read anyway and dropped afterwards.
    """
    present = list(update_files and [f for f in update_files if f.exists()] or [])
    if not present:
        return pd.read_csv(base_file, **read_kw)

    kw = dict(read_kw)
    usecols = kw.pop("usecols", None)
    extra: list[str] = []
    if usecols is not None:
        extra = [c for c in key_cols if c not in usecols]
        kw["usecols"] = list(usecols) + extra

    frames = [pd.read_csv(base_file, **kw)] + [pd.read_csv(f, **kw) for f in present]
    df = pd.concat(frames, ignore_index=True)
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise KeyError(f"vintage key columns {missing} absent from {base_file}")
    df = df.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
    return df.drop(columns=extra) if extra else df


def read_dengue(**read_kw) -> pd.DataFrame:
    return read_updated(config.DENGUE_FILE, config.DENGUE_UPDATE_FILES,
                        ["geocode", "epiweek"], **read_kw)


def read_climate(**read_kw) -> pd.DataFrame:
    return read_updated(config.CLIMATE_FILE, config.CLIMATE_UPDATE_FILES,
                        ["geocode", "epiweek"], **read_kw)


def read_forecast(**read_kw) -> pd.DataFrame:
    """Copernicus, base + update. The update spells humidity `rel_umid_med`; normalize it."""
    frames = [pd.read_csv(config.FORECAST_FILE, **read_kw)]
    for f in config.FORECAST_UPDATE_FILES:
        if not f.exists():
            continue
        kw = dict(read_kw)
        cols = kw.pop("usecols", None)
        u = pd.read_csv(f, **kw)
        u = u.rename(columns={"rel_umid_med": "umid_med"})
        if cols is not None:
            u = u[[c for c in cols]]
        frames.append(u)
    if len(frames) == 1:
        return frames[0]
    df = pd.concat(frames, ignore_index=True)
    key = ["geocode", "reference_month", "forecast_months_ahead"]
    return df.drop_duplicates(subset=key, keep="last").reset_index(drop=True)


def read_ocean(**read_kw) -> pd.DataFrame:
    """ENSO/IOD/PDO. Prefers the refreshed full file over the original snapshot, then updates.

    The refreshed and update vintages carry an `epiweek` column the original snapshot lacks;
    it is dropped, because the consumer derives epiweek by as-of merging onto the dengue date
    grid and that alignment must stay the single source of truth.
    """
    base = (config.OCEAN_REFRESHED_FILE if config.OCEAN_REFRESHED_FILE.exists()
            else config.OCEAN_FILE)
    df = read_updated(base, config.OCEAN_UPDATE_FILES, ["date"], **read_kw)
    drop = [c for c in df.columns if c.startswith("Unnamed")] + ["epiweek"]
    return df.drop(columns=drop, errors="ignore")


def read_population(extend_to: int = config.POP_EXTEND_TO_YEAR, **read_kw) -> pd.DataFrame:
    """DATASUS population, carried forward per geocode to `extend_to` (see module docstring)."""
    pop = pd.read_csv(config.POP_FILE, **read_kw)
    last_year = int(pop["year"].max())
    if extend_to <= last_year:
        return pop
    tail = pop[pop["year"] == last_year]
    extra = pd.concat([tail.assign(year=y) for y in range(last_year + 1, extend_to + 1)],
                      ignore_index=True)
    extra["year"] = extra["year"].astype(pop["year"].dtype)
    return pd.concat([pop, extra], ignore_index=True)
