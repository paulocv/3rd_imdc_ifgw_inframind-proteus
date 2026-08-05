"""Preprocess the dengue time series data into InfraMIND-internal format.

Starts from dengue time series as provided by the IMDC organization.
"""
import os
import warnings
from pathlib import Path

import pandas as pd

from inframind_proteus.empirical_data import DiseaseTimeSeriesVariables

class Config:
    def __init__(self):
        # Locate project root directory
        self.root_dir = Path(__file__).resolve().parent.parent.parent

        self.imdc_population_fpath = Path("data/data_imdc_2026/datasus_population_2001_2025.csv.gz")

        self.imdc_dengue_fpaths = [
            Path("data/data_imdc_2026/dengue.csv.gz"),
            Path("data/data_imdc_2026/dengue_update_2026.csv.gz"),
        ]

        self.uf_dengue_dirpath = Path("data/disease/dengue_cases_uf_weekly")
        self.uf_dengue_fname_fmt = "dengue_{uf}.csv"

        self.copy_population_years: dict = {
            # tgt: src,
            2026: 2025,  # Repeat 2025 population data for 2026
            2027: 2025,  # Repeat 2025 population data for 2027
        }

        self.do_export = True

def main():
    config = Config()
    os.chdir(config.root_dir)
    print("Working directory: ", os.getcwd())

    dvars = DiseaseTimeSeriesVariables()
    imdc_population_df = pd.read_csv(config.imdc_population_fpath)
    imdc_dengue_df = pd.concat(
        [
            pd.read_csv(fpath, parse_dates=["date"])
            for fpath in config.imdc_dengue_fpaths
        ],
        ignore_index=True
    )

    population_df = copy_population_years(
        imdc_population_df, config.copy_population_years
    )

    uf_dengue_df = aggregate_dengue_to_uf(
        imdc_dengue_df, population_df, dvars
    )

    export_dengue_uf_weekly(
        uf_dengue_df, config.uf_dengue_dirpath, config.uf_dengue_fname_fmt, dvars,
        config.do_export
    )


def copy_population_years(
        imdc_population_df: pd.DataFrame,
        copy_years: dict[int, int],
):
    """Expand the IMDC population data by repeating years as specified,
    to augment available years.

    Obs: Assumes `imdc_population_df` has a simple contiguous index,
    which is reset during the process.
    """
    df = imdc_population_df.copy()
    for new_year, old_year in copy_years.items():
        # Check if actually missing the year
        if new_year in df["year"].values:
            warnings.warn(
                f"Year {new_year} already exists in population data. "
                f"Skipping copy from {old_year}.",
            )
            continue

        # Augment the dataset
        year_df = df[df["year"] == old_year].copy()
        year_df["year"] = new_year
        df = pd.concat([df, year_df], ignore_index=True)
    return df


def aggregate_dengue_to_uf(
        imdc_dengue_df: pd.DataFrame,
        imdc_population_df: pd.DataFrame,
        dvars: DiseaseTimeSeriesVariables,
):
    """"""
    df: pd.DataFrame = imdc_dengue_df.copy()

    # --- Add year and population data
    df["year"] = df["date"].dt.year
    df = df.merge(
        imdc_population_df,
        on=["geocode", "year"],
        how="left",
    )

    if df["population"].hasnans:
        warnings.warn(
            "Population data is missing for some rows. "
            "Possibly because there are dengue cases reported outside the available"
            "population data range."
        )

    # --- Aggregate and calculate per-populaiton incidence
    df = (
        df.groupby(["uf", "date"], sort=True)
        .agg({"casos": "sum", "population": "sum"})
        .reset_index()
    )
    # state_sr = df.groupby(["uf", "date"])["casos"].sum()

    df["case_inc_100k"] = df["casos"] / df["population"] * 1E5

    # --- Standardize variable names
    df = df.rename(columns={
        "casos": dvars.value_variable,
        # "date": dvars.time_variable,
    })

    return df


def export_dengue_uf_weekly(
        uf_dengue_df: pd.DataFrame,
        uf_dengue_dirpath: Path,
        uf_dengue_fname_fmt: str,
        dvars: DiseaseTimeSeriesVariables,
        do_export
):
    """"""
    if not do_export:
        print("Export disabled. Skipping export of dengue time series.")
        return

    for uf, uf_df in uf_dengue_df.groupby("uf"):
        uf_fpath = uf_dengue_dirpath / uf_dengue_fname_fmt.format(uf=uf)
        uf_fpath.parent.mkdir(parents=True, exist_ok=True)
        uf_df[[dvars.time_variable, dvars.value_variable]].to_csv(uf_fpath, index=False)
        print(f"Exported: {uf_fpath}")


if __name__ == "__main__":
    main()
