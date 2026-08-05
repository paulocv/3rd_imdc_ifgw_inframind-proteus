"""Prepare necessary data for projections with the outbreak dynamic model.

This script includes at least two crucial stages:
- Combine calibration distributions and predictions from multiple years.
- Incorporate predictions from the outbreak features model to generate parameter
  samples for the projections.
"""
import sys
import warnings
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Union, Tuple, Any, Literal

import pandas as pd
import matplotlib as mpl
import matplotlib.dates
import matplotlib.ticker
from matplotlib import pyplot as plt
from scipy.stats import (
    gaussian_kde, norm as normal_distribution, lognorm as lognormal_distribution
)

from inframind_proteus.empirical_data import DiseaseTimeSeriesCache
from inframind_proteus.outbreak_dynamics import SimulationConfig
from inframind_proteus.outbreak_dynamics.outbreak_features import OutbreakFeaturePredictionsCache

from inframind_proteus import BaseConfig
from inframind_proteus.outbreak_dynamics.utils import load_yaml_dict, parse_set_arguments_with_yaml, add_set_argument, \
    apply_include_exclude_logic, map_parallel_or_sequential, make_axes_seq, save_yaml_dict, make_yaml_exportable_dict


def main(argv: Union[list[str], None] = None):
    """process_data_for_projections.py"""
    cfg, data = initialize_program(argv)

    load_global_data(cfg, data)

    # --- Run location-specific logic in parallel
    def _task(location_id):
        """
        RAM WARNING: If `data` carries heavy data at this point, it will
        copy to all processes. If necessary, pass only the necessary
        data elements to the function.
        """
        try:
            process_location(location_id, cfg, data)
        except Exception as e:
            if cfg.raise_on_location_error:
                raise e
            else:
                print(f"Error processing location {location_id}: {e}")

    _contents = list(data.location_ids)

    map_parallel_or_sequential(
        _task, _contents,
        ncpus=cfg.ncpus,
    )


class ProgramConfig(BaseConfig):
    """Internal configuration data class for the
    `process_data_for_projections` script.
    """
    config_fpath: Path = Path("configs/process_data_for_projections_default.yaml")
    # base_sim_config_fpath: Path = Path("configs/simulation_config_default.yaml")
    # # NOTE: This loads the defaults, which may have diverged from the actual simulation.
    # # When possible, use a reconstructed config dict from the simulation outputs.

    # # -()- Validation round paths and params
    # calibrations_main_dir: Path = Path("outputs/validation_round_calibration")
    # outbreak_features_predictions_dir: Path = Path("outputs/validation_round_outbreak_features")
    # location_year_subdir_fmt: str = "{location_id}_{year}"
    # # outbreak_features_predictions_dir: Path = Path("predictions")
    # output_dir: Path = Path("outputs/validation_round_projections")
    # use_projection_years: list[int] = [2022, 2023, 2024, 2025]  # Validation round projection years
    # use_calibration_years: list[int] = [
    #     2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021,
    #     2022, 2023, 2024,
    # ]

    # -()- Forecast round paths and params
    calibrations_main_dir: Path = Path("outputs/forecast_round/calibrations")
    outbreak_features_predictions_dir: Path = Path("outputs/forecast_round_outbreak_features") # TODO Get updated
    location_year_subdir_fmt: str = "{location_id}_{year}"
    output_dir: Path = Path("outputs/forecast_round/projections")
    use_projection_years: list[int] = [2026]  # Validation round projection years
    use_calibration_years: list[int] = [
        2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021,
        2022, 2023, 2024,
        2025,  # Additional for forecast round
    ]

    uf_table_fpath = Path("data/demographic/uf_table.csv")

    num_projection_samples = 5000  # Final parameter samples to have in the end
    projection_sampling_seed = 5

    use_location_ids: list = []

    # Years which should not be used for calibration for each location,
    #   e.g. due to missing data or known anomalies.
    exclude_years_by_location: dict[str, list[int]] = defaultdict(
        lambda: list(),
    )

    outbreak_feature_names: list[str] = OutbreakFeaturePredictionsCache.feature_names

    outbreak_features_fit_model: Literal[
        "kde", "normal", "lognormal"
    ]  = "kde"

    ll_temperature: float = 1.0  # Temperature for loglikelihood evaluation (1.0 = no change)
    # This allows us to tune the overall influence of outbreak features.
    # < 1 values will sharpen the likelihood, increasing their influence.
    # > 1 values will flatten the likelihood, decreasing their influence.

    ncpus: int = 1  # Parallelize only over locations.

    raise_on_location_error: bool = True
    # If True, raise an error if a location fails to process.
    # If False, skip that location.


    def preprocess(self, *args, **kwargs):
        super().preprocess(*args, **kwargs)

        # Enforce `exclude_years_by_location` as a defaultdict
        if not isinstance(self.exclude_years_by_location, defaultdict):
            self.exclude_years_by_location = defaultdict(
                lambda: list(),
                **self.exclude_years_by_location,
            )

        self.convert_path_fields()


class ProgramData:
    """Internal data class for the `process_data_for_projections` script.
    """
    uf_table_df: pd.DataFrame
    # base_sim_config_dict: dict[str, Any]
    # base_sim_config: SimulationConfig
    # # Note: The sim config is a payload because it is not fully compatible with
    # #   the BaseConfig scheme, as it runs through a custom validation `from_dict()`.
    # #   If this gets fixed, one could override params through command line.

    location_ids: list
    projection_years: list
    location_year_index: pd.MultiIndex

    # Location-specific data
    # =============
    # Will only be loaded within the location process
    observations_sr: pd.Series
    outb_feats_cache: OutbreakFeaturePredictionsCache
    calibration_config_dicts: dict[int, dict[str, Any]]
    # Indexed by calibration year
    calibration_years: list[int]
    # param_samples_df: pd.DataFrame
    # max_likelihood_df: pd.DataFrame
    case_stats_trajectories_df: pd.DataFrame
    # outbreak_features_df: pd.DataFrame

    multiyear_samples_df: pd.DataFrame
    # This is the most complete data frame, with
    # - parameters (fixed and explored)
    # - outbreak features
    # - weights (posterior and multiyear)
    # ...
    # For all years available from calibration

    posterior_samples_df: pd.DataFrame
    # Final product for each location and projection year


def parse_args_get_dict(argv) -> dict:
    """"""
    parser = ArgumentParser()

    # --- Config file path
    parser.add_argument(
        "--config-fpath", "--cfg", "-c",
        default=ProgramConfig.config_fpath,
        type=Path,
        help="Path to the calibration YAML configuration file.",
    )

    parser.add_argument(
        "--output-dir", "--out", "-o",
        # default=ProgramConfig.output_dir,
        type=Path,
        help="Path to the output directory.",
    )

    parser.add_argument(
        "--use-location-ids",  "-l",
        default=None,
        type=str,
        action="append",
        help=(
            "List of location IDs to process. Can be specified multiple times. "
            "Using this argument resets the default list from the program or "
            "config file. Example: \"-l SP -l MG\" will set use_location_ids to "
            "['SP', 'MG'], regardless of what other locations have been specified"
            "on the defaults."
        ),
    )

    parser.add_argument(
        "--use-projection-years",  "-y",
        default=None,
        type=int,
        action="append",
        help=(
            "List of years to prepare projections for. "
            "Can be specified multiple times. "
            "Using this argument resets the default list from the program or "
            "config file. Example: \"-y 2022 -y 2023\" will set use_projection_years to "
            "[2022, 2023], regardless of what other years have been specified"
            "on the defaults."
        )
    )

    # --- Generic nested `--set` argument.
    add_set_argument(parser)

    # ======

    args = parser.parse_args(argv)
    # Retain only informed arguments to avoid overriding config.
    args_dict = {k: v for k, v in args.__dict__.items() if v is not None}

    # Proceess --set arguments to override config values.
    set_args = args_dict.pop("set")
    overrides = parse_set_arguments_with_yaml(set_args)
    args_dict.update(overrides)

    return args_dict


def initialize_program(argv) -> Tuple[ProgramConfig, ProgramData]:
    # --- Program initialization sequence
    args_dict = parse_args_get_dict(argv)
    cfg = ProgramConfig()
    if "config_fpath" in args_dict:
        config_dict = load_yaml_dict(args_dict["config_fpath"])
        cfg.update_from_dict(config_dict)
    cfg.update_from_dict(args_dict)
    cfg.preprocess()
    data = ProgramData()

    return cfg, data


def load_global_data(cfg: ProgramConfig, data: ProgramData):
    """Load global data that is not location/year specific."""
    # --- Load UF table
    uf_table = pd.read_csv(cfg.uf_table_fpath)
    data.uf_table = uf_table

    # --- Set list of contents to iterate on
    data.location_ids = apply_include_exclude_logic(
        all_series=data.uf_table["uf"],
        include_list=cfg.use_location_ids,
        exclude_list=None,
    )
    data.projection_years = cfg.use_projection_years

    # # --- Load base simulation config to retrieve parameters from
    # data.base_sim_config_dict = load_yaml_dict(cfg.base_sim_config_fpath)
    # data.base_sim_config = SimulationConfig.from_dict(data.base_sim_config_dict)


def load_and_preprocess_location_data(
        location_id, cfg: ProgramConfig, data: ProgramData
):
    """"""
    _ptab = " -- "  # Tabulation of print messages
    print(f"{_ptab}Loading and preprocessing data {location_id=}")

    data.calibration_years = calibration_years = apply_include_exclude_logic(
        all_series=pd.Series(cfg.use_calibration_years),
        exclude_list=cfg.exclude_years_by_location[location_id],
    )

    # === Load observations (disease time series)
    data.observations_sr = observations_sr = (
        DiseaseTimeSeriesCache()  # Use default path and variable names
    ).get_location(location_id)

    # === Load outbreak feature predictions
    cache = OutbreakFeaturePredictionsCache(
        main_dir=cfg.outbreak_features_predictions_dir,
    )
    for feature_name in cfg.outbreak_feature_names:
        # Note - This can be improved and streamlined once final shape is defined
        df = cache.load_file(
            cache.get_file_path(
                location_id=location_id, feature_name=feature_name
            )
        )
        cache.add_df_to_cache(
            df,
            feature_name=feature_name,
        )

    data.outb_feats_cache = cache

    # ==== Load data from all specified calibration years
    keys_list = list()
    data_lists = defaultdict(lambda: list())
    for cal_year in calibration_years:
        _ptab = " -- -- "
        out_dir = (
                cfg.calibrations_main_dir
                / cfg.location_year_subdir_fmt.format(
                    location_id=location_id, year=cal_year
                )
        )

        if not out_dir.exists():
            raise FileNotFoundError(
                f"Output directory for {location_id=} {cal_year=} does not exist: "
                f"{out_dir}"
            )

        # print(f"{_ptab}Loading from {out_dir}...")
        keys_list.append(cal_year)

        # --- Calibration configuration dictionary
        data_lists["config_dict"].append(
            load_yaml_dict(
                out_dir / "calibration_program_config.yaml",
                safe=False
            )
        )

        # --- Posterior parameter samples
        df = pd.read_csv(
            out_dir / "stage3_posterior_samples.csv.gz",
            index_col="i_simulation",
        )
        data_lists["param_samples"].append(df)

        # --- Maximum likelihood parameters from stage 1
        series = (
            pd.read_csv(
                out_dir / "stage1_max_ll_params.csv",
                index_col="parameter_name"
            )["value"]
        )
        data_lists["maximum_likelihood"].append(series)

        # --- Case trajectories (summary stats by day only, not full trajectories)
        df = (
            pd.read_csv(
                out_dir / "stage3_case_stats_trajectories.csv",
                index_col="date",
                parse_dates=["date"],
            )
        )
        data_lists["case_stats_trajectories"].append(df)

        # --- Calculated outbreak features
        df = pd.read_csv(
            out_dir / "outbreak_feature_stats.csv.gz",
            index_col=0, header=[0, 1], skiprows=[2]
        )
        df.index.name = "i_simulation"
        data_lists["outbreak_features"].append(df)

    # Concatenate datasets
    # ----------
    if len(keys_list) == 0:
        raise ValueError("No data loaded. Check the output directories.")

    # --- Configuration dictionaries
    data.calibration_config_dicts = {
        year: config_dict
        for year, config_dict in zip(keys_list, data_lists["config_dict"])
    }

    # --- Posterior samples
    param_samples_df = pd.concat(
        data_lists["param_samples"],  # List of dataframes
        axis=0,
        keys=keys_list,  # Use years as keys for the multi-index
        names=["year"],
    )

    # --- Max likelihood param (list of series into dataframe)
    max_likelihood_df = pd.concat(
        data_lists["maximum_likelihood"],  # List of series
        axis=1,
    ).T
    max_likelihood_df.index = keys_list
    max_likelihood_df.index.name = "year"

    # --- Case trajectories (summary stats by day only, not full trjaectories)
    case_stats_trajectories_df = data.case_stats_trajectories_df = (
        pd.concat(
            data_lists["case_stats_trajectories"],
            keys=keys_list,
            names=["year"]
        )
    )

    # --- Outbreak features
    outbreak_features_df = pd.concat(
        data_lists["outbreak_features"],
        axis=0,
        keys=keys_list,
        names=["year"],
    )

    # ==========================

    # General preprocessing
    # ======================

    # Combine sample-based datasets into a single data frame with relevant variables
    # -------------
    # Prepare a full data frame with all relevant variables
    exclude_vars = [
        "notif_scaling_factor",  # Placeholder, real scaling is `notif_relative_scale`
    ]

    # --- Merge fixed parameters from stage 1
    # (Take only parameters that were not explored)
    _new_params = [
        p for p in max_likelihood_df.columns
        if p not in param_samples_df.columns
           and p not in exclude_vars
    ]
    df = pd.merge(
        left=param_samples_df.reset_index(),
        right=max_likelihood_df[_new_params],
        on="year",
    ).set_index(["year", "i_simulation"])

    # --- Merge with outbreak features
    # ONLY MEAN to simplify column indexing and plotting
    mean_outbreak_features = outbreak_features_df.xs("mean", level="stat", axis=1)
    df = pd.merge(
        left=df,
        right=mean_outbreak_features,
        left_index=True, right_index=True,
    )

    # Rebalance weights for between-years equality
    # ----------
    # Modify param_samples_df in place
    _year_index = df.index.get_level_values("year")
    sum_weights = df.groupby("year")["posterior_weight"].sum().reindex(_year_index)
    sum_weights.index = df.index
    df["multiyear_weight"] = df["posterior_weight"] / sum_weights  # Rebalance by year
    df["multiyear_weight"] /= df["multiyear_weight"].max()  # Normalize to max = 1

    # ---

    data.multiyear_samples_df = df

    # ---- Some validation
    df: pd.DataFrame = data.multiyear_samples_df

    if df.isna().any().any():
        warnings.warn(
            "NaN values found in multiyear_samples_df. "
            "Check the data loading and merging steps."
        )


def process_location(location_id, cfg: ProgramConfig, data: ProgramData):
    """"""
    print(f"Processing {location_id=}")
    load_and_preprocess_location_data(location_id, cfg, data)

    for proj_year in data.projection_years:
        # --- Select calibration samples from available years only
        avail_samples_df = _calculate_projection_priors(
            location_id,
            proj_year,
            cfg, data
        )

        # --- Combine with predicted outbreak features
        likelihood_sr = _calc_likelihood_of_outbreak_features(
            location_id, proj_year,
            avail_samples_df=avail_samples_df,
            cfg=cfg, data=data
        )

        # --- Build posterior and obtain final samples
        posterior_samples_df = _calc_posterior_samples(
            location_id, proj_year,
            avail_samples_df=avail_samples_df,
            likelihood_sr=likelihood_sr,
            cfg=cfg, data=data
        )

        data.posterior_samples_df = posterior_samples_df

        # --- Export outputs of this iteration
        _export_location_outputs(
            location_id, proj_year,
            cfg=cfg, data=data
        )

    # --- Extra plot: Combined calibration time series (all years)
    _plot_all_calibrations(
        cfg=cfg, data=data,
        location_id=location_id,
    )

    return


def _calculate_projection_priors(
        location_id, proj_year, cfg: ProgramConfig, data: ProgramData
) -> pd.DataFrame:
    """ Prepare a prior distribution of model parameters combining
    all available calibration years before the projection year.
    """
    _ptab = " -- "  # Tabulation of print messages

    print(f"{_ptab}Calculating priors for {location_id=} {proj_year=}")
    # ---

    # --- Define and filter available calibration years
    avail_years = data.multiyear_samples_df.index.get_level_values("year").unique()
    calib_years = [y for y in avail_years if y < proj_year]

    avail_samples_df: pd.DataFrame = data.multiyear_samples_df.loc[
        pd.IndexSlice[calib_years, :], :
    ]

    return avail_samples_df


def _get_and_check_zero_date_epiweek(
        calibration_config_dicts
):
    """Fetch the reference date epiweek used for each year, and warn if they
    are not the same among all years.
    """
    zero_epiweeks = [
        cfg_dict["zero_date_epiweek"]
        for cfg_dict in calibration_config_dicts.values()
    ]
    if len(set(zero_epiweeks)) > 1:
        warnings.warn(
            f"Different zero_date_epiweek values found among calibration years: "
            f"{zero_epiweeks}. Using the first one."
        )
    return zero_epiweeks[0]


def _fit_and_eval_outbreak_features(
        outb_feats_predicted_samples: pd.Series,  # Predictions
        out_feats_avail_samples: pd.Series,  # Available samples to evaluate
        model: Literal["kde", "normal", "lognormal"] = "kde",
):
    """Adjust predicted outbreak feature samples to a selected model, then
    evaluate it on the available samples to obtain likelihoods.
    """
    # -()- KDE over the predicted samples
    if model == "kde":
        prediction_kde = (
            gaussian_kde(
                outb_feats_predicted_samples,
            )
        )
        pdf_evals = prediction_kde.evaluate(
            out_feats_avail_samples
        )

    # -()- Simple Gaussian fit over all predicted samples
    elif model == "normal":
        mean = outb_feats_predicted_samples.mean()
        std = outb_feats_predicted_samples.std()
        dist = normal_distribution(loc=mean, scale=std)
        pdf_evals = dist.pdf(out_feats_avail_samples)

    # -()- Lognormal fit - Did better on statistical tests with all features
    elif model == "lognormal":
        fit_params = lognormal_distribution.fit(
            outb_feats_predicted_samples
        )
        dist = lognormal_distribution(*fit_params)
        pdf_evals = dist.pdf(out_feats_avail_samples)

    else:
        raise ValueError(f"Unknown outbreak_features_fit_model: {model}")

    return pdf_evals


def _calc_likelihood_of_outbreak_features(
        location_id,
        proj_year,
        avail_samples_df: pd.DataFrame,
        cfg: ProgramConfig,
        data: ProgramData,
):
    """Update the prior distribution of model parameters with outbreak
    features predictions for the projection year.
    """
    _ptab = " -- "  # Tabulation of print messages

    # Initialize a likelihood series
    likelihood_sr = pd.Series(1.0, index=avail_samples_df.index, name="likelihood")

    # Main loop over outbreak features
    # ============
    for feature_name in cfg.outbreak_feature_names:
        # --- Fetch predicted samples
        outb_feats_predicted_samples = (
            data.outb_feats_cache.get_sample_series(
                feature_name=feature_name,
                location_id=location_id,
                year=proj_year,
            )
        )

        # --- Adjust offset on reference week for peak week predictions
        if feature_name in ["peak_week"]:
            dyn_zero_epiweek = _get_and_check_zero_date_epiweek(
                data.calibration_config_dicts
            )
            feat_zero_epiweek = data.outb_feats_cache.peak_ref_epiweek
            if dyn_zero_epiweek - feat_zero_epiweek:
                # Adjust on predictions
                outb_feats_predicted_samples -= (dyn_zero_epiweek - feat_zero_epiweek)

        # --- Evaluate calibration samples with predicted distributions
        pdf_evals = _fit_and_eval_outbreak_features(
            outb_feats_predicted_samples=outb_feats_predicted_samples,
            out_feats_avail_samples=avail_samples_df[feature_name],
            model=cfg.outbreak_features_fit_model,
        )

        # (Optional) Apply temperature to the likelihood evaluation
        if cfg.ll_temperature != 1.0:
            pdf_evals = pdf_evals ** (1.0 / cfg.ll_temperature)

        likelihood_sr *= pdf_evals

    return likelihood_sr


def _calc_posterior_samples(
        location_id,
        proj_year,
        avail_samples_df: pd.DataFrame,
        likelihood_sr: pd.Series,
        cfg: ProgramConfig,
        data: ProgramData,
) -> pd.DataFrame:
    """Calculate posterior samples for the projection year."""
    _ptab = " -- "  # Tabulation of print messages
    print(f"{_ptab}Calculating posterior samples for {location_id=} {proj_year=}")

    # --- Calculate posterior weights
    posterior_weights = avail_samples_df["multiyear_weight"] * likelihood_sr
    posterior_weights /= posterior_weights.max()  # Normalize to max = 1
#     posterior_weights /= posterior_weights.sum()  # Normalize to sum = 1


    # --- Sample from the prior distribution using the posterior weights
    sampled_indices = avail_samples_df.sample(
        n=cfg.num_projection_samples,
        replace=True,
        weights=posterior_weights,
        random_state=cfg.projection_sampling_seed,
    ).index

    # --- Construct the posterior samples DataFrame, with potentially relevant info
    posterior_samples_df = (
        avail_samples_df
        .reindex(sampled_indices)
        .reset_index()
        .drop(columns=["i_simulation", "multiyear_weight"])
        .rename(columns={"year": "calibration_year"})
    )
    # Optional: Keep weights used to sample (not to be reused!)
    posterior_samples_df["weight"] = posterior_weights.reindex(sampled_indices).values

    posterior_samples_df.index.name = "i_simulation"  # For projection simulations

    # Plots and diagnostics
    # =====
    _plot_bayesian_update(
        cfg=cfg, data=data,
        location_id=location_id,
        proj_year=proj_year,
        avail_samples_df=avail_samples_df,
        likelihood_sr=likelihood_sr,
        posterior_weights=posterior_weights,
        posterior_samples_df=posterior_samples_df,
    )

    return posterior_samples_df


def _get_and_make_plots_dir(
        main_out_dir: Path, subdir_fmt,
        location_id, proj_year
):
    out_dir = main_out_dir / "process_data_diagnostics" / subdir_fmt.format(
        location_id=location_id, year=proj_year
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _plot_all_calibrations(
        cfg: ProgramConfig, data: ProgramData,
        location_id,
):
    """"""
    _ptab = " -- "  # Tabulation of print messages
    plots_out_dir = _get_and_make_plots_dir(
        main_out_dir=cfg.output_dir,
        subdir_fmt=cfg.location_year_subdir_fmt,
        location_id=location_id,
        proj_year="all-years",
    )
    # ==== Plot year-wise. Should define a window.

    with plt.rc_context({
        "patch.linewidth": 0,
    }):
        fig, ax = plt.subplots(figsize=(15, 6))

        # Prediction stats from each year's calibration
        # --------
        for year, df in data.case_stats_trajectories_df.groupby("year"):
            dates = df.index.get_level_values("date")
            color = "palevioletred"

            if year in cfg.exclude_years_by_location[location_id]:
                color = "gray"

            pred_ci = ax.fill_between(
                dates, df["q025"], df["q975"],
                alpha=0.5, color=color, label="95% CI"
            )
            pred_med = ax.plot(
                dates, df["median"], color=color, label="Median prediction"
            )[0]

        # --- Observations
        obs = ax.plot(
            data.observations_sr, "ko",
            ms=2, label="Observations"
        )[0]

        ax.xaxis.set_major_locator(mpl.dates.YearLocator())
        ax.set_ylabel("Weekly dengue cases")

        ax.legend(handles=[obs, pred_med, pred_ci])
        fig.suptitle(f"Calibration time series - {location_id}")
        fig.tight_layout()

        fpath = plots_out_dir / f"all_calibration_trajectories.pdf"
        fig.savefig(fpath)
        print(f"{_ptab}Exported calibration trajectory plots: {fpath}")
        plt.close(fig)

    return


def _plot_bayesian_update(
        cfg: ProgramConfig, data: ProgramData,
        location_id, proj_year,
        avail_samples_df: pd.DataFrame,
        likelihood_sr: pd.Series,
        posterior_weights: pd.Series,
        posterior_samples_df: pd.DataFrame
):
    plots_out_dir = _get_and_make_plots_dir(
        main_out_dir=cfg.output_dir,
        subdir_fmt=cfg.location_year_subdir_fmt,
        location_id=location_id,
        proj_year=proj_year,
    )

    parameter_update_names = [
        "rt_logist_r_high",
        "rt_logist_start",
        "notif_nb_overdispersion",
    ]

    # Update on relevant distributions
    # ================================
    with plt.rc_context({
        "hist.bins": 30,
    }):
        fig, axes = make_axes_seq(
            num_axes=(
                len(cfg.outbreak_feature_names)
                + len(parameter_update_names)
            ),
        )
        ax_count = 0

        # Update on each outbreak feature
        # -------------------------------
        for i, feature_name in enumerate(cfg.outbreak_feature_names):
            ax = axes[i]
            ax_count += 1

            # Prior distribution (calibrated parameters)
            ax.hist(
                avail_samples_df[feature_name],
                weights=avail_samples_df["multiyear_weight"],
                density=True,
                alpha=0.5,
                label='Prior',
            )

            # Distribution of predictions (likelihood)
            _samples = data.outb_feats_cache.get_sample_series(
                feature_name=feature_name,
                location_id=location_id,
                year=proj_year,
            )
            ax.hist(
                _samples,
                density=True,
                alpha=0.5,
                label='Predicted',
            )

            # Posterior distribution (after outbreak feature update)
            ax.hist(
                avail_samples_df[feature_name],
                weights=posterior_weights,
                density=True,
                alpha=0.5,
                label="Posterior"
            )

            ax.set_xlabel(feature_name)
            ax.set_ylabel("Density")

        # Update on selected parameters
        # -------------------------------
        for i, param_name in enumerate(parameter_update_names):
            ax = axes[ax_count]
            ax_count += 1

            # Prior distribution (calibrated parameters)
            ax.hist(
                avail_samples_df[param_name],
                weights=avail_samples_df["multiyear_weight"],
                density=True,
                alpha=0.5,
                label='Prior',
            )

            # "predictions" (just skip to use the color)
            ax.bar([], [], label="Predicted")

            # Posterior distribution (after outbreak feature update)
            ax.hist(
                avail_samples_df[param_name],
                weights=posterior_weights,
                density=True,
                alpha=0.5,
                label="Posterior"
            )

            ax.set_xlabel(param_name)
            ax.set_ylabel("Density")

        axes[0].legend()
        fig.suptitle(f"Bayesian update: {location_id}")
        fig.tight_layout()

        fig.savefig(
            plots_out_dir / "bayesian_update_features.pdf"
        )

    # Parameters sampled from posterior
    # =================================
    # Evaluate quality of the sampled parameters that go to projections
    with plt.rc_context({
        "hist.bins": 30,
    }):
        fig, axes = make_axes_seq(2)

        # Distribution of calibration years
        # ---------------------------------
        ax = axes[0]
        sr = posterior_samples_df["calibration_year"].value_counts(normalize=True).sort_index()
        ax.bar(
            sr.index, sr.values
        )
        ax.set_title("Calibration years")
        ax.set_ylabel("Proportion")
        ax.xaxis.set_major_locator(
            mpl.ticker.MultipleLocator(4)
        )
        ax.xaxis.set_minor_locator(
            mpl.ticker.MultipleLocator(1)
        )

        # General numbers
        # ----------------
        ax = axes[1]
        txt = ""
        _n_samples = posterior_samples_df.shape[0]
        _n_dup = posterior_samples_df.duplicated().sum()
        # _weights = posterior_samples_df["weight"]
        _weights = posterior_weights  # All weights, not just sampled ones
        _dominance = (_weights**2).sum() / (_weights.sum()**2)

        txt += f"Total samples: {_n_samples}\n"
        txt += f"Duplicate samples: {_n_dup} ({_n_dup / _n_samples * 100:0.2f}%)\n"
        txt += f"Weight dominance: {_dominance:0.3e}\n"

        ax.text(0.05, 0.9, txt, va="top")

        fig.tight_layout()


        fig.savefig(
            plots_out_dir / "posterior_samples_summary.pdf"
        )

    return


def _export_location_outputs(
        location_id, proj_year,
        cfg: ProgramConfig, data: ProgramData,
):
    """"""
    _ptab = " -- -- "  # Tabulation of print messages
    # --- Obtain and create the output directory
    out_dir = (
        cfg.output_dir
        / "projections"
        / cfg.location_year_subdir_fmt.format(
            location_id=location_id, year=proj_year
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Configuration
    d = make_yaml_exportable_dict(
        cfg.to_dict(), copy=True
    )
    save_yaml_dict(
        d, out_dir / "process_data_for_projections_config.yaml",
        # safe=False
    )

    # --- Posterior parameter samples
    posterior_df = data.posterior_samples_df
    fpath = out_dir / "projection_parameter_samples.csv"
    posterior_df.to_csv(fpath, index=True)
    print(f"{_ptab}Exported posterior parameter samples to {fpath}")



if __name__ == "__main__":
    main(sys.argv[1:])
