import gc
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

from inframind_proteus.outbreak_dynamics import RenewalSimulator, SimulationConfig, SimulationOutput
from inframind_proteus.outbreak_dynamics.sampling import GammaPrior
from inframind_proteus.outbreak_dynamics.simulator import sample_negative_binomial_trajectories
from inframind_proteus.outbreak_dynamics.utils import make_yaml_exportable_dict, save_yaml_dict
from .calibration_stage_1 import Stage1Outputs
from .calibration_stage_2 import Stage2Outputs
from .helpers import _set_config_dict_common, prepare_output_subdirs
from .program_config import ProgramConfig


@dataclass
class Stage3Outputs:
    param_samples: pd.DataFrame
    posterior_kde: gaussian_kde


def run_calibration_stage_3(
        location_id, year,
        cfg: ProgramConfig,
        base_sim_config_dict: dict,
        observations_sr: pd.Series,
        uf_table_df: pd.DataFrame,
        stage1_outputs: Stage1Outputs,
        stage2_outputs: Stage2Outputs,
):
    """"""
    print(f"\trun_calibration_stage_3({location_id}, {year})")

    # Preamble
    # =====================

    # --- Instantiate simulation dictionary for this round
    sim_config_dict = deepcopy(base_sim_config_dict)
    _set_config_dict_common(
        cfg, sim_config_dict,
        location_id,
        year,
        uf_table_df,
        num_simulations=cfg.stage3.num_simulations,
        scoring_metrics=[
            "coverages",
        ]
    )

    # --- Prepare parameter exploration
    # - Set nuisance parameters to their stage 1 values
    # - Remove all parameters from exploration
    # - Retain original range of the overdispersion for later use
    param_ranges = sim_config_dict["sampling"]["param_ranges"]
    _d = sim_config_dict
    _sampling = _d["sampling"]
    nuisance_param_names = [
        p for p in param_ranges.keys()
        if p not in stage2_outputs.kde_param_names
    ]
    for param_name in nuisance_param_names:
        # Set max likelihood value from stage 1 as fixed value for this stage
        # (Try to guess parameter location from its name)
        if param_name.startswith("rt_"):
            sub_d = _d["reproduction_number"]["params"]
        elif param_name.startswith("notif_"):
            sub_d = _d["observation_model"]["params"]
        else:
            raise ValueError(f"Cannot guess parameter location for {param_name}")
        sub_d[param_name] = stage1_outputs.max_ll_params[param_name]

    # Remove all parameters from exploration - We'll sample them separately
    # Keep overdispersion's exploration range for later use.
    overdisp_range = param_ranges.get("notif_nb_overdispersion", [0.1, 100.0])
    param_ranges.clear()

    # --- Sample parameters with custom procedure
    sampling_rng = np.random.default_rng(
        cfg.stage3.sampling_seed
    )
    sampled_params_df = _sample_parameters_for_stage_3(
        n_samples=cfg.stage3.num_simulations,
        kde=stage2_outputs.posterior_kde,
        kde_param_names=stage2_outputs.kde_param_names,
        overdisp_range=overdisp_range,
        rng=sampling_rng,
    )

    # --- Adjust outputs and retained data
    # True so we can plot Rt trajectories
    sim_config_dict["output"]["keep_rt_trajectories"] = True

    # Simulations
    # ==================
    # --- Create simulator object with modified configuration dictionary
    simulator = RenewalSimulator.from_config_dict(sim_config_dict)
    sim_cfg = simulator.config

    # --- Run the simulation and scoring
    params_df, initial_infec_df = (
        simulator.build_simulation_data()
    )
    # Override parameters with the custom samples
    # params_df[sampled_params_df.columns] = sampled_params_df
    params_df.update(sampled_params_df)

    _kwargs = dict()
    if cfg.simulator_max_chunk_size is not None:
        _kwargs["max_chunk_size"] = cfg.simulator_max_chunk_size
    sim_results = simulator.run_sequential_chunks(
        params_df=params_df,
        initial_infec_df=initial_infec_df,
        observations_sr=observations_sr,
        **_kwargs
    )
    gc.collect()


    # Simulation postprocessing
    # =========================
    post_samples_df, post_kde, cases_df = (
        _postprocess_stage_3_simulations(
            cfg, params_df, sim_results, simulator
        )
    )

    # Directory for exporting selected outputs
    out_dir = cfg.output_dir / cfg.location_year_subdir_fmt.format(
        location_id=location_id, year=year
    )
    out_dir.mkdir(exist_ok=True, parents=True)

    # Plots and diagnostics
    # ---------------------
    _stage_3_plots_and_diagnostics(
        location_id, year,
        # out_dir=out_dir,
        cfg=cfg,
        observations_sr=observations_sr,
        params_df=params_df,
        sampled_param_names=sampled_params_df.columns.tolist(),
        post_kde=post_kde,
        post_samples_df=post_samples_df,
        cases_df=cases_df,
        sim_cfg=sim_cfg,
        sim_results=sim_results,
        simulator=simulator,
    )

    _export_stage3_data(
        location_id, year,
        # out_dir=out_dir,
        cfg=cfg,
        post_samples_df=post_samples_df,
        sampled_param_names=sampled_params_df.columns.tolist(),
        sim_results=sim_results,
        simulator=simulator,
        cases_df=cases_df,
    )

    return Stage3Outputs(
        param_samples=post_samples_df,
        posterior_kde=post_kde
    )


# Internal helper functions
# =========================

def _sample_parameters_for_stage_3(
        n_samples: int,
        kde: gaussian_kde,
        kde_param_names: list[str],
        overdisp_range: list[float],
        rng: np.random.Generator
):
    """
      Performs two independent sampling series:
        - Stage 2 free parameters (from the KDE)
        - Overdispersion (random inverse sampling).
    """
    # --- Override params_df with the manually sampled stuff
    # Sample from the free parameters' KDE
    kde_param_samples = kde.resample(n_samples, seed=rng).T

    # Sample overdispersion inversely
    overdisp_param_samples = 1. / (
        rng.uniform(
            low=1. / overdisp_range[1],
            high=1. / overdisp_range[0],
            size=n_samples,
        )
    )

    # Store
    df = pd.DataFrame(
        kde_param_samples,
        columns=kde_param_names
    )
    df["notif_nb_overdispersion"] = overdisp_param_samples

    return df


def _postprocess_stage_3_simulations(
        cfg: ProgramConfig,
        params_df: pd.DataFrame,
        sim_results: SimulationOutput,
        simulator: RenewalSimulator
) -> tuple[pd.DataFrame, gaussian_kde, pd.DataFrame]:
    """"""

    # Build posterior distributions
    # =============================
    rng = np.random.default_rng(cfg.stage3.posterior_seed)

    # Treat the likelihood function
    df = pd.concat([params_df, sim_results.scoring.summary], axis=1)
    # _temperature = 1.
    tempered_ll = df["coverage_loglikelihood"] / cfg.stage3.ll_temperature
    tempered_ll -= tempered_ll.max()  # Displace by max
    df["tempered_likelihood"] = np.exp(tempered_ll)

    # --- Apply prior and combine
    # df["prior_weight"] = cfg.stage3.prior(df)  # Disabled for now
    df["prior_weight"] = 1.
    df["posterior_weight"] = df["tempered_likelihood"] * df["prior_weight"]

    # --- Regularize small weights
    max_weight = df["posterior_weight"].max()
    w_cutoff = max_weight * cfg.stage3.rel_weight_cutoff
    num_relevant_weights = (df["posterior_weight"] >= w_cutoff).sum()

    # --- Decision tree: Number of samples to keep based on surviving weights
    min_samples = cfg.stage3.min_samples_to_kde
    max_samples = cfg.stage3.max_samples_to_kde
    if num_relevant_weights < min_samples:
        # Keep min number or as many as available on the original sample size
        df = (
            df
            .sort_values("posterior_weight")
            .iloc[-min_samples:]
        )
    elif num_relevant_weights < max_samples:
        # Keep all above cutoff
        df = (
            df[df["posterior_weight"] >= w_cutoff]
            .sort_values("posterior_weight")
        )
    else:  # num_relevant_weights >= max_samples
        # Re-sample from weights above threshold to keep max number
        df = (
            df[df["posterior_weight"] >= w_cutoff]
            .sample(
                n=max_samples,
                replace=False,
                weights=None,  # No weights when just reducing nr. of samples
                random_state=rng,
            )
            .sort_values("posterior_weight")
        )

    # At this point, df contains selected samples and their weights, with a
    #    variable size between specified min and max.
    post_samples_df = df

    post_kde = gaussian_kde(
        post_samples_df[cfg.stage2.free_params].T.values,
        weights=post_samples_df["posterior_weight"]
    )

    # Predictiion intervals from re-sampled trajectories
    # ===================
    # OBS: This code below is directly adapted from its prototype.
    # The entire re-sampling will be done later by dedicated methods.
    # The code can be shortened and improved!
    rng = np.random.default_rng(seed=42)

    # --- Re-sample from the posterior to choose trajectories to sample from
    num_trajectories = 1000
    re_sample_params: pd.DataFrame = post_samples_df.sample(
        n=num_trajectories,
        replace=True,
        weights=post_samples_df["posterior_weight"],
        random_state=rng,
    )

    # FUTURE IMPROVE: Use `sim_results.infec_df` instead and call a simulator internal method
    # to apply the observation model agnostically instead of a standalone negative binomial
    re_sample_mean_cases = sim_results.mean_cases_df.reindex(re_sample_params.index)

    # Before this is improved, we can just double-check that the observation
    # model is negative binomial
    if simulator.config.observation_model.model != "negative_binomial":
        raise ValueError(
            "The observation model is not negative binomial. "
            "This procedure is currently hardcoded for that model only."
        )

    # After identifying original simulations, reset the indexes to sequential
    for obj in [re_sample_params, re_sample_mean_cases]:
        obj.reset_index(drop=True, inplace=True)
        obj.index.name = "i_sample"

    cases_vec = (
        sample_negative_binomial_trajectories(
            expectancy=re_sample_mean_cases.to_numpy(),
            overdisp=re_sample_params["notif_nb_overdispersion"].to_numpy(),
            rng=rng
        )
    )
    cases_df = pd.DataFrame(
        cases_vec,
        index=re_sample_mean_cases.index,
        columns=re_sample_mean_cases.columns,
    )

    return post_samples_df, post_kde, cases_df


def _stage_3_plots_and_diagnostics(
        location_id, year,
        # out_dir: Path,
        cfg: ProgramConfig,
        observations_sr: pd.Series,
        params_df: pd.DataFrame,
        sampled_param_names: list[str],
        post_kde: gaussian_kde,
        post_samples_df: pd.DataFrame,
        cases_df: pd.DataFrame,
        sim_cfg: SimulationConfig,
        sim_results: SimulationOutput,
        simulator: RenewalSimulator
):
    """"""
    _, out_dir = prepare_output_subdirs(
        location_id, year,
        output_dir=cfg.output_dir,
        location_year_subdir_fmt=cfg.location_year_subdir_fmt,
        mkdirs=True
    )

    # Prior and posterior distributions as histograms
    # =====================
    param_names = sampled_param_names

    fig, axes = plt.subplots(
        nrows=len(param_names),
        ncols=1,
        figsize=(8, 3 * len(param_names))
    )
    df = post_samples_df

    for ax, param_name in zip(axes, param_names):
        weighted = ax.hist(
            x=df[param_name],
            weights=df["posterior_weight"],
            alpha=0.5, label="weighted",
            density=True,
        )

        unweighted = ax.hist(
            x=df[param_name],
            alpha=0.5, label="unweighted",
            density=True,
        )

        ax.set_xlabel(param_name)
        ax.set_ylabel("Density")

    fig.suptitle(f"Stage 3 - Prior vs Posterior for {location_id} {year}")
    fig.tight_layout()
    axes[0].legend()
    # fig.show()

    out_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(out_dir / "stage3_prior-posterior_histograms.pdf")
    plt.close(fig)

    # = = = = EXTRA DIAGNOSTIC PLOTs
    # --- Median trajectories from the prior ensemble (with info from Stage 2)
    # (Unweighted)
    medians_df: pd.DataFrame = sim_results.case_beam_df.xs(0.5)
    sampled_medians_df: pd.DataFrame = medians_df.sample(
        n=500, replace=True, random_state=333,
    )
    with plt.rc_context(
            {
                "patch.linewidth": 0,
                "lines.markeredgewidth": 0,
            }
    ):
        fig, ax = plt.subplots()

        ax.plot(sampled_medians_df.T, color="palevioletred", alpha=0.1)

        sr = observations_sr[sampled_medians_df.columns]
        ax.plot(
            sr, label="Observations",
            color="black", marker="o", linestyle="", markersize=3, alpha=0.75
        )

        # fig.show()

    # --- Abstract infection trajectories
    sampled_infec_df = sim_results.infec_df.sample(
        n=500, replace=True, random_state=337
    )
    with plt.rc_context(
            {
                "patch.linewidth": 0,
                "lines.markeredgewidth": 0,
            }
    ):
        fig, ax = plt.subplots()

        ax.plot(sampled_infec_df.T, color="palevioletred", alpha=0.1)

        # fig.show()



    # --- Visualize sampled trajectories
    with plt.rc_context(
            {
                "patch.linewidth": 0,
                "lines.markeredgewidth": 0,
            }
    ):
        fig, ax = plt.subplots()
        # === Model trajectories as quantiles
        ax.fill_between(
            cases_df.columns,
            cases_df.quantile(0.025), cases_df.quantile(0.975),
            color="palevioletred", alpha=0.5, label="95% CI"
        )
        ax.fill_between(
            cases_df.columns,
            cases_df.quantile(0.25), cases_df.quantile(0.75),
            color="palevioletred", alpha=0.5, label="50% CI"
        )

        ax.plot(cases_df.median(), label="Median", color="palevioletred")

        # === Observations
        sr = observations_sr[cases_df.columns]
        ax.plot(
            sr, label="Observations",
            color="black", marker="o", linestyle="", markersize=3, alpha=0.75
        )

        ax.set_ylabel("Weekly cases")
        ax.set_title(f"Stage 3 - Sampled trajectories for {location_id} {year}")
        ax.legend()

        fig.savefig(out_dir / "stage3_pred-interval_from-resample.pdf")
        plt.close(fig)


def _export_stage3_data(
        location_id, year,
        # out_dir: Path,
        cfg: ProgramConfig,
        post_samples_df: pd.DataFrame,
        sampled_param_names: list[str],
        # post_kde: gaussian_kde,
        sim_results: SimulationOutput,
        simulator: RenewalSimulator,
        cases_df: pd.DataFrame,
):
    """"""
    data_out_dir, plots_out_dir = prepare_output_subdirs(
        location_id, year,
        output_dir=cfg.output_dir,
        location_year_subdir_fmt=cfg.location_year_subdir_fmt,
        mkdirs=True
    )

    # --- Export simulation config (serialized from config object, not informed)
    d = make_yaml_exportable_dict(
        sim_results.config.to_dict()
    )
    save_yaml_dict(d, data_out_dir / "stage3_sim_config.yaml", safe=False)

    # # --- Export fixed-value parameters
    # _cfg = sim_results.config
    # d = _cfg.sampling.rt_params.copy()
    # d.update(_cfg.sampling.observation_params)
    # sr = pd.Series(d, name="value")
    # sr.index.name = "parameter_name"
    # sr.to_csv(out_dir / "stage3_fixed_parameters.csv")

    # --- Export posterior samples - Free parameters only
    cols = [
        *sampled_param_names,
        "posterior_weight",
    ]
    df = post_samples_df[cols]
    df.to_csv(
        data_out_dir / "stage3_posterior_samples.csv.gz",
        compression={
            "method": "gzip",
            "compresslevel": 9,
        },
    )

    # --- Export mean trajectories of remaining samples
    df = sim_results.mean_cases_df
    df = df.reindex(post_samples_df.index)

    df.to_csv(
        data_out_dir / "stage3_mean_cases.csv.gz",
        compression={
            "method": "gzip",
            "compresslevel": 9,
        },
    )

    # --- Export summary of sampled cases trajectories
    df = pd.DataFrame({
        "mean": cases_df.mean(),
        "median": cases_df.median(),
        "std": cases_df.std(),
        "q025": cases_df.quantile(0.025),
        "q250": cases_df.quantile(0.25),
        "q750": cases_df.quantile(0.75),
        "q975": cases_df.quantile(0.975),
        "min": cases_df.min(),
        "max": cases_df.max(),
    })
    df.index.name = "date"

    df.to_csv(
        data_out_dir / "stage3_case_stats_trajectories.csv",
        index=True
    )
