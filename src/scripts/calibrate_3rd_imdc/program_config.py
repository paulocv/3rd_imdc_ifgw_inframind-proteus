from __future__ import annotations

from dataclasses import dataclass, Field, field
from email.policy import default
from pathlib import Path

from inframind_proteus import BaseConfig


# @dataclass
class Stage1Config(BaseConfig):
    """Stage 1 configuration: Broad exploration most free
    parameters."""

    num_simulations: int = 2 ** 20

    # Number of data points in the pre-simulation period
    # This is used to determine the prior for the notif_relative_scale parameter
    presim_period_num_points: int = 6


# @dataclass
class Stage2Config(BaseConfig):
    """Stage 2 configuration: Focused exploration of a subset of
    free parameters.

    Nuisance parameters are fixed to optimal values
    provided by stage 1.
    """
    num_simulations: int = 2 ** 20

    # free_params: Field[list[str]] = field(default_factory=lambda: [
    free_params: Field[list[str]] = [
        "rt_logist_r_high",
        "rt_logist_start",
        # "notif_nb_overdispersion",
        # "notif_relative_scale",
    ]#)

    # Posterior building
    posterior_seed: int = 42  # Seed for any posterior-sampling procedure
    ll_temperature: float = 1.  # Higher values flatten the likelihood distribution
    rel_weight_cutoff: float = 1e-3  # Cutoff relative to maximum weight
    min_samples_to_kde: int = 1000  # Minimum number of samples to keep after cutoff (overrides cutoff if not met)
    max_samples_to_kde: int = 5000  # Maximum number of samples to keep, avoids heavy KDE calculation


# @dataclass
class Stage3Config(BaseConfig):
    """Stage 3 configuration: Adjustment of confidence
    intervals to match coverages.
    """
    num_simulations: int = 2 ** 18

    sampling_seed: int = 321
    posterior_seed: int = 45  # Seed for any sampling procedure in stage 3 (e.g. KDE sampling)
    ll_temperature: float = 1.  # Higher values flatten the likelihood distribution
    rel_weight_cutoff: float = 1e-3  # Cutoff relative to maximum weight
    min_samples_to_kde: int = 1000  # Minimum number of samples to keep after cutoff (overrides cutoff if not met)
    max_samples_to_kde: int = 5000  # Maximum number of samples to keep, avoids heavy KDE calculation


# @dataclass
class ProgramConfig(BaseConfig):
    """Internal configuration data class for the calibration procedure script."""
    config_fpath: Path = Path("configs/calibrate_3rd_imdc_default.yaml")
    base_sim_config_fpath: Path = Path("configs/simulation_config_default.yaml")  # Optional path to a separate simulation config file (overrides config.sim_cfg)

    # output_dir: Path = Path("outputs/validation_round_calibration")  # Validation round
    output_dir: Path = Path("outputs/forecast_round/calibrations")  # Forecast round
    location_year_subdir_fmt: str = "{location_id}_{year}"

    uf_table_fpath = Path("data/demographic/uf_table.csv")

    # --- Temporal config, epiweek-based (year-agnostic)
    zero_date_epiweek: int = 41  # Reference date for t = 0 (not simulation start)
    sim_start_epiweek: int = 26  # Note: Train data for each season ends at epiweek 25.
    calibration_start_epiweek: int = 41
    calibration_end_epiweek: int = 25  # Of the next year

    # --- Locations and years to run
    use_location_ids = []  # Runs all!
    exclude_location_ids: list[str] = list()
    use_years = list(range(2022, 2023))
    ncpus = 1

    # Split simulations in sequential chunks.
    #   Use lower values to reduce RAM usage.
    #   Use None to keep default
    simulator_max_chunk_size: int | None = None

    stage1: Stage1Config = Stage1Config()
    stage2: Stage2Config = Stage2Config()
    stage3: Stage3Config = Stage3Config()

    def preprocess(self, *args, **kwargs):
        super().preprocess(*args, **kwargs)
        self.convert_path_fields()
