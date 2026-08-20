"""
Calibration procedure of the outbreak dynamics model (renewal equation) of
Inframind Proteus.

This script is originally developed for the
3rd Infodengue-Mosqlimate Dengue Challenge (3rd IMDC).

Usage
-----

"""

from __future__ import annotations

import io
import sys
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from inframind_proteus import BaseConfig
from inframind_proteus.empirical_data import DiseaseTimeSeriesCache
from inframind_proteus.outbreak_dynamics import SimulationConfig
from inframind_proteus.outbreak_dynamics.utils import (
    parse_set_arguments_with_yaml, load_yaml_dict, year_week_to_date,
    apply_include_exclude_logic, map_parallel_or_sequential, make_yaml_exportable_dict, save_yaml_dict, add_set_argument
)
from scripts.calibrate_3rd_imdc.program_config import ProgramConfig
from scripts.calibrate_3rd_imdc.calibration_stage_1 import (
    run_calibration_stage_1, Stage1Outputs
)
from scripts.calibrate_3rd_imdc.calibration_stage_2 import (
    run_calibration_stage_2, Stage2Outputs
)
from scripts.calibrate_3rd_imdc.calibration_stage_3 import (
    run_calibration_stage_3, Stage3Outputs
)


def parse_args_get_dict(argv: list[str] | None = None) -> dict[str, Any]:
    """Parse command-line arguments and return them as a dictionary."""
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
        default=None,
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


class ProgramData:
    """Internal payload data class for the calibration procedure script."""
    uf_table_df: pd.DataFrame
    base_sim_config_dict: dict[str, Any]
    base_sim_config: SimulationConfig
    # Note: The sim config is a payload because it is not fully compatible with
    #   the BaseConfig scheme, as it runs through a custom validation `from_dict()`.
    #   If this gets fixed, one could override params through command line.

    location_ids: list
    years: list
    location_year_index: pd.MultiIndex


def main(argv: list[str] | None = None) -> None:

    # --- Program initialization sequence
    args_dict = parse_args_get_dict(argv)
    config_dict = load_yaml_dict(args_dict["config_fpath"])
    cfg = ProgramConfig()
    cfg.update_from_dict(config_dict)
    cfg.update_from_dict(args_dict)
    cfg.preprocess()

    # --- Load basic payload
    data = ProgramData()
    data.uf_table_df = pd.read_csv(cfg.uf_table_fpath)
    data.base_sim_config_dict = load_yaml_dict(cfg.base_sim_config_fpath)

    # --- Combine all years and locations to be run
    data.location_ids = apply_include_exclude_logic(
        data.uf_table_df["uf"],
        include_list=cfg.use_location_ids,
        exclude_list=cfg.exclude_location_ids,
    )
    data.years = list(cfg.use_years)
    data.location_year_index = pd.MultiIndex.from_product(
        [data.location_ids, data.years],
        names=["uf", "season"]
    )

    # --- Run calibration algorithm for all location-year combinations
    def _task(location_year_tuple):
        location_id, year = location_year_tuple
        run_calibration_stages_location_year(
            location_id, year,
            cfg=cfg,
            base_sim_config_dict=data.base_sim_config_dict,
            uf_table_df=data.uf_table_df
        )

    _contents = data.location_year_index
    map_parallel_or_sequential(
        _task, _contents,
        ncpus=cfg.ncpus,
        chunksize=1
    )



# Main routines
# =========================

def run_calibration_stages_location_year(
        location_id, year,
        cfg: ProgramConfig,
        base_sim_config_dict: dict,
        uf_table_df: pd.DataFrame,
):
    """"""
    print(f"run_calibration_stages_location_year(({location_id}, {year}))")

    # --- Load location/year specific data
    observations_sr = (
        DiseaseTimeSeriesCache()
        .get_location(location_id)
    )

    # --- Export program configuration
    d = make_yaml_exportable_dict(cfg.to_dict())
    out_dir = cfg.output_dir / cfg.location_year_subdir_fmt.format(
        location_id=location_id,
        year=year
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    save_yaml_dict(d, out_dir / "calibration_program_config.yaml")

    # --- Stage 1
    s1_xt0 = time.time()
    stage1_outputs = run_calibration_stage_1(
        location_id, year,
        cfg=cfg,
        base_sim_config_dict=base_sim_config_dict,
        observations_sr=observations_sr,
        uf_table_df=uf_table_df
    )
    s1_xt1 = time.time()

    # --- Stage 2
    s2_xt0 = time.time()
    stage2_outputs = run_calibration_stage_2(
        location_id, year,
        cfg=cfg,
        base_sim_config_dict=base_sim_config_dict,
        observations_sr=observations_sr,
        uf_table_df=uf_table_df,
        stage1_outputs=stage1_outputs
    )
    s2_xt1 = time.time()

    # --- Stage 3
    s3_xt0 = time.time()
    stage3_outputs = run_calibration_stage_3(
        location_id, year,
        cfg=cfg,
        base_sim_config_dict=base_sim_config_dict,
        observations_sr=observations_sr,
        uf_table_df=uf_table_df,
        stage1_outputs=stage1_outputs,
        stage2_outputs=stage2_outputs
    )
    s3_xt1 = time.time()


    print(
        f"Completed ({location_id}, {year}): "
        f"stage1 {s1_xt1-s1_xt0:0.1f}s  "
        f"stage2 {s2_xt1-s2_xt0:0.1f}s  "
        f"stage3 {s3_xt1-s3_xt0:0.1f}s  "
        f""
    )

    pass

if __name__ == "__main__":
    main(sys.argv[1:])
