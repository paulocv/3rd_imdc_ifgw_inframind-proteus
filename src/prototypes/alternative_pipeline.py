"""Convenience for running the entire pipeline on alternative folders,
with alternative parameters - e.g. for quick sensitivity analyses.
"""
from pathlib import Path

from inframind_proteus.outbreak_dynamics.utils import map_parallel_or_sequential


def main():

    # Define stuff
    # ================
    calibration_config_fpath = Path(".tmp/calibrate_3rd_imdc_higher-overdisp-bound.yaml")
    process_config_fpath = Path(".tmp/process_data_for_projections_higher-overdisp-bound.yaml")
    project_config_fpath = Path(".tmp/project_3rd_imdc_higher-overdisp-bound.yaml")

    master_output_dir = Path("outputs/sensitivity/higher_overdisp_bound")

    run_calibration     = True
    run_outb_features   = True
    run_data_processing = True
    run_projections     = True

    # # Define these on the config file instead
    # location = "AC"
    # calibration_years = [2020, 2021, 2022, 2023]


    calibrations_dir = master_output_dir / "calibrations"
    projections_dir = master_output_dir / "projections"

    # Calibration
    # ===============
    if run_calibration:
        from scripts.calibrate_3rd_imdc import main as calibrate_3rd_imdc_main
        arg_str = (
            f"-c {calibration_config_fpath} "
            f"-o {str(calibrations_dir)} "
        )
        calibrate_3rd_imdc_main(arg_str.split())

    # Outbreak features
    # =================
    if run_outb_features:
        from scripts.calc_outbreak_features_from_posteriors import main as outb_feat_main
        # Regex for finding folders
        pattern = "??_[0-9][0-9][0-9][0-9]"
        subdirs = list(calibrations_dir.glob(pattern))
        # subdirs = list(glob.glob(str(main_out_dir / pattern)))
        subdirs.sort()

        def _task(subdir):
            _arg_str = f"-o {subdir} "
            outb_feat_main(_arg_str.split())
        _contents  = list(subdirs)[:]

        map_parallel_or_sequential(
            _task, _contents, ncpus=6
        )


    # Processing
    # =================
    if run_data_processing:
        from scripts.process_data_for_projections import main as process_data_main

        arg_str = (
            f"-c {str(process_config_fpath)} "
            f"-o {str(projections_dir)} "
            f"--set calibrations_main_dir \"{str(calibrations_dir)}\" "
        )

        process_data_main(arg_str.split())

    # Projections
    # =================
    if run_projections:
        from scripts.project_3rd_imdc import main as projection_main

        arg_str = (
            f"-c {str(project_config_fpath)} "
            f"-o {str(projections_dir)} "
        )
        projection_main(arg_str.split())




if __name__ == "__main__":
    main()
