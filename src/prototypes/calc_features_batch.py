import glob
import os
import time
from pathlib import Path

from inframind_proteus.outbreak_dynamics.utils import map_parallel_or_sequential


def main():
    xt0 = time.time()
    main_out_dir = Path("outputs/post_validation_round/calibrations")
    # main_out_dir = Path("outputs/validation_round_calibration")
    # main_out_dir = Path("outputs/quick_test_runs/calibrations2")
    # main_out_dir = Path(".local/mind-runner_local/validation_round_calibration")

    # --- Regex
    pattern = "??_[0-9][0-9][0-9][0-9]"
    subdirs = list(glob.glob(str(main_out_dir / pattern)))
    subdirs.sort()

    # -()- Call to main() approach
    # ============================
    from scripts.calc_outbreak_features_from_posteriors import main as program_main

    def _task(subdir):
        program_argv = f"-o {subdir}".split(" ")
        program_main(program_argv)
    _contents  = list(subdirs)[:]

    map_parallel_or_sequential(
        _task, _contents, ncpus=8
    )

    # =========

    xtf = time.time()

    print(f" === Total time taken: {xtf - xt0:.2f} seconds ===")


if __name__ == "__main__":
    main()
