"""Run the SAKC hyperparameter optimisation on the four benchmarks, one after the other.

    uv run -m koopmanrl_utils.run_sakc_optimization --storage_dir <directory>

Each benchmark is one call of `python -m koopmanrl.sakc_optuna_opt --env_id=<env>
--output_file=sakc_<env_slug>_hparams --storage_dir=<directory>` (Ray Tune with Optuna search), with the Python
interpreter that runs this script, whose output is logged to `<env_slug>_opt.txt` in the working directory. The best
configuration of each study is written to `<directory>/sakc_<env_slug>_hparams.json`, the names of the files in
`configurations/`; the directory has to exist, and is written into only at the end of a study.

`--storage_dir` is required, and `--help` prints the options without starting a study. `--cpu_cores_per_trial` is
passed on to every study when it is given; otherwise the studies use the default of `koopmanrl.sakc_optuna_opt`
(28; Ray cannot schedule a trial that asks for more cores than the machine has). All other settings are the defaults
of `koopmanrl.sakc_optuna_opt`; to run a single study with other settings, call it directly.
"""

import shlex
import subprocess
import sys
from typing import List, Optional

from tap import Tap

# environment id, and the slug of the names of its log and its output file
ENVIRONMENTS = [
    ("LinearSystem-v0", "linear_system"),
    ("FluidFlow-v0", "fluid_flow"),
    ("Lorenz-v0", "lorenz"),
    ("DoubleWell-v0", "double_well"),
]


class ArgumentParser(Tap):
    storage_dir: str  # Directory the best configuration of every study is written into; it has to exist
    cpu_cores_per_trial: Optional[int] = None  # Number of CPU cores per trial; the default of sakc_optuna_opt if unset


def run_command(command: List[str], log_file: str) -> None:
    """Run a shell command, log stdout/stderr to a file, and raise on failure."""
    print(f"Running: {shlex.join(command)} > {log_file}", flush=True)
    with open(log_file, "w") as f:
        subprocess.run(command, check=True, stdout=f, stderr=subprocess.STDOUT)


def main() -> None:
    """Run Optuna-based hyperparameter optimization for all environments.

    It assumes it is run from the repository root.
    """
    args = ArgumentParser().parse_args()
    options = [f"--storage_dir={args.storage_dir}"]
    if args.cpu_cores_per_trial is not None:
        options.append(f"--cpu_cores_per_trial={args.cpu_cores_per_trial}")

    for env_id, slug in ENVIRONMENTS:
        run_command(
            [
                sys.executable,
                "-m",
                "koopmanrl.sakc_optuna_opt",
                f"--env_id={env_id}",
                f"--output_file=sakc_{slug}_hparams",
                *options,
            ],
            log_file=f"{slug}_opt.txt",
        )


if __name__ == "__main__":
    main()
