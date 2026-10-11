"""Run the SKVI hyperparameter optimisation on the four benchmarks, one after the other.

    uv run -m koopmanrl_utils.run_skvi_optimization

Each benchmark is one call of `python -m koopmanrl.skvi_optuna_opt --env_id=<env>
--output_file=skvi_<env_slug>_hparams` (Ray Tune with Optuna search), whose output is logged to
`<env_slug>_opt.txt` in the working directory. The best configuration of each study is written to
`<storage_dir>/skvi_<env_slug>_hparams.json`, the names of the files in `configurations/`.

The script takes no arguments: any argument, `--help` included, is ignored and the four studies start. The
studies run with the defaults of `koopmanrl.skvi_optuna_opt`, so before running it check there:
`--cpu_cores_per_trial` (default 28; Ray cannot schedule a trial that asks for more cores than the machine has) and
`--storage_dir` (the default is a path on the authors' machine, to which the result is written only at the end of a
study). To run a single study with other settings, call `koopmanrl.skvi_optuna_opt` directly.
"""

import subprocess
from typing import List


def run_command(command: List[str], log_file: str) -> None:
    """Run a shell command, log stdout/stderr to a file, and raise on failure."""
    print(f"Running: {' '.join(command)} > {log_file}", flush=True)
    with open(log_file, "w") as f:
        subprocess.run(command, check=True, stdout=f, stderr=subprocess.STDOUT)


def run_linear_system() -> None:
    run_command(
        [
            "python",
            "-m",
            "koopmanrl.skvi_optuna_opt",
            "--env_id=LinearSystem-v0",
            "--output_file=skvi_linear_system_hparams",
        ],
        log_file="linear_system_opt.txt",
    )


def run_fluid_flow() -> None:
    run_command(
        [
            "python",
            "-m",
            "koopmanrl.skvi_optuna_opt",
            "--env_id=FluidFlow-v0",
            "--output_file=skvi_fluid_flow_hparams",
        ],
        log_file="fluid_flow_opt.txt",
    )


def run_lorenz() -> None:
    run_command(
        [
            "python",
            "-m",
            "koopmanrl.skvi_optuna_opt",
            "--env_id=Lorenz-v0",
            "--output_file=skvi_lorenz_hparams",
        ],
        log_file="lorenz_opt.txt",
    )


def run_double_well() -> None:
    run_command(
        [
            "python",
            "-m",
            "koopmanrl.skvi_optuna_opt",
            "--env_id=DoubleWell-v0",
            "--output_file=skvi_double_well_hparams",
        ],
        log_file="double_well_opt.txt",
    )


def main() -> None:
    """Run Optuna-based hyperparameter optimization for all environments.

    It assumes it is run from the repository root.
    """
    run_linear_system()
    run_fluid_flow()
    run_lorenz()
    run_double_well()


if __name__ == "__main__":
    main()
