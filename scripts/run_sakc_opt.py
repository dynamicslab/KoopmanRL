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
            "koopmanrl.sakc_optuna_opt.py",
            "--env_id=LinearSystem-v0",
            "--output_file=sakc_linear_system_hparams",
        ],
        log_file="linear_system_opt.txt",
    )


def run_fluid_flow() -> None:
    run_command(
        [
            "python",
            "-m",
            "koopmanrl.sakc_optuna_opt.py",
            "--env_id=FluidFlow-v0",
            "--output_file=sakc_fluid_flow_hparams",
        ],
        log_file="fluid_flow_opt.txt",
    )


def run_lorenz() -> None:
    run_command(
        [
            "python",
            "-m",
            "koopmanrl.sakc_optuna_opt.py",
            "--env_id=Lorenz-v0",
            "--output_file=sakc_lorenz_hparams",
        ],
        log_file="lorenz_opt.txt",
    )


def run_double_well() -> None:
    run_command(
        [
            "python",
            "-m",
            "koopmanrl.sakc_optuna_opt.py",
            "--env_id=DoubleWell-v0",
            "--output_file=sakc_double_well_hparams",
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
