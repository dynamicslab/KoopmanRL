import os
import pytest

from tests.utils import run_module

TOTAL_TIMESTEPS = 1000
ENVS = ["LinearSystem-v0", "FluidFlow-v0", "Lorenz-v0", "DoubleWell-v0"]


@pytest.mark.parametrize("env_id", ENVS)
def test_sakc_opt(env_id, tmp_path):
    env = os.environ.copy()
    env["RAY_TMPDIR"] = str(tmp_path)
    result = run_module(
        "koopmanrl.sakc_optuna_opt",
        [
            "--env_id",
            env_id,
            "--num_samples=1",
            f"--total_timesteps={TOTAL_TIMESTEPS}",
            "--cpu_cores_per_trial=16",
        ],
        timeout=600,
        env=env,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("env_id", ENVS)
def test_skvi_opt(env_id, tmp_path):
    env = os.environ.copy()
    env["RAY_TMPDIR"] = str(tmp_path)
    result = run_module(
        "koopmanrl.skvi_optuna_opt",
        [
            "--env_id",
            env_id,
            "--num_samples=1",
            f"--total_timesteps={TOTAL_TIMESTEPS}",
            "--cpu_cores_per_trial=16",
        ],
        timeout=600,
        env=env,
    )
    assert result.returncode == 0, result.stderr
