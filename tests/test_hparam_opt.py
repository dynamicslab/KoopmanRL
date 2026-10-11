"""Smoke tests of the hyperparameter optimizations `koopmanrl.sakc_optuna_opt` and `koopmanrl.skvi_optuna_opt`.

Each test runs one Ray Tune study with a single sample on one CPU core and checks that the best configuration is
written. By default only the linear system is run, one study per algorithm; the other three environments are marked
`hpo_full` and deselected by `addopts` in pyproject.toml. Run them with `uv run pytest tests/test_hparam_opt.py -m
hpo_full`, or everything with `-m ""`.
"""

import os
import shutil
import tempfile

import pytest

from tests.utils import run_module

TOTAL_TIMESTEPS = 1000
FULL = pytest.mark.hpo_full
ENVS = [
    "LinearSystem-v0",
    pytest.param("FluidFlow-v0", marks=FULL),
    pytest.param("Lorenz-v0", marks=FULL),
    pytest.param("DoubleWell-v0", marks=FULL),
]


@pytest.fixture
def ray_env():
    """Environment for the study, with Ray's temporary directory at a short path.

    Ray places AF_UNIX sockets under RAY_TMPDIR, whose paths may not exceed about 104 bytes on macOS; pytest's tmp_path
    is too long for that, so a short directory under /tmp is used where it exists.
    """
    ray_tmpdir = tempfile.mkdtemp(prefix="ray", dir="/tmp" if os.path.isdir("/tmp") else None)
    env = os.environ.copy()
    env["RAY_TMPDIR"] = ray_tmpdir
    yield env
    shutil.rmtree(ray_tmpdir, ignore_errors=True)


def run_study(module, env_id, storage_dir, env):
    return run_module(
        module,
        [
            "--env_id",
            env_id,
            "--num_samples=1",
            "--max_concurrent=1",
            f"--total_timesteps={TOTAL_TIMESTEPS}",
            "--cpu_cores_per_trial=1",
            f"--storage_dir={storage_dir}",
            "--output_file=best_config",
        ],
        timeout=600,
        env=env,
    )


@pytest.mark.parametrize("env_id", ENVS)
def test_sakc_opt(env_id, tmp_path, ray_env):
    result = run_study("koopmanrl.sakc_optuna_opt", env_id, tmp_path, ray_env)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "best_config.json").is_file()


@pytest.mark.parametrize("env_id", ENVS)
def test_skvi_opt(env_id, tmp_path, ray_env):
    result = run_study("koopmanrl.skvi_optuna_opt", env_id, tmp_path, ray_env)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "best_config.json").is_file()
