import subprocess
import sys

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from tests.utils import run_module

TOTAL_TIMESTEPS = 1000
ENVS = ["LinearSystem-v0", "FluidFlow-v0", "Lorenz-v0", "DoubleWell-v0"]


@pytest.mark.parametrize("env_id", ENVS)
def test_lqr(env_id):
    result = run_module(
        "koopmanrl.linear_quadratic_regulator",
        [f"--env_id={env_id}", f"--total_timesteps={TOTAL_TIMESTEPS}"],
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("env_id", ENVS)
def test_sac_q(env_id):
    result = run_module(
        "koopmanrl.sac_continuous_action",
        [f"--env_id={env_id}", f"--total_timesteps={TOTAL_TIMESTEPS}"],
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("env_id", ENVS)
def test_sac_v(env_id):
    result = run_module(
        "koopmanrl.value_based_sac_continuous_action",
        [f"--env_id={env_id}", f"--total_timesteps={TOTAL_TIMESTEPS}"],
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("env_id", ENVS)
def test_skvi(env_id):
    result = run_module(
        "koopmanrl.soft_koopman_value_iteration",
        [f"--env_id={env_id}", f"--total_timesteps={TOTAL_TIMESTEPS}"],
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("env_id", ENVS)
def test_sakc(env_id):
    result = run_module(
        "koopmanrl.soft_actor_koopman_critic",
        [f"--env_id={env_id}", f"--total_timesteps={TOTAL_TIMESTEPS}"],
    )
    assert result.returncode == 0, result.stderr


def _episodic_returns(run_dir):
    (log,) = [str(p) for p in run_dir.rglob("events.out.tfevents.*")]
    return [(e.step, e.value) for e in EventAccumulator(log).Reload().Scalars("charts/episodic_return")]


@pytest.mark.parametrize(
    "module, options",
    [
        ("koopmanrl.linear_quadratic_regulator", ["--total_timesteps=400"]),
        # the soft actor-critics start to train early, so that the two runs are compared through network updates
        ("koopmanrl.sac_continuous_action", ["--total_timesteps=800", "--learning_starts=200"]),
        ("koopmanrl.value_based_sac_continuous_action", ["--total_timesteps=800", "--learning_starts=200"]),
    ],
)
def test_baseline_run_is_reproducible_from_its_seed(module, options, tmp_path):
    returns = []
    for repeat in ["first", "second"]:
        cwd = tmp_path / repeat
        cwd.mkdir()
        result = subprocess.run(
            [sys.executable, "-m", module, "--env_id=LinearSystem-v0", "--seed=7", *options],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        (run_dir,) = (cwd / "runs").iterdir()
        assert run_dir.name.split("__")[2] == "7"  # the seed of the run is the one that was asked for
        returns.append(_episodic_returns(run_dir))
    assert len(returns[0]) >= 2 and returns[0] == returns[1]
