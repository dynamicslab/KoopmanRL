import os
import subprocess
import sys

import pytest
import torch
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


def _same_seed_runs(module, options, tmp_path):
    """Working directories of two runs of a module on the linear system with seed 7."""
    cwds = []
    for repeat in ["first", "second"]:
        cwd = tmp_path / repeat
        cwd.mkdir()
        result = subprocess.run(
            [sys.executable, "-m", module, "--env_id=LinearSystem-v0", "--seed=7", *options],
            cwd=cwd,
            capture_output=True,
            text=True,
            env={**os.environ, "OMP_NUM_THREADS": "1"},  # one numerical thread keeps the two runs short
        )
        assert result.returncode == 0, result.stderr
        cwds.append(cwd)
    return cwds


def test_sakc_run_is_reproducible_from_its_seed(tmp_path):
    # training starts early, so that the two runs are compared through updates of the critic, which is trained
    # through the Koopman tensor; the tensor is identified from few transitions, which keeps the runs short
    options = ["--total_timesteps=600", "--learning_starts=200", "--num_paths=20", "--num_steps_per_path=100"]
    cwds = _same_seed_runs("koopmanrl.soft_actor_koopman_critic", options, tmp_path)
    returns = [_episodic_returns(cwd) for cwd in cwds]
    assert len(returns[0]) >= 2 and returns[0] == returns[1]


def test_skvi_run_is_reproducible_from_its_seed(tmp_path):
    # few epochs on small batches. The returns of SKVI do not show a difference in the last digits of the tensor
    # or of the value-function weights, so the weights that the run saves after its last epoch are compared as well
    options = ["--total_timesteps=200", "--num_training_epochs=10", "--batch_size=1024"]
    cwds = _same_seed_runs("koopmanrl.soft_koopman_value_iteration", options, tmp_path)
    returns = [_episodic_returns(cwd) for cwd in cwds]
    assert len(returns[0]) >= 1 and returns[0] == returns[1]
    weights = [torch.load(path) for cwd in cwds for path in cwd.rglob("epoch_10.pt")]
    assert len(weights) == 2 and torch.equal(weights[0], weights[1])
