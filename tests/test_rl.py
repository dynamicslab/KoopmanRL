import pytest

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
