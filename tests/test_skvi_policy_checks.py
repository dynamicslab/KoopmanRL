"""The helpers of koopmanrl_utils.skvi_policy_checks reproduce the package's SKVI quantities."""

import contextlib
import io

import gym
import numpy as np
import torch

import koopmanrl.environments  # noqa: F401
from koopmanrl.soft_koopman_value_iteration import (
    DiscreteKoopmanValueIterationPolicy,
    generate_koopman_tensor,
)
from koopmanrl_utils.skvi_policy_checks import (
    gibbs_policy,
    monomials,
    quadratic_cost,
    quadratic_form,
)


def small_policy(env_id="LinearSystem-v0", seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        env = gym.make(env_id)
        tensor = generate_koopman_tensor(env_id, seed, 5, 50, 2, 2, "ols")
    actions = torch.from_numpy(np.linspace(env.action_space.low, env.action_space.high, 11)).T
    policy = DiscreteKoopmanValueIterationPolicy(
        env_id, 0.99, 1.0, tensor, actions, env.vectorized_cost_fn, seed, use_ols=True, dt=None
    )
    policy.value_function_weights = torch.from_numpy(np.random.randn(tensor.phi_dim, 1))
    return env, policy


def test_quadratic_cost_matches_environment():
    env, policy = small_policy()
    states = np.random.randn(4, 3)
    expected = env.vectorized_cost_fn(torch.from_numpy(states), policy.all_actions.T).numpy()
    actual = quadratic_cost(env)(states, policy.all_actions.T.numpy()).numpy()
    assert np.allclose(actual, expected)


def test_gibbs_policy_matches_package():
    _, policy = small_policy()
    states = np.array([[1.0, 2.0, -3.0], [10.0, -5.0, 0.5]])
    batched = gibbs_policy(policy, states)
    for n, state in enumerate(states):
        # the package's pis takes one state as a (1, d) row and returns identical columns
        single = policy.pis(torch.from_numpy(state.reshape(1, -1))).detach().numpy()[:, 0]
        assert np.allclose(batched[:, n], single, atol=1e-12)


def test_quadratic_form_reads_the_degree_two_terms():
    names, powers = monomials(2, 2)
    coefficients = np.zeros(len(names))
    coefficients[names.index("x1^2")] = 3.0
    coefficients[names.index("x1x2")] = 4.0
    coefficients[names.index("x2^2")] = 5.0
    assert np.allclose(quadratic_form(coefficients, powers), [[3.0, 2.0], [2.0, 5.0]])
