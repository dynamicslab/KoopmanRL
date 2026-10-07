"""The helpers of koopmanrl_utils.skvi_sensitivity_checks reproduce the quantities they stand for."""

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
    identify_tensor,
    monomials,
    quadratic_cost,
)
from koopmanrl_utils.skvi_sensitivity_checks import (
    check_variant_names,
    double_well_conditional_mean,
    evaluate,
    features,
    gibbs_sampler,
    linear_indices,
    local_dynamics,
    lqr_controller,
    one_step_errors,
    predict,
    write,
)


def small_tensor(env_id="LinearSystem-v0", seed=1, state_order=2, action_order=2):
    np.random.seed(seed)
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        env = gym.make(env_id)
        tensor = generate_koopman_tensor(env_id, seed, 5, 50, state_order, action_order, "ols")
    return env, tensor


def small_policy(env_id="LinearSystem-v0", seed=1):
    env, tensor = small_tensor(env_id, seed)
    actions = torch.from_numpy(np.linspace(env.action_space.low, env.action_space.high, 11)).T
    policy = DiscreteKoopmanValueIterationPolicy(
        env_id, 0.99, 1.0, tensor, actions, quadratic_cost(env), seed, use_ols=True, dt=None
    )
    policy.value_function_weights = torch.from_numpy(np.random.randn(tensor.phi_dim, 1))
    return env, policy


def test_gibbs_sampler_matches_package():
    _, policy = small_policy()
    probabilities, _ = gibbs_sampler(policy)
    for state in np.array([[1.0, 2.0, -3.0], [10.0, -5.0, 0.5]]):
        single = policy.pis(torch.from_numpy(state.reshape(1, -1))).detach().numpy()[:, 0]
        assert np.allclose(probabilities(state), single, atol=1e-12)


def test_predict_matches_tensor():
    _, tensor = small_tensor()
    states = np.random.default_rng(0).normal(size=(4, 3))
    actions = np.array([-1.0, 0.0, 0.5, 2.0])
    Phi = features(tensor, states)
    K_u = tensor.K_(torch.from_numpy(actions.reshape(1, -1)).to(tensor.K.dtype)).numpy().astype(np.float64)
    assert np.allclose(predict(tensor, Phi, actions), np.einsum("nij,nj->ni", K_u, Phi), rtol=1e-5, atol=1e-8)


def test_double_well_conditional_mean_through_the_environment():
    # For a dictionary of degree two the Gaussian expectation equals the average over the four noise draws
    # (+-1, +-1), which match the first and second moments; the draws go through the environment's own step.
    env = gym.make("DoubleWell-v0")
    env.reset()
    _, powers = monomials(2, 2)
    X = np.array([[0.3, -0.4], [-1.2, 0.8]])
    U = np.array([1.5, -2.0])
    exact, mean_state = double_well_conditional_mean(env, X, U, powers)
    for n, (x, u) in enumerate(zip(X, U)):
        next_states = []
        for draw in ([1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]):
            env.unwrapped.step_count = 0
            env.unwrapped.random_draws[0] = np.array(draw).reshape(2, 1)
            next_states.append(env.unwrapped.f(x, np.array([u])))
        next_states = np.array(next_states)
        average = np.prod(next_states[:, :, None] ** powers[None, :, :], axis=1).mean(axis=0)
        assert np.allclose(exact[n], average, atol=1e-12)
        assert np.allclose(mean_state[n], next_states.mean(axis=0), atol=1e-12)


def test_one_step_errors_at_a_common_scale():
    rng = np.random.default_rng(0)
    target = rng.normal(size=(50, 3)) * np.array([1.0, 1.0, 10.0])
    target[:, 0] = 1.0  # constant dictionary function, excluded from the error
    prediction = target + 0.1
    base = target + 1.0
    X, next_state = rng.normal(size=(50, 2)), rng.normal(size=(50, 2))
    nonconstant = np.array([False, True, True])
    out = one_step_errors(prediction, target, base, X, next_state, nonconstant, [1, 2], common_scale=np.ones(2))
    assert np.isclose(out["common_scale"]["model"], 0.1)
    assert np.isclose(out["common_scale"]["persistence"], 1.0)
    assert np.isclose(out["common_scale"]["ratio"], 0.1)
    own = np.sqrt(np.mean(target[:, 1:] ** 2, axis=0))
    assert np.isclose(out["own_scale"]["model"], np.sqrt(np.mean((0.1 / own) ** 2)))
    # the state error compares the predicted degree-one functions (here columns 1 and 2) with the next state
    error = np.linalg.norm(prediction[:, [1, 2]] - next_state, axis=1)
    assert np.isclose(out["state"]["error_rms"], np.sqrt(np.mean(error**2)))
    assert np.isclose(out["state"]["change_rms"], np.sqrt(np.mean(np.linalg.norm(next_state - X, axis=1) ** 2)))


def test_local_dynamics_of_the_linear_system():
    env, tensor = small_tensor()
    _, powers = monomials(3, 2)
    dynamics = local_dynamics(env, tensor, linear_indices(powers))
    assert np.allclose(dynamics["A_true"], env.unwrapped.A, atol=1e-8)
    assert dynamics["max_abs_entry_error"] < 1e-3  # the order-2 dictionary contains the linear dynamics exactly


def test_lqr_controller_stabilises_the_linearisation():
    env = gym.make("DoubleWell-v0")
    for scale in (1.0, 0.01):
        _, K = lqr_controller(env, scale)
        dt = env.unwrapped.dt
        A = np.eye(2) + dt * np.array([[4.0, 0.0], [0.0, -2.0]])  # drift 4x - 4x^3 and -2y at the origin
        B = dt * np.ones((2, 1))
        assert np.abs(np.linalg.eigvals(A - B @ K)).max() < 1


def test_action_cost_scale():
    env = gym.make("DoubleWell-v0")
    states, actions = np.random.default_rng(0).normal(size=(5, 2)), np.linspace(-3, 3, 7).reshape(-1, 1)
    assert torch.equal(quadratic_cost(env, 1.0)(states, actions), quadratic_cost(env)(states, actions))
    cheap = quadratic_cost(env, 0.01)(states, actions) - quadratic_cost(env)(states, actions)
    assert np.allclose(cheap.numpy(), -0.99 * np.asarray(env.unwrapped.R).item() * actions**2)


class ScalarSystem:
    """Deterministic test system x' = x / 2 + u with cost x^2 + 2 u^2, starting at x = 1."""

    def __init__(self):
        self.unwrapped = self
        self.Q, self.R, self.reference_point = np.eye(1), 2 * np.eye(1), np.zeros(1)

    def reset(self):
        self.x = np.ones(1)
        return self.x.copy()

    def step(self, u):
        reward = -float(self.x @ self.Q @ self.x + u @ self.R @ u)
        self.x = self.x / 2 + u
        return self.x.copy(), reward, False, {}


def test_evaluate_sums_costs_and_measures_the_distance():
    env = ScalarSystem()
    free = evaluate(env, lambda x: np.zeros(1), seed=0, episodes=2, horizon=4, near_radius=0.3)
    # states 1, 1/2, 1/4, 1/8: state cost 1 + 1/4 + 1/16 + 1/64, two of four steps within 0.3, last quarter 1/8
    assert np.isclose(free["state_cost"], 1.328125) and free["action_cost"] == 0
    assert np.isclose(free["return"], -1.328125)
    assert free["fraction_near"] == 0.5 and np.isclose(free["late_distance"], 0.125)
    # u = -x/2 sends the state to the target in one step: state cost 1, action cost 2 (1/2)^2
    controlled = evaluate(env, lambda x: -x / 2, seed=0, episodes=1, horizon=4, near_radius=0.3)
    assert np.isclose(controlled["state_cost"], 1.0) and np.isclose(controlled["action_cost"], 0.5)
    assert np.isclose(controlled["return"], -1.5) and controlled["fraction_near"] == 0.75


def test_identify_tensor_appends_transitions_in_single_precision():
    config = {"num-paths": 3, "num-steps-per-path": 20, "state-order": 2, "action-order": 2}
    rng = np.random.default_rng(0)
    extra = (rng.normal(size=(5, 2)), rng.normal(size=(5, 1)), rng.normal(size=(5, 2)))
    with contextlib.redirect_stdout(io.StringIO()):
        base = identify_tensor("DoubleWell-v0", 1, config, 0.01)
        refitted = identify_tensor("DoubleWell-v0", 1, config, 0.01, extra)
    for name, appended in zip(("X", "U", "Y"), extra):
        stored = getattr(refitted, name).numpy()
        assert stored.shape[1] == getattr(base, name).shape[1] + 5
        assert np.array_equal(stored[:, : getattr(base, name).shape[1]], getattr(base, name).numpy())
        assert np.array_equal(stored[:, -5:], appended.T.astype(np.float32).astype(stored.dtype))


def test_runs_with_other_settings_are_not_overwritten(tmp_path):
    write(str(tmp_path), "sensitivity", "Lorenz-v0", 100, {"settings": {"variant_subset": []}})
    write(str(tmp_path), "sensitivity", "Lorenz-v0", 100, {"settings": {"variant_subset": []}})  # a rerun is fine
    try:
        write(str(tmp_path), "sensitivity", "Lorenz-v0", 100, {"settings": {"variant_subset": ["order 3"]}})
    except FileExistsError:
        pass
    else:
        raise AssertionError("a run with other settings replaced the file")


def test_unknown_variant_names_are_rejected():
    check_variant_names(["order 3, refit x1, tensor only"], ["Lorenz"])
    try:
        check_variant_names(["order 3, refit"], ["Lorenz"])
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown variant name was accepted")
