"""Read the deployed policy of Soft Koopman Value Iteration (SKVI) off the Koopman tensor, and check the reading.

For each seed, SKVI is trained with the tuned configuration in `configurations/`. With the polynomial action
dictionary psi(u) = (1, u, u^2, ...), the fitted continuation w^T K^u phi(x) is a polynomial in the action whose
coefficients are functions of the state read off the slices of the tensor (`action_fields`), so the deployed Gibbs
policy can be written down in closed form (`policy_reading`). The checks compare that reading with references that
do not use the tensor.

Linear system
    The learned quadratic form of the value function is compared with the Riccati matrix of the discounted linear
    quadratic regulator (LQR), and the gain of the policy's mean action with the discounted LQR gain.

Double well
    The value function is pruned in two ways (odd terms removed; diagonal quadratic only), and the returns of paired
    rollouts and the total-variation distance between the pruned and the full policy are compared. The gain of the
    mean action near the origin and the agreement of the closed-form Gaussian mean with the grid policy are reported.

Usage (from the repository root):

    uv run -m koopmanrl_utils.skvi_policy_checks                       # both environments, seeds 100-107
    uv run -m koopmanrl_utils.skvi_policy_checks --environments LinearSystem --seeds 100 101
    uv run -m koopmanrl_utils.skvi_policy_checks --summarize_only --plot

Each run writes one JSON file into `--output_dir` (default `skvi_policy_checks_results/`, not tracked by git). Random
draws use NumPy's global generator, seeded per run as in `koopmanrl.soft_koopman_value_iteration`, in a fixed order, so
a run is reproducible from its seed.

Reference: the theory and the results of these checks are in the electronic supplementary material of
"Koopman-Assisted Reinforcement Learning" (Rozwood, Mehrez, Paehler, Sun and Brunton), section "Additional validation
and interpretability material".
"""

import contextlib
import glob
import io
import json
import os
import time

import gym
import numpy as np
import torch
from scipy.linalg import solve_discrete_are
from tap import Tap

import koopmanrl.environments  # noqa: F401  (registers the environments with gym)
from koopmanrl.koopman_observables import allMonomialPowers
from koopmanrl.koopman_observables import monomials as monomial_dictionary
from koopmanrl.soft_koopman_value_iteration import (
    DiscreteKoopmanValueIterationPolicy,
    KoopmanTensor,
    Regressor,
    generate_koopman_tensor,
)

# Settings shared with the SKVI script (koopmanrl/soft_koopman_value_iteration.py defaults).
GAMMA = 0.99
ALPHA = 1.0
NUM_ACTIONS = 101
BATCH_SIZE = 2**14
PROBABILITY_FLOOR = torch.finfo(torch.float64).eps  # the `delta` added to every action probability by the package

CONFIG_FILES = {
    "LinearSystem-v0": "skvi_linear_system_hparams.json",
    "DoubleWell-v0": "skvi_double_well_hparams.json",
    "FluidFlow-v0": "skvi_fluid_flow_hparams.json",  # used by skvi_sensitivity_checks
    "Lorenz-v0": "skvi_lorenz_hparams.json",  # used by skvi_sensitivity_checks
}
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ArgumentParser(Tap):
    environments: list[str] = ["LinearSystem", "DoubleWell"]  # environments to check
    seeds: list[int] = list(range(100, 108))  # one SKVI run per seed
    output_dir: str = "skvi_policy_checks_results"  # one JSON file per run is written here
    episodes: int = 10  # paired rollout episodes per variant (double well)
    horizon: int = 2000  # steps per rollout episode (double well)
    summarize_only: bool = False  # skip the runs and summarize the files already in output_dir
    plot: bool = False  # also draw the two figures of `plot` into output_dir


# --------------------------------------------------------------------------------------------------------------------
# Training and the deployed policy
# --------------------------------------------------------------------------------------------------------------------


def load_config(env_id):
    """Tuned SKVI configuration for `env_id`, from its file in `configurations/`."""
    with open(os.path.join(REPO_ROOT, "configurations", CONFIG_FILES[env_id])) as f:
        return json.load(f)


def quadratic_cost(env, action_cost_scale=1.0):
    """The environment's cost (x - x_ref)^T Q (x - x_ref) + s u^T R u as an (actions, states) matrix.

    With the default s = 1 this is the benchmark cost, with the same values as `env.vectorized_cost_fn`, which forms
    an N x N intermediate and runs out of memory at the SKVI batch size of 2^14 states. R is diagonal in every
    environment of the package. `skvi_sensitivity_checks` also trains with s = dt, which makes actions cheaper relative
    to the state cost by that factor.
    """
    Q = torch.as_tensor(env.unwrapped.Q, dtype=torch.float64)
    R_diagonal = torch.diag(torch.as_tensor(env.unwrapped.R, dtype=torch.float64)) * action_cost_scale
    reference = torch.as_tensor(env.unwrapped.reference_point, dtype=torch.float64)

    def cost(states, actions):
        deviation = torch.as_tensor(np.asarray(states), dtype=torch.float64) - reference  # (N, d)
        state_cost = torch.einsum("ni,ij,nj->n", deviation, Q, deviation)  # (N,)
        action_cost = torch.as_tensor(np.asarray(actions), dtype=torch.float64) ** 2 @ R_diagonal  # (L,)
        return (state_cost.unsqueeze(-1) + action_cost.unsqueeze(0)).T  # (L, N)

    return cost


def identify_tensor(env_id, seed, config, dt, extra_transitions=None):
    """Identify the Koopman tensor from random-agent data as the SKVI script does, optionally with more transitions.

    extra_transitions  (X, U, Y) arrays of shape (n, d), (n, 1), (n, d) appended to the random-agent data before the
                       tensor is refitted, for example transitions of a trained policy (re-identification along the
                       policy). SKVI then also fits its value function on these states, because it fits it on the
                       identification states. They are rounded to single precision before they are appended (a
                       relative change below 1e-7, far below the model's one-step error), which keeps the results
                       identical to those of the scripts these checks were first run with.
    """
    tensor = generate_koopman_tensor(
        env_id=env_id,
        seed=seed,
        num_paths=config["num-paths"],
        num_steps_per_path=config["num-steps-per-path"],
        state_order=config["state-order"],
        action_order=config["action-order"],
        regressor="ols",
    )
    if extra_transitions is None:
        return tensor
    X_extra, U_extra, Y_extra = (torch.from_numpy(a.T.astype(np.float32)) for a in extra_transitions)
    return KoopmanTensor(
        torch.cat([tensor.X, X_extra], 1),
        torch.cat([tensor.Y, Y_extra], 1),
        torch.cat([tensor.U, U_extra], 1),
        phi=monomial_dictionary(config["state-order"]),
        psi=monomial_dictionary(config["action-order"]),
        regressor=Regressor("ols"),
        **({} if dt is None else {"dt": dt}),
    )


def train_skvi(
    env_id, seed, state_order=None, action_cost_scale=1.0, extra_transitions=None, extra_transitions_for="both"
):
    """Identify the Koopman tensor and train SKVI exactly as the SKVI script does, without writing checkpoints.

    The defaults give the tuned configuration in `configurations/`. The optional arguments, used by
    `skvi_sensitivity_checks`, change the order of the state dictionary, scale the action-cost weight R (in training
    and in the deployed policy) and append transitions to the identification data (see `identify_tensor`). SKVI draws
    the states on which it fits its value function from the identification states, so appended transitions change
    both the tensor and those states. `extra_transitions_for` separates the two: "both" (re-identification), "tensor"
    (the value function is fitted on the random-agent states only) or "value" (the tensor is fitted on the
    random-agent data only and the appended states join the value fit).
    """
    if extra_transitions_for not in ("both", "tensor", "value"):
        raise ValueError("extra_transitions_for must be 'both', 'tensor' or 'value'")
    config = load_config(env_id)
    if state_order is not None:
        config = dict(config, **{"state-order": state_order})
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = gym.make(env_id)
    dt = getattr(env.unwrapped, "dt", None)
    with contextlib.redirect_stdout(io.StringIO()):  # the package prints progress for every epoch
        for_tensor = extra_transitions if extra_transitions_for in ("both", "tensor") else None
        tensor = identify_tensor(env_id, seed, config, dt, for_tensor)
        if extra_transitions is not None and extra_transitions_for != "both":
            random_agent = config["num-paths"] * config["num-steps-per-path"]
            if extra_transitions_for == "tensor":  # fit the value function on the random-agent states only
                tensor.X, tensor.Phi_X = tensor.X[:, :random_agent], tensor.Phi_X[:, :random_agent]
            else:  # add the transitions' states to the value fit, keeping the random-agent tensor
                states = torch.from_numpy(extra_transitions[0].T.astype(np.float32)).to(tensor.X.dtype)
                tensor.X = torch.cat([tensor.X, states], 1)
                tensor.Phi_X = torch.cat([tensor.Phi_X, tensor.phi(states).to(tensor.Phi_X.dtype)], 1)
        actions = torch.from_numpy(np.linspace(env.action_space.low, env.action_space.high, NUM_ACTIONS)).T
        policy = DiscreteKoopmanValueIterationPolicy(
            env_id=env_id,
            gamma=GAMMA,
            alpha=ALPHA,
            dynamics_model=tensor,
            all_actions=actions,
            cost=quadratic_cost(env, action_cost_scale),
            seed=seed,
            use_ols=True,
            dt=dt,
        )
        policy.train(config["number-of-train-epochs"], BATCH_SIZE, 1, how_often_to_chkpt=10**9)
    return env, tensor, policy, config


def gibbs_policy(policy, states):
    """Action probabilities (actions, states) of the deployed SKVI policy at `states` (N, d).

    Identical to `DiscreteKoopmanValueIterationPolicy.pis`, which accepts one state at a time: the score of action u
    is -(c(x, u) + discount * w^T K^u phi(x)) / alpha, shifted by its maximum, exponentiated, floored by machine
    epsilon and normalised over the action grid.
    """
    states = np.asarray(states, dtype=np.float64)
    tensor = policy.dynamics_model
    phi = tensor.phi(torch.from_numpy(states.T).to(tensor.K.dtype))  # (d_phi, N)
    K_u = tensor.K_(policy.all_actions).to(phi.dtype)  # (L, d_phi, d_phi)
    w = policy.value_function_weights.reshape(-1).to(phi.dtype)  # (d_phi,)
    continuation = torch.einsum("p,apq,qn->an", w, K_u, phi)  # (L, N)
    cost = torch.as_tensor(policy.cost(torch.from_numpy(states), policy.all_actions.T), dtype=continuation.dtype)
    score = -(cost + policy.discount_factor * continuation) / policy.alpha
    unnormalised = torch.exp(score - score.amax(dim=0, keepdim=True)) + PROBABILITY_FLOOR
    return (unnormalised / unnormalised.sum(dim=0, keepdim=True)).detach().numpy()


def mean_action(policy, states):
    """Mean action of the deployed policy at `states` (N, d), for a scalar action."""
    grid = policy.all_actions.numpy().reshape(-1, 1)
    return (grid * gibbs_policy(policy, states)).sum(axis=0)


# --------------------------------------------------------------------------------------------------------------------
# Reading the value function and the action fields
# --------------------------------------------------------------------------------------------------------------------


def monomials(state_dim, order):
    """Names and exponent matrix (state_dim, d_phi) of the package's monomial dictionary."""
    powers = allMonomialPowers(state_dim, order).numpy().astype(int)
    names = []
    for k in range(powers.shape[1]):
        factors = [f"x{i + 1}" if p == 1 else f"x{i + 1}^{p}" for i, p in enumerate(powers[:, k]) if p > 0]
        names.append("".join(factors) or "1")
    return names, powers


def quadratic_form(coefficients, powers):
    """Symmetric matrix P with x^T P x equal to the degree-two part of sum_k coefficients[k] * monomial_k(x)."""
    P = np.zeros((powers.shape[0],) * 2)
    for k in range(powers.shape[1]):
        if powers[:, k].sum() != 2:
            continue
        variables = np.nonzero(powers[:, k])[0]
        if len(variables) == 1:  # x_i^2
            i = variables[0]
            P[i, i] = coefficients[k]
        else:  # x_i x_j, split evenly between the two off-diagonal entries
            i, j = variables
            P[i, j] = P[j, i] = coefficients[k] / 2
    return P


def linear_coefficients(coefficients, powers):
    """Coefficients of the degree-one monomials, in state order."""
    linear = np.zeros(powers.shape[0])
    for k in range(powers.shape[1]):
        if powers[:, k].sum() == 1:
            linear[np.argmax(powers[:, k])] = coefficients[k]
    return linear


def action_fields(tensor, w):
    """Coefficient vectors of the fields h_k(x) = w^T T[:, :, k] phi(x), one row per action-dictionary function.

    With the polynomial action dictionary psi(u) = (1, u, u^2, ...), the fitted continuation is a polynomial in the
    action, w^T K^u phi(x) = sum_k h_k(x) u^k, and row k holds the coefficients of h_k on the state dictionary phi.
    The deployed policy weighs action u by exp(-(c(x, u) + discount * sum_k h_k(x) u^k) / alpha), so it is an
    exponential family over the action dictionary with natural parameters -discount * h_k(x) / alpha and depends on w
    only through these fields. This is Proposition S11.1 ("The policy is an exponential family over the action
    dictionary") of the reference in the module docstring. The reference states it for the value function, whose
    fields have the opposite sign, because the package stores the coefficients w of the cost-to-go.
    """
    T = tensor.K.numpy()
    return np.array([T[:, :, k].T @ w for k in range(T.shape[2])])


def discounted_lqr(A, B, Q, R, gamma):
    """Riccati matrix P and gain K (u = -K x) of the discounted LQR."""
    A_discounted, B_discounted = np.sqrt(gamma) * A, np.sqrt(gamma) * B
    P = solve_discrete_are(A_discounted, B_discounted, Q, R)
    K = np.linalg.solve(R + B_discounted.T @ P @ B_discounted, B_discounted.T @ P @ A_discounted)
    return P, K


def relative_error(estimate, reference):
    """Ratio of Euclidean (Frobenius for matrices) norms."""
    return float(np.linalg.norm(np.asarray(estimate) - np.asarray(reference)) / np.linalg.norm(reference))


# --------------------------------------------------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------------------------------------------------


def policy_reading(env_id, env, tensor, policy, w, names, powers):
    """Gain of the mean action against an LQR comparator, and the closed-form Gaussian mean.

    When the action cost is R u^2 and the fields h_k of degree three and higher vanish (`action_fields`), the policy's
    weights on the action grid follow a Gaussian with mean -discount * h_1(x) / (2 (R + discount * h_2(x))) and
    variance alpha / (2 (R + discount * h_2(x))). This is Corollary S11.2 ("Polynomial action dictionary, quadratic
    action cost") of the reference in the module docstring, with the sign of the fields as in `action_fields`. With
    h_2 constant, the degree-one part of h_1 gives the closed-form gain, which is compared with the gain fitted to the
    grid policy's mean action. On the double well the closed-form mean is also compared with the grid mean on a box
    of states.

    Linear system: the comparator is the discounted LQR of the seed's (A, B, Q, R), on states whose LQR action lies
    well inside the action grid (|Kx| < 7 for the grid [-10, 10]). Double well: the comparator is the LQR of the
    drift linearised at the origin, on states near the origin.
    """
    state_dim = powers.shape[0]
    discount = policy.discount_factor
    R = float(env.unwrapped.R[0, 0])
    fields = action_fields(tensor, w)
    constant_index = names.index("1")
    h2_constant = fields[2][constant_index] if fields.shape[0] > 2 else 0.0
    closed_form_gain = discount * linear_coefficients(fields[1], powers) / (2 * (R + discount * h2_constant))

    if env_id == "LinearSystem-v0":
        A, B = env.unwrapped.A, env.unwrapped.B
        P, K = discounted_lqr(A, B, env.unwrapped.Q, env.unwrapped.R, discount)
        states = np.random.uniform(-25, 25, size=(6000, state_dim))
        states = states[np.abs(states @ K.T).reshape(-1) < 7.0]
    else:
        A = np.eye(2) + env.unwrapped.dt * np.diag([4.0, -2.0])
        B = env.unwrapped.dt * np.ones((2, 1))
        P, K = discounted_lqr(A, B, np.eye(2), np.eye(1), discount)
        states = np.random.uniform(-0.3, 0.3, size=(6000, state_dim))
    grid_gain = -np.linalg.lstsq(states, mean_action(policy, states), rcond=None)[0]
    learned_P = quadratic_form(w, powers)

    reading = {
        "fields": fields.tolist(),
        "h2_nonconstant_share": (
            float(np.abs(np.delete(fields[2], constant_index)).sum() / max(np.abs(fields[2]).sum(), 1e-12))
            if fields.shape[0] > 2
            else None
        ),
        "h3_relative": (
            float(np.abs(fields[3]).max() / max(np.abs(fields[1]).max(), 1e-12)) if fields.shape[0] > 3 else None
        ),
        "P_lqr": P.tolist(),
        "P_learned": learned_P.tolist(),
        "P_relative_error": relative_error(learned_P, P),
        "K_lqr": K.reshape(-1).tolist(),
        "K_closed_form": closed_form_gain.tolist(),
        "K_grid": grid_gain.tolist(),
        "K_grid_relative_error": relative_error(grid_gain, K.reshape(-1)),
        "K_closed_form_vs_grid": relative_error(closed_form_gain, grid_gain),
        "discount": discount,
        "n_states": int(len(states)),
    }
    if env_id == "DoubleWell-v0":
        axis = np.linspace(-2, 2, 41)
        X1, X2 = np.meshgrid(axis, axis)
        points = np.c_[X1.ravel(), X2.ravel()]
        grid_mean = mean_action(policy, points)
        h = [sum(c * np.prod(points ** powers[:, k], axis=1) for k, c in enumerate(fields[z])) for z in (1, 2)]
        closed_form_mean = -discount * h[0] / (2 * (R + discount * h[1]))
        reading["mean_action_field"] = {"axis": axis.tolist(), "grid_mean": grid_mean.reshape(41, 41).tolist()}
        reading["closed_form_vs_grid_max_abs"] = float(np.abs(closed_form_mean - grid_mean).max())
    return reading


def rollout_returns(env, policy, seed, episodes, horizon, record_every=None):
    """Returns of `episodes` rollouts of the deployed policy, with common random numbers across calls.

    Episode e is seeded with seed * 1000 + e, so pruned and full policies meet the same initial states and noise.
    With `record_every`, the states visited every that many steps are returned as well.
    """
    returns, visited = [], []
    for episode in range(episodes):
        np.random.seed(seed * 1000 + episode)
        observation = env.reset()
        total = 0.0
        for step in range(horizon):
            if record_every and step % record_every == 0:
                visited.append(np.asarray(observation, dtype=np.float64).copy())
            probabilities = gibbs_policy(policy, np.asarray(observation, dtype=np.float64).reshape(1, -1))[:, 0]
            action = policy.all_actions[0][np.random.choice(len(probabilities), p=probabilities)].reshape(-1).numpy()
            observation, reward, done, _ = env.step(action)
            total += float(reward)
            if done:
                break
        returns.append(total)
    return returns, np.array(visited)


def pruning_experiment(env, tensor, policy, w, powers, seed, episodes, horizon):
    """Double well: does the policy need the small terms of the learned value function?

    Variants: the full value function; the odd terms (x, y) removed; only the constant and the squares kept. Returns
    are compared on paired episodes, and the total-variation distance to the full policy is measured on the states
    the full policy visits.
    """
    states = np.random.uniform(-2, 2, size=(20000, 2))
    phi = tensor.phi(torch.from_numpy(states.T)).numpy()
    degree = powers.sum(axis=0)
    variables = (powers > 0).sum(axis=0)
    masks = {
        "even": (degree % 2 == 0).astype(float),  # drop the odd monomials x and y
        "diag": ((degree % 2 == 0) & (variables <= 1)).astype(float),  # keep 1, x^2 and y^2
    }
    out = {"importance": (np.abs(w) * phi.std(axis=1)).tolist()}

    policy.value_function_weights = torch.from_numpy(w.reshape(-1, 1))
    out["returns_full"], visited = rollout_returns(env, policy, seed, episodes, horizon, record_every=20)
    full = gibbs_policy(policy, visited)
    for variant in ("even", "diag"):
        policy.value_function_weights = torch.from_numpy((w * masks[variant]).reshape(-1, 1))
        tv = 0.5 * np.abs(gibbs_policy(policy, visited) - full).sum(axis=0)
        out[f"tv_{variant}"] = {"mean": float(tv.mean()), "max": float(tv.max())}
        out[f"returns_{variant}"] = rollout_returns(env, policy, seed, episodes, horizon)[0]
    policy.value_function_weights = torch.from_numpy(w.reshape(-1, 1))
    return out


def run(env_id, seed, args):
    """Train SKVI for one seed, run the checks and write the result file."""
    start = time.time()
    env, tensor, policy, config = train_skvi(env_id, seed)
    names, powers = monomials(env.observation_space.shape[0], config["state-order"])
    w = policy.value_function_weights.detach().clone().flatten().numpy()
    result = {"env": env_id, "seed": seed, "names": names, "w": w.tolist(), "train_seconds": time.time() - start}
    result["reading"] = policy_reading(env_id, env, tensor, policy, w, names, powers)
    if env_id == "DoubleWell-v0":
        result.update(pruning_experiment(env, tensor, policy, w, powers, seed, args.episodes, args.horizon))
    result["total_seconds"] = time.time() - start
    path = os.path.join(args.output_dir, f"{env_id.split('-')[0]}_{seed}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    reading = result["reading"]
    if env_id == "LinearSystem-v0":
        detail = (
            f"|P_learned - P|/|P| = {reading['P_relative_error']:.3f}, "
            f"|K_grid - K|/|K| = {reading['K_grid_relative_error']:.3f}"
        )
    else:
        full = np.mean(result["returns_full"])
        changes = [100 * (np.mean(result[f"returns_{v}"]) - full) / abs(full) for v in ("even", "diag")]
        detail = (
            f"gain of the mean action {np.round(reading['K_grid'], 3).tolist()}, "
            f"return change when pruned {changes[0]:+.2f}% / {changes[1]:+.2f}%"
        )
    print(f"{env_id} seed {seed}: {detail} ({result['total_seconds'] / 60:.1f} min)", flush=True)


# --------------------------------------------------------------------------------------------------------------------
# Summary and figures
# --------------------------------------------------------------------------------------------------------------------


def median_iqr(values, digits=3):
    values = np.asarray(values, dtype=float)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return f"{median:.{digits}f} [{q1:.{digits}f}, {q3:.{digits}f}]"


def load_results(output_dir, name):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(output_dir, f"{name}_*.json")))]


def summarize(output_dir):
    """Print the results of all seeds in output_dir: medians and interquartile ranges over seeds, and the extremes."""
    linear = load_results(output_dir, "LinearSystem")
    if linear:
        P_err = [r["reading"]["P_relative_error"] for r in linear]
        K_err = [r["reading"]["K_grid_relative_error"] for r in linear]
        agreement = [r["reading"]["K_closed_form_vs_grid"] for r in linear]
        print(f"Linear system, {len(linear)} seeds")
        print(f"  |P_learned - P|/|P|     median [IQR] {median_iqr(P_err)}, max {max(P_err):.3f}")
        print(f"  |K_grid - K|/|K|        median [IQR] {median_iqr(K_err)}, max {max(K_err):.3f}")
        print(f"  closed form vs grid gain median [IQR] {median_iqr(agreement, 5)}")
        print(f"  h2 non-constant share   max {max(r['reading']['h2_nonconstant_share'] for r in linear):.2e}")
        print(f"  |h3| / |h1|             max {max(r['reading']['h3_relative'] for r in linear):.2e}")

    well = load_results(output_dir, "DoubleWell")
    if well:
        names = well[0]["names"]
        W = np.array([r["w"] for r in well])
        importance = np.array([r["importance"] for r in well])
        share = importance / importance.sum(axis=1, keepdims=True)
        odd_share = share[:, [names.index("x1"), names.index("x2")]].sum(axis=1)
        print(f"Double well, {len(well)} seeds (cost-to-go coefficients as stored by the package)")
        for j, name in enumerate(names):
            share_percent = 100 * np.median(share[:, j])
            print(f"  {name:>5}: coefficient {median_iqr(W[:, j], 2)}, importance share {share_percent:.1f}%")
        print(f"  odd terms together: median {100 * np.median(odd_share):.1f}%, max {100 * odd_share.max():.1f}%")
        full = np.array([np.mean(r["returns_full"]) for r in well])
        print(f"  return, full value function: {median_iqr(full, 1)}")
        for variant in ("even", "diag"):
            pruned = np.array([np.mean(r[f"returns_{variant}"]) for r in well])
            change = 100 * (pruned - full) / np.abs(full)
            tv_max = max(r[f"tv_{variant}"]["max"] for r in well)
            print(
                f"  {variant:>4}: relative change of the return {median_iqr(change, 2)} % "
                f"(min {change.min():+.2f}, max {change.max():+.2f}), largest TV distance {tv_max:.3f}"
            )
        gains = np.array([r["reading"]["K_grid"] for r in well])
        agreement = [r["reading"]["closed_form_vs_grid_max_abs"] for r in well]
        print(f"  gain of the mean action near the origin: {np.round(np.median(gains, axis=0), 3).tolist()}")
        print(f"  closed-form vs grid mean action, largest |difference| on the box: median {np.median(agreement):.1e}")


def plot(output_dir):
    """Draw two figures into output_dir.

    linear_costate.pdf      linear system, all seeds: gain from the mean action against the LQR gain, and the
                            learned quadratic form against the Riccati matrix
    double_well_costate.pdf double well, first seed: the learned cost-to-go and the mean action on [-2, 2]^2, with
                            the direction of the drift
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    linear = load_results(output_dir, "LinearSystem")
    if linear:
        fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.3))
        K = np.concatenate([r["reading"]["K_lqr"] for r in linear])
        K_grid = np.concatenate([r["reading"]["K_grid"] for r in linear])
        upper = np.triu_indices(3)
        P = np.concatenate([np.array(r["reading"]["P_lqr"])[upper] for r in linear])
        P_learned = np.concatenate([np.array(r["reading"]["P_learned"])[upper] for r in linear])
        for axis, (x, y, xlabel, ylabel) in zip(
            ax,
            [
                (K, K_grid, "LQR gain component $K_i$", "gain from the Gibbs mean action"),
                (P, P_learned, "Riccati entry $P_{ij}$", "learned quadratic form"),
            ],
        ):
            axis.scatter(x, y, s=18)
            low, high = min(x.min(), y.min(), 0) * 1.05, max(x.max(), y.max()) * 1.05
            axis.plot([low, high], [low, high], "k--", lw=0.8)
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "linear_costate.pdf"))

    well = load_results(output_dir, "DoubleWell")
    if well:
        result = well[0]
        field = result["reading"]["mean_action_field"]
        axis_values = np.array(field["axis"])
        X1, X2 = np.meshgrid(axis_values, axis_values)
        points = np.c_[X1.ravel(), X2.ravel()]
        _, powers = monomials(2, 2)
        V = sum(c * np.prod(points ** powers[:, k], axis=1) for k, c in enumerate(result["w"])).reshape(X1.shape)
        U = np.array(field["grid_mean"])
        fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.4))
        fig.colorbar(ax[0].contourf(X1, X2, V, 30, cmap="viridis"), ax=ax[0])
        ax[0].set_title("learned soft cost-to-go $\\hat V(x)$")
        largest = np.abs(U).max()
        fig.colorbar(ax[1].contourf(X1, X2, U, 30, cmap="RdBu_r", vmin=-largest, vmax=largest), ax=ax[1])
        ax[1].set_title("Gibbs mean action (arrows: drift direction)")
        coarse = axis_values[::5]
        A1, A2 = np.meshgrid(coarse, coarse)
        F1, F2 = 4 * A1 - 4 * A1**3, -2 * A2
        norm = np.sqrt(F1**2 + F2**2) + 1e-9
        ax[1].quiver(A1, A2, F1 / norm, F2 / norm, color="k", alpha=0.45, scale=28, width=0.004, headwidth=4)
        for axis in ax:
            axis.set_xlabel("$x$")
            axis.set_ylabel("$y$")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "double_well_costate.pdf"))


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.set_num_threads(4)
    if not args.summarize_only:
        for environment in args.environments:
            for seed in args.seeds:
                run(f"{environment}-v0", seed, args)
    summarize(args.output_dir)
    if args.plot:
        plot(args.output_dir)
