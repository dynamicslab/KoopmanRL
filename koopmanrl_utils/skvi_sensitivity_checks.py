"""Accuracy of the fitted Koopman tensor along the trained SKVI policy, and sensitivity of SKVI's control.

The Koopman tensor is identified from random-agent data, as in `koopmanrl.soft_koopman_value_iteration`, while the
trained policy visits other states. The first study measures how accurate the fitted model is along the policy. The
second retrains SKVI with other dictionaries, action costs and identification data, and compares its control with
references that do not use the tensor.

Accuracy along the policy (`--study accuracy`; all four benchmarks)
    The tensor is evaluated on transitions of the trained SKVI policy and on fresh random-agent transitions. The
    one-step error is the root mean square, over transitions n and nonconstant dictionary functions j, of the
    normalised prediction error

        (K^{u_n} phi(x_n) - E[phi(x'_n) | x_n, u_n])_j / s_j,

    where the conditional mean is phi(x'_n) itself for the deterministic benchmarks and the exact conditional mean of
    the quadratic dictionary under one Euler-Maruyama step for the double well. The scale s_j is the root mean square
    of the target on the random-agent transitions and is used for both distributions ("common scale"), so that the two
    errors are comparable. Recomputing it on the data being evaluated ("own scale") inflates the error near the
    target, where the high-degree functions are tiny, by orders of magnitude, so that version is recorded too. The same
    error with phi(x_n) in place of the prediction is the persistence baseline (predicting no change). The error of
    the predicted state is reported in the units of the state. The run also records the one-step Jacobian of the
    fitted model at the target against that of the environment's one-step map, and the distance from the target of the
    closed loop and of the uncontrolled system.

Sensitivity of the control (`--study sensitivity`; double well and Lorenz)
    SKVI is retrained with other orders of the state dictionary, with the action-cost weight R multiplied by the time
    step (actions a hundred times cheaper relative to the state cost), and after re-identifying the tensor on
    transitions of the trained policy. Each variant is compared with no control and with the LQR controller of the
    linearisation at the target, on the same episodes. Only SKVI is retrained.

Usage (from the repository root):

    uv run -m koopmanrl_utils.skvi_sensitivity_checks --study accuracy           # four benchmarks, seeds 100-107
    uv run -m koopmanrl_utils.skvi_sensitivity_checks --study sensitivity        # double well and Lorenz
    uv run -m koopmanrl_utils.skvi_sensitivity_checks --study sensitivity --environments Lorenz --seeds 100 101
    uv run -m koopmanrl_utils.skvi_sensitivity_checks --summarize_only

Each run writes one JSON file into `--output_dir` (default `skvi_sensitivity_checks_results/`, not tracked by git),
with the settings it used; the summary refuses to combine runs made with different settings. Random draws use NumPy's
global generator, seeded per episode or per data set in a fixed order, so a run is reproducible from its seed. Tensors
and policies are built by `skvi_policy_checks.train_skvi`, as in `koopmanrl.soft_koopman_value_iteration`, with the
tuned configurations in `configurations/`.

The results were first produced by exploratory scripts that this module consolidates. With the same seeds it
reproduced their numbers exactly, except the one-step Jacobians of the fitted model recorded by the sensitivity study,
which it evaluates in double precision with a step of 1e-4 (differences below 1e-7). Since d0d3016 the regressions of
`koopmanrl.soft_koopman_value_iteration` use the LAPACK driver `gelsd` instead of the default `gelsy`; on the tuned
configurations the tensors of the two drivers agree to rounding, so the numbers are expected to agree to rounding rather
than exactly. The study has not been run again with `gelsd`.

Reference: the results of these checks are in the electronic supplementary material of "Koopman-Assisted Reinforcement
Learning" (Rozwood, Mehrez, Paehler, Sun and Brunton), section "Accuracy along the learned policy and sensitivity of
SKVI".
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
from koopmanrl_utils.skvi_policy_checks import PROBABILITY_FLOOR, monomials, train_skvi

ACCURACY_ENVIRONMENTS = ["LinearSystem", "DoubleWell", "FluidFlow", "Lorenz"]
SENSITIVITY_ENVIRONMENTS = ["DoubleWell", "Lorenz"]

# Steps per closed-loop episode: the benchmark episode of 2,000 steps, and 200 for the discrete-time linear system.
HORIZON = {"LinearSystem-v0": 200, "DoubleWell-v0": 2000, "FluidFlow-v0": 2000, "Lorenz-v0": 2000}

# A state is near the target when its Euclidean distance from the target is below this radius; the sensitivity study
# reports the fraction of each episode spent near the target.
NEAR_RADIUS = {"DoubleWell-v0": 0.25, "Lorenz-v0": 2.0}
# A seed counts as failed when its mean-action policy is near the target for less than this fraction of the episode.
FAILURE_FRACTION = 0.001

# Variants of the sensitivity study: (name, state-dictionary order, action-cost scale, rounds of re-identification,
# what the policy transitions are used for: "both" re-identifies, "tensor" or "value" separates the two effects, see
# `skvi_policy_checks.train_skvi`). The action-cost scale 0.01 is R * dt for both systems (dt = 0.01).
VARIANTS = {
    "DoubleWell-v0": [
        ("order 2", 2, 1.0, 0, "both"),
        ("order 2, R*dt", 2, 0.01, 0, "both"),
        ("order 4, R*dt", 4, 0.01, 0, "both"),
        ("order 5, R*dt", 5, 0.01, 0, "both"),
        ("order 6, R*dt", 6, 0.01, 0, "both"),
    ],
    "Lorenz-v0": [
        ("order 3", 3, 1.0, 0, "both"),
        ("order 3, refit x1", 3, 1.0, 1, "both"),
        ("order 3, refit x1, tensor only", 3, 1.0, 1, "tensor"),
        ("order 3, refit x1, value states only", 3, 1.0, 1, "value"),
        ("order 4", 4, 1.0, 0, "both"),
        ("order 4, refit x1", 4, 1.0, 1, "both"),
        ("order 4, refit x2", 4, 1.0, 2, "both"),
    ],
}
# LQR references of the sensitivity study: (name, action-cost scale).
LQR_REFERENCES = {
    "DoubleWell-v0": [("LQR, benchmark R", 1.0), ("LQR, R*dt", 0.01)],
    "Lorenz-v0": [("LQR, benchmark R", 1.0)],
}


class ArgumentParser(Tap):
    study: str = "accuracy"  # "accuracy" (model error along the policy) or "sensitivity" (SKVI retrained in variants)
    environments: list[str] = []  # default: all four benchmarks for accuracy, double well and Lorenz for sensitivity
    seeds: list[int] = list(range(100, 108))  # one SKVI training per seed and variant
    output_dir: str = "skvi_sensitivity_checks_results"  # one JSON file per study, environment and seed
    accuracy_episodes: int = 16  # closed-loop episodes sampled for the accuracy study
    random_agent_paths: int = 20  # random-agent trajectories for the reference distribution
    evaluation_episodes: int = 10  # closed-loop episodes per controller in the sensitivity study
    variants: list[str] = []  # sensitivity study: run only these variants (default: all); each variant is seeded alone
    summarize_only: bool = False  # skip the runs and summarize the files already in output_dir


# --------------------------------------------------------------------------------------------------------------------
# Training and the deployed policy
# --------------------------------------------------------------------------------------------------------------------


def gibbs_sampler(policy):
    """Fast deployed policy for one state at a time: (probabilities(x), action grid).

    probabilities(x) returns the Gibbs weights over the action grid, computed as in the package (score
    -(c(x, u) + discount * w^T K^u phi(x)) / alpha, shifted by its maximum, exponentiated, floored by machine epsilon
    and normalised), in double precision with w^T K^u precomputed for every grid action.
    """
    tensor = policy.dynamics_model
    K_u = tensor.K_(policy.all_actions).to(torch.float64)  # (L, d_phi, d_phi)
    w = policy.value_function_weights.reshape(-1).to(torch.float64)
    continuation = torch.einsum("p,apq->aq", w, K_u).numpy()  # (L, d_phi)
    grid = policy.all_actions.numpy().reshape(-1)

    def probabilities(x):
        x = np.asarray(x, dtype=np.float64)
        phi = tensor.phi(torch.from_numpy(x.reshape(-1, 1)).to(tensor.K.dtype)).numpy().astype(np.float64).reshape(-1)
        cost = policy.cost(x.reshape(1, -1), grid.reshape(-1, 1)).numpy().reshape(-1)
        score = -(cost + policy.discount_factor * (continuation @ phi)) / policy.alpha
        unnormalised = np.exp(score - score.max()) + PROBABILITY_FLOOR
        return unnormalised / unnormalised.sum()

    return probabilities, grid


def deployed_controller(policy, mean=False):
    """Controller x -> action of the deployed policy: a sampled grid action, or with mean=True the mean action."""
    probabilities, grid = gibbs_sampler(policy)

    def act(x):
        p = probabilities(x)
        if mean:
            return np.array([float(p @ grid)])
        return np.array([grid[np.random.choice(len(p), p=p)]])

    return act


# --------------------------------------------------------------------------------------------------------------------
# Episodes and data
# --------------------------------------------------------------------------------------------------------------------


def closed_loop_episode(env, controller, seed, horizon):
    """States, actions and next states of one episode from the environment's initial state, seeded with `seed`."""
    np.random.seed(seed)
    observation = env.reset()
    X, U, Y = [], [], []
    for _ in range(horizon):
        x = np.asarray(observation, dtype=np.float64).copy()
        u = controller(x)
        observation, _, done, _ = env.step(u)
        X.append(x), U.append(float(u[0])), Y.append(np.asarray(observation, dtype=np.float64).copy())
        if done:
            break
    return np.array(X), np.array(U), np.array(Y)


def random_agent_data(env, seed, paths, steps):
    """Transitions of the uniform random agent that identifies the tensor, from fresh trajectories."""
    np.random.seed(seed)
    env.action_space.seed(seed)
    X, U, Y = [], [], []
    for _ in range(paths):
        observation = env.reset()
        for _ in range(steps):
            x = np.asarray(observation, dtype=np.float64).copy()
            u = env.action_space.sample()
            observation, _, _, _ = env.step(u)
            (
                X.append(x),
                U.append(float(np.asarray(u).reshape(-1)[0])),
                Y.append(np.asarray(observation, dtype=np.float64).copy()),
            )
    return np.array(X), np.array(U), np.array(Y)


def policy_transitions(env, policy, paths, steps, seed):
    """Transitions of the trained (sampled) policy from the environment's initial states, for re-identification."""
    act = deployed_controller(policy)
    np.random.seed(seed)
    X, U, Y = [], [], []
    for _ in range(paths):
        observation = env.reset()
        for _ in range(steps):
            x = np.asarray(observation, dtype=np.float64)
            u = act(x)
            observation, _, done, _ = env.step(u)
            X.append(x), U.append(u), Y.append(np.asarray(observation, dtype=np.float64))
            if done:
                break
    return np.array(X), np.array(U), np.array(Y)


# --------------------------------------------------------------------------------------------------------------------
# One-step accuracy
# --------------------------------------------------------------------------------------------------------------------


def features(tensor, states):
    """Dictionary values (N, d_phi) at `states` (N, d), in double precision."""
    states = np.asarray(states, dtype=np.float64)
    return tensor.phi(torch.from_numpy(states.T).to(tensor.K.dtype)).numpy().astype(np.float64).T


def predict(tensor, Phi, U):
    """Fitted one-step prediction K^u phi(x) per transition, with the action dictionary psi(u) = (1, u, u^2, ...)."""
    T = tensor.K.numpy().astype(np.float64)  # (d_phi, d_phi, d_psi)
    Psi = np.stack([np.asarray(U, dtype=np.float64) ** k for k in range(T.shape[2])], axis=1)
    return np.einsum("ijk,nj,nk->ni", T, Phi, Psi)


def linear_indices(powers):
    """Positions of the degree-one monomials x_1, ..., x_d in the dictionary."""
    d = powers.shape[0]
    return [int(np.where((powers == np.eye(d, dtype=int)[:, [i]]).all(axis=0))[0][0]) for i in range(d)]


def double_well_conditional_mean(env, X, U, powers):
    """Exact E[phi(x') | x, u] and E[x' | x, u] for a dictionary of degree at most two, under one Euler-Maruyama step.

    x' = x + f(x, u) dt + S(x) xi sqrt(dt) with S(x) = [[0.7, x_1], [0, 0.5]] (the environment's noise), so
    E[x'] = m and E[x' x'^T] = m m^T + S S^T dt.
    """
    if powers.sum(axis=0).max() > 2:
        raise ValueError("the closed form holds for dictionaries of degree at most two")
    dt = env.unwrapped.dt
    targets, means = [], []
    for x, u in zip(X, U):
        m = x + np.asarray(env.unwrapped.continuous_f(np.array([u]))(0, x), dtype=np.float64).reshape(-1) * dt
        S = np.array([[0.7, x[0]], [0.0, 0.5]])
        C = S @ S.T * dt
        row = []
        for p in powers.T:
            variables = np.nonzero(p)[0]
            if p.sum() == 0:
                row.append(1.0)
            elif p.sum() == 1:
                row.append(m[variables[0]])
            else:  # x_i^2 or x_i x_j
                i, j = variables[0], variables[-1]
                row.append(m[i] * m[j] + C[i, j])
        targets.append(row), means.append(m)
    return np.array(targets), np.array(means)


def one_step_errors(prediction, target, base, X, next_state, nonconstant, linear, common_scale):
    """Normalised one-step errors of the model and of persistence, at two scales, and the error of the state.

    The normalised error is sqrt(mean_n mean_j ((prediction - target)_nj / s_j)^2) over the nonconstant dictionary
    functions j, with s_j the root mean square of the target either on the data itself ("own scale") or on the
    random-agent data ("common scale", which makes the two distributions comparable). The persistence error is the
    same with `base` (phi(x), no change) in place of the prediction.
    """
    p, t, b = prediction[:, nonconstant], target[:, nonconstant], base[:, nonconstant]
    out = {"n": int(len(X))}
    for name, scale in [("own_scale", np.sqrt(np.mean(t**2, axis=0))), ("common_scale", common_scale)]:
        model = float(np.sqrt(np.mean(((p - t) / scale) ** 2)))
        persistence = float(np.sqrt(np.mean(((b - t) / scale) ** 2)))
        out[name] = {"model": model, "persistence": persistence, "ratio": model / persistence}
    error = np.linalg.norm(prediction[:, linear] - next_state, axis=1)
    change = np.linalg.norm(next_state - X, axis=1)
    out["state"] = {"error_rms": float(np.sqrt(np.mean(error**2))), "change_rms": float(np.sqrt(np.mean(change**2)))}
    return out


def one_step_map(env, x, u):
    """The environment's one-step map without noise: the integrated flow, or the Euler drift for the double well."""
    if env.spec.id == "DoubleWell-v0":
        drift = np.asarray(env.unwrapped.continuous_f(np.array([u]))(0, x), dtype=np.float64).reshape(-1)
        return x + drift * env.unwrapped.dt
    return np.asarray(env.unwrapped.f(np.asarray(x, dtype=np.float64), np.array([u])), dtype=np.float64).reshape(-1)


def local_dynamics(env, tensor, linear, h=1e-4):
    """One-step Jacobians (in x and in u) at the target, of the fitted model and of the environment's one-step map."""
    target = np.asarray(env.unwrapped.reference_point, dtype=np.float64)
    d = len(target)

    def model(x, u):
        return predict(tensor, features(tensor, x.reshape(1, -1)), np.array([u]))[0, linear]

    def jacobians(step):
        A = np.stack([(step(target + h * e, 0.0) - step(target - h * e, 0.0)) / (2 * h) for e in np.eye(d)], axis=1)
        B = (step(target, h) - step(target, -h)) / (2 * h)
        return A, B

    A_model, B_model = jacobians(model)
    A_true, B_true = jacobians(lambda x, u: one_step_map(env, x, u))
    return {
        "A_model": A_model.tolist(),
        "A_true": A_true.tolist(),
        "B_model": B_model.tolist(),
        "B_true": B_true.tolist(),
        "max_abs_entry_error": float(np.abs(A_model - A_true).max()),
        "relative_error": float(np.linalg.norm(A_model - A_true) / np.linalg.norm(A_true - np.eye(d))),
        "eigenvalue_moduli_model": sorted(np.abs(np.linalg.eigvals(A_model)).tolist()),
        "eigenvalue_moduli_true": sorted(np.abs(np.linalg.eigvals(A_true)).tolist()),
    }


def accuracy_run(env_id, seed, args):
    """Accuracy study for one environment and seed: one-step error along the policy and on random-agent data."""
    start = time.time()
    env, tensor, policy, config = train_skvi(env_id, seed)
    d = env.observation_space.shape[0]
    _, powers = monomials(d, config["state-order"])
    nonconstant = powers.sum(axis=0) > 0
    linear = linear_indices(powers)
    target = np.asarray(env.unwrapped.reference_point, dtype=np.float64)
    act = deployed_controller(policy)
    horizon = HORIZON[env_id]

    # transitions of the policy: every few steps of each episode, so that every phase of the episode is represented
    on_policy, distance, distance_free = [], [], []
    for episode in range(args.accuracy_episodes):
        X, U, Y = closed_loop_episode(env, act, seed * 1000 + episode, horizon)
        stride = max(1, len(X) // 250)
        on_policy.append((X[::stride], U[::stride], Y[::stride]))
        distance.append(np.linalg.norm(X - target, axis=1))
        X_free, _, _ = closed_loop_episode(env, lambda x: np.zeros(1), seed * 1000 + episode, horizon)
        distance_free.append(np.linalg.norm(X_free - target, axis=1))
    X_on, U_on, Y_on = (np.concatenate(a) for a in zip(*on_policy))

    X_r, U_r, Y_r = random_agent_data(env, seed + 50_000, args.random_agent_paths, config["num-steps-per-path"])
    keep = np.random.default_rng(seed).choice(len(X_r), min(len(X_r), 4000), replace=False)
    X_r, U_r, Y_r = X_r[keep], U_r[keep], Y_r[keep]

    sets = {}
    for name, (X, U, Y) in {"on_policy": (X_on, U_on, Y_on), "random_agent": (X_r, U_r, Y_r)}.items():
        Phi = features(tensor, X)
        if env_id == "DoubleWell-v0":
            target_features, next_state = double_well_conditional_mean(env, X, U, powers)
        else:
            target_features, next_state = features(tensor, Y), Y
        sets[name] = (predict(tensor, Phi, U), target_features, Phi, X, next_state)
    common_scale = np.sqrt(np.mean(sets["random_agent"][1][:, nonconstant] ** 2, axis=0))

    result = {
        "env": env_id,
        "seed": seed,
        "settings": {
            "configuration": config,
            "horizon": horizon,
            "accuracy_episodes": args.accuracy_episodes,
            "random_agent_paths": args.random_agent_paths,
        },
    }
    for name, (prediction, target_features, Phi, X, next_state) in sets.items():
        result[name] = one_step_errors(
            prediction, target_features, Phi, X, next_state, nonconstant, linear, common_scale
        )
    result["local_dynamics"] = local_dynamics(env, tensor, linear)
    second_half = [r[len(r) // 2 :] for r in distance]
    result["distance"] = {
        "final_median": float(np.median([r[-1] for r in distance])),
        "second_half_max": float(max(r.max() for r in second_half)),
        "uncontrolled_final_median": float(np.median([r[-1] for r in distance_free])),
    }
    result["seconds"] = time.time() - start
    write(args.output_dir, "accuracy", env_id, seed, result)
    o, r = result["on_policy"]["common_scale"], result["random_agent"]["common_scale"]
    print(
        f"{env_id} seed {seed}: model error {o['model']:.3g} along the policy vs {r['model']:.3g} on random-agent data "
        f"(persistence {o['persistence']:.3g} vs {r['persistence']:.3g}); local Jacobian relative error "
        f"{result['local_dynamics']['relative_error']:.1e} ({result['seconds'] / 60:.1f} min)",
        flush=True,
    )


# --------------------------------------------------------------------------------------------------------------------
# Sensitivity of the control
# --------------------------------------------------------------------------------------------------------------------


def linearisation(env):
    """Per-step (A, B) of the noise-free dynamics at the target (Euler step of the vector field) and the target."""
    dt = env.unwrapped.dt
    target = np.asarray(env.unwrapped.reference_point, dtype=np.float64)
    d = len(target)

    def f(x, u):
        return np.asarray(env.unwrapped.continuous_f(np.array([u]))(0, x), dtype=np.float64).reshape(-1)

    h = 1e-5
    J = np.stack([(f(target + h * e, 0.0) - f(target - h * e, 0.0)) / (2 * h) for e in np.eye(d)], axis=1)
    b = (f(target, h) - f(target, -h)) / (2 * h)
    return np.eye(d) + dt * J, dt * b.reshape(-1, 1), target


def lqr_controller(env, action_cost_scale):
    """LQR controller of the linearisation at the target, u = clip(-K (x - x_ref)), and its gain K.

    The reference is the per-step (undiscounted) LQR with the environment's Q and the scaled R, a benchmark of what a
    linear controller achieves near the target, not the optimum of the discounted nonlinear problem.
    """
    A, B, target = linearisation(env)
    Q = np.asarray(env.unwrapped.Q, dtype=np.float64)
    R = np.asarray(env.unwrapped.R, dtype=np.float64) * action_cost_scale
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    low, high = env.action_space.low, env.action_space.high
    return (lambda x: np.clip(-K @ (x - target), low, high)), K


def evaluate(env, controller, seed, episodes, horizon, near_radius):
    """Benchmark return, its state and action parts, and how close the state stays to the target, per episode mean.

    Episode e is seeded with seed * 1000 + e, so every controller meets the same initial states (and, for the double
    well, the same noise). The state and action costs use the benchmark weights Q and R. "Near" means a Euclidean
    distance from the target below `near_radius`; the late distance is the mean distance over the last quarter.
    """
    target = np.asarray(env.unwrapped.reference_point, dtype=np.float64)
    Q, R = np.asarray(env.unwrapped.Q), np.asarray(env.unwrapped.R)
    totals = {"return": [], "state_cost": [], "action_cost": [], "late_distance": [], "fraction_near": []}
    for episode in range(episodes):
        np.random.seed(seed * 1000 + episode)
        observation = env.reset()
        total, state_cost, action_cost, distance = 0.0, 0.0, 0.0, []
        for _ in range(horizon):
            x = np.asarray(observation, dtype=np.float64)
            u = controller(x)
            state_cost += float((x - target) @ Q @ (x - target))
            action_cost += float(u @ R @ u)
            observation, reward, done, _ = env.step(u)
            total += float(reward)
            distance.append(float(np.linalg.norm(x - target)))
            if done:
                break
        distance = np.array(distance)
        totals["return"].append(total)
        totals["state_cost"].append(state_cost)
        totals["action_cost"].append(action_cost)
        totals["late_distance"].append(float(distance[-len(distance) // 4 :].mean()))
        totals["fraction_near"].append(float((distance < near_radius).mean()))
    return {key: float(np.mean(values)) for key, values in totals.items()}


def gain_near_target(policy, env, samples=3000):
    """Least-squares gain K of the mean action, u = -K (x - x_ref), on states near the target.

    The states are drawn uniformly from the cube of half-width 0.3 around the target with NumPy's global generator,
    which `sensitivity_run` leaves in the state reached after the mean-action evaluation. This order is kept so that
    the gains equal those of the scripts these checks were first run with. The drawn states, and hence the gain,
    therefore depend on that call order and on the episode count.
    """
    target = np.asarray(env.unwrapped.reference_point, dtype=np.float64)
    act = deployed_controller(policy, mean=True)
    states = target + np.random.uniform(-0.3, 0.3, size=(samples, len(target)))
    actions = np.array([act(x) for x in states])
    return (-np.linalg.lstsq(states - target, actions, rcond=None)[0]).reshape(-1).tolist()


def sensitivity_run(env_id, seed, args):
    """Sensitivity study for one environment and seed: the references, then every variant of VARIANTS[env_id]."""
    start = time.time()
    env = gym.make(env_id)
    near = NEAR_RADIUS[env_id]
    horizon = HORIZON[env_id]
    result = {
        "env": env_id,
        "seed": seed,
        "settings": {
            "horizon": horizon,
            "evaluation_episodes": args.evaluation_episodes,
            "near_radius": near,
            "variants": VARIANTS[env_id],
            "variant_subset": args.variants,
            "lqr_references": LQR_REFERENCES[env_id],
        },
        "references": {},
        "variants": {},
    }

    references = [("no control", lambda x: np.zeros(1), None)]
    for name, scale in LQR_REFERENCES[env_id]:
        controller, K = lqr_controller(env, scale)
        references.append((name, controller, K))
    for name, controller, K in references:
        summary = evaluate(env, controller, seed, args.evaluation_episodes, horizon, near)
        result["references"][name] = {"evaluation": summary, "gain": None if K is None else K.ravel().tolist()}

    for name, order, scale, rounds, used_for in VARIANTS[env_id]:
        if args.variants and name not in args.variants:
            continue
        env, tensor, policy, config = train_skvi(env_id, seed, state_order=order, action_cost_scale=scale)
        transitions = None
        for round_ in range(rounds):  # re-identify on the random-agent data plus all transitions of the policies so far
            paths, steps = max(1, config["num-paths"] // 2), config["num-steps-per-path"]
            new = policy_transitions(env, policy, paths, steps, seed + 50_000 + round_)
            transitions = (
                new if transitions is None else tuple(np.concatenate([a, b]) for a, b in zip(transitions, new))
            )
            env, tensor, policy, config = train_skvi(
                env_id,
                seed,
                state_order=order,
                action_cost_scale=scale,
                extra_transitions=transitions,
                extra_transitions_for=used_for,
            )
        _, powers = monomials(env.observation_space.shape[0], order)
        dynamics = local_dynamics(env, tensor, linear_indices(powers))
        entry = {
            "state_order": order,
            "action_cost_scale": scale,
            "reidentification_rounds": rounds,
            "policy_transitions_used_for": used_for,
            "policy_transitions": 0 if transitions is None else int(len(transitions[0])),
            "sampled": evaluate(env, deployed_controller(policy), seed, args.evaluation_episodes, horizon, near),
            "mean_action": evaluate(
                env, deployed_controller(policy, mean=True), seed, args.evaluation_episodes, horizon, near
            ),
            "gain": gain_near_target(policy, env),
            "A_model": dynamics["A_model"],
        }
        result["variants"][name] = entry
        print(
            f"{env_id} seed {seed} {name:>18}: return {entry['sampled']['return']:.4g} (mean action "
            f"{entry['mean_action']['return']:.4g}), near target {100 * entry['sampled']['fraction_near']:.0f}%, "
            f"gain {np.round(entry['gain'], 2).tolist()}",
            flush=True,
        )
    result["seconds"] = time.time() - start
    write(args.output_dir, "sensitivity", env_id, seed, result)


# --------------------------------------------------------------------------------------------------------------------
# Results and summaries
# --------------------------------------------------------------------------------------------------------------------


def write(output_dir, study, env_id, seed, result):
    """Write one run, refusing to replace a file of the same run made with other settings (e.g. another variant set)."""
    path = os.path.join(output_dir, f"{study}_{env_id.split('-')[0]}_{seed}.json")
    if os.path.exists(path):
        with open(path) as f:
            previous = json.load(f).get("settings")
        if json.dumps(previous, sort_keys=True) != json.dumps(result.get("settings"), sort_keys=True):
            raise FileExistsError(f"{path} holds a run made with other settings; use another --output_dir")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)


def check_variant_names(names, environments):
    """Raise if a name passed to --variants is not a variant of the sensitivity study for these environments."""
    known = {name for environment in environments for name, *_ in VARIANTS.get(f"{environment}-v0", [])}
    unknown = sorted(set(names) - known)
    if unknown:
        raise ValueError(f"unknown variants {unknown}; known: {sorted(known)}")


def load_results(output_dir, study, environment):
    """All runs of one study and environment in `output_dir`, which must share their settings."""
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(output_dir, f"{study}_{environment}_*.json")))]
    settings = {json.dumps(run.get("settings"), sort_keys=True) for run in runs}
    if len(settings) > 1:
        raise ValueError(f"{output_dir} holds {study} runs of {environment} made with different settings")
    return runs


def box_corner_distance(env_id):
    """Distance from the target to the farthest corner of the environment's state box."""
    env = gym.make(env_id)
    target = np.asarray(env.unwrapped.reference_point, dtype=np.float64)
    low, high = env.observation_space.low, env.observation_space.high
    return float(np.linalg.norm(np.maximum(np.abs(low - target), np.abs(high - target))))


def summarize(output_dir):
    """Print the medians over seeds of both studies, with the failure counts and the model's local coefficient."""
    for environment in ACCURACY_ENVIRONMENTS:
        runs = load_results(output_dir, "accuracy", environment)
        if not runs:
            continue

        def median(*keys):
            values = []
            for run in runs:
                value = run
                for key in keys:
                    value = value[key]
                values.append(value)
            return float(np.median(values))

        print(
            f"Accuracy, {environment} ({len(runs)} seed{'s' if len(runs) != 1 else ''}): model error "
            f"{median('on_policy', 'common_scale', 'model'):.2g} along the policy, "
            f"{median('random_agent', 'common_scale', 'model'):.2g} on random-agent data; persistence error "
            f"{median('on_policy', 'common_scale', 'persistence'):.2g} and "
            f"{median('random_agent', 'common_scale', 'persistence'):.2g}; state error along the policy "
            f"{median('on_policy', 'state', 'error_rms'):.2g} against a change of "
            f"{median('on_policy', 'state', 'change_rms'):.2g}; local Jacobian: largest entry error "
            f"{max(r['local_dynamics']['max_abs_entry_error'] for r in runs):.1e}, relative error "
            f"{max(r['local_dynamics']['relative_error'] for r in runs):.1e} (worst seed)"
        )

    for environment in SENSITIVITY_ENVIRONMENTS:
        runs = load_results(output_dir, "sensitivity", environment)
        if not runs:
            continue
        print(f"Sensitivity, {environment} ({len(runs)} seed{'s' if len(runs) != 1 else ''}): medians over seeds")
        scale = 0.01  # R * dt (dt = 0.01): the double-well costs are also reported with actions this much cheaper
        rows = [(name, [r["references"][name]["evaluation"] for r in runs]) for name in runs[0]["references"]]
        for name in runs[0]["variants"]:
            rows.append((f"SKVI {name}, deployed", [r["variants"][name]["sampled"] for r in runs]))
            rows.append((f"SKVI {name}, mean action", [r["variants"][name]["mean_action"] for r in runs]))
        for name, evaluations in rows:
            cost = np.median([-e["return"] for e in evaluations])
            cheap = np.median([e["state_cost"] + scale * e["action_cost"] for e in evaluations])
            state = np.median([e["state_cost"] for e in evaluations])
            near = [100 * e["fraction_near"] for e in evaluations]
            over_half = sum(v > 50 for v in near)
            extra = f", cost under R*dt {cheap:,.0f}" if environment == "DoubleWell" else ""
            print(
                f"  {name:<34} cost {cost:>10,.0f}{extra}, state cost {state:,.0f}, near target {np.median(near):.0f}%"
                f" (seeds over half the episode: {over_half} of {len(evaluations)})"
            )
        corner = box_corner_distance(f"{environment}-v0")
        for name in runs[0]["variants"]:
            mean_action = [r["variants"][name]["mean_action"] for r in runs]
            failed = sum(e["fraction_near"] < FAILURE_FRACTION for e in mean_action)
            left_box = sum(e["late_distance"] > corner for e in mean_action)
            print(
                f"  {name:<34} mean action: seeds near the target under {100 * FAILURE_FRACTION:g}% of the episode "
                f"{failed} of {len(runs)}, seeds whose late "
                f"distance exceeds the farthest corner of the state box ({corner:.2f}) {left_box} of {len(runs)}"
            )
        for name in runs[0]["variants"]:
            gains = np.array([r["variants"][name]["gain"] for r in runs])
            first = np.array([r["variants"][name]["A_model"][0][0] for r in runs])
            print(
                f"  {name:<34} gain median {np.round(np.median(gains, axis=0), 2).tolist()}, "
                f"model coefficient A[0, 0] {first.min():.4f} to {first.max():.4f}"
            )


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    if args.study not in ("accuracy", "sensitivity"):
        raise ValueError("--study must be 'accuracy' or 'sensitivity'")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.set_num_threads(4)
    if not args.summarize_only:
        defaults = ACCURACY_ENVIRONMENTS if args.study == "accuracy" else SENSITIVITY_ENVIRONMENTS
        run = accuracy_run if args.study == "accuracy" else sensitivity_run
        if args.variants:
            check_variant_names(args.variants, args.environments or defaults)
        for environment in args.environments or defaults:
            for seed in args.seeds:
                run(f"{environment}-v0", seed, args)
    summarize(args.output_dir)
