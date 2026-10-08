"""Held-out accuracy of the Koopman tensor under different regression algorithms.

The Koopman tensor is identified from the regression phi(x') ~ M kron(psi(u), phi(x)). The results of the paper use
ordinary least squares. This module compares it, on held-out random-agent transitions and with the tuned SKVI
dictionary orders and identification budgets of `configurations/`, with the other regressors of the package and with
eight alternatives, thirteen regressors in all.

Regressors
    As shipped: `ols` (the torch class used by SKVI and SAKC), run through the package's class, and `ols_numpy` (the
    NumPy copy of the class, which solved the normal equations), `ridge`, `sindy` and `rrr` as they were in the
    package when the results of the paper were produced (up to commit b06d974). These four are frozen copies of that
    code, so that the comparison keeps describing it after the package's implementations change; on that commit they
    give the classes' tensors to within the classes' own run-to-run rounding.

    Alternatives, implemented here: `ols_scaled`, `tsvd`, `ridge_cv`, `stlsq_cv`, `lasso_cv`, `rrr_cv`, `tls` and
    `huber`. They share two changes of coordinates that leave ordinary least squares unchanged and matter for every
    regularised estimator. The regression target is the increment phi(x') - phi(x): because psi_0 = 1, phi(x) is a
    block of the regressors, so M = M_inc + [I 0 ... 0], and shrinkage, thresholding and rank truncation act on the
    departure from persistence instead of on the identity. Every regressor column and every target column is divided
    by its root mean square, so that penalties and thresholds are dimensionless. Hyperparameters are chosen from a
    small grid on the last 20% of the identification transitions (whole trajectories, since the data are stored
    trajectory by trajectory) and the model is refitted on all of them.

Error
    The one-step error in dictionary space of `koopman_prediction_validation`: the root mean square, over held-out
    transitions n and nonconstant dictionary functions j, of (K^{u_n} phi(x_n) - E[phi(x'_n) | x_n, u_n])_j / s_j with
    s_j the root mean square of the target. It is reported relative to the same error of the persistence predictor
    phi(x_n), so that 1 means "no better than predicting no change". For the double well the conditional mean is the
    exact one of the quadratic dictionary under one Euler-Maruyama step.

Accuracy study (`--study accuracy`; `accuracy.json`, `accuracy_<benchmark>.dat`, `accuracy_seeds_<benchmark>.dat`,
`accuracy_offscale_<benchmark>.dat`, `accuracy_table.csv`, `regressor_accuracy.tikz`)
    Every regressor on every benchmark at the tuned identification budget, on 12 seeds.

Budget study (`--study budget`; `budget.json`, `budget_<benchmark>.dat`, `regressor_budget.tikz`)
    The same error against the number of identification transitions, on fractions of the tuned budget from 1% to
    100% (whole trajectories, at least two), on 8 seeds. Regularised and sparse estimators can only help where
    ordinary least squares is short of data.

Protocol
    The identification data of seed s are random-agent trajectories collected as in
    `koopmanrl.soft_koopman_value_iteration.generate_koopman_tensor`. The tensor is identified on all of them. The
    held-out transitions are separately generated random-agent trajectories (seed 10000; 20% of the tuned number of
    trajectories, with the tuned length) and are the same for every seed and regressor. The matrix A of the linear
    system, which the environment draws at construction, is fixed (system seed 0), so that the seeds of the linear
    system are data sets of one system.

Usage (from the repository root):

    uv run -m koopmanrl_utils.koopman_regressor_comparison      # both studies, about 40 minutes on one core
    uv run -m koopmanrl_utils.koopman_regressor_comparison --study accuracy
    uv run -m koopmanrl_utils.koopman_regressor_comparison --study budget --environments DoubleWell
    uv run -m koopmanrl_utils.koopman_regressor_comparison --regressors ols lasso_cv --accuracy_seeds 2
    uv run -m koopmanrl_utils.koopman_regressor_comparison --summarize_only      # rewrite tables and figures

The files are written into `--output_dir` (default `koopman_regressor_comparison_results/`, not tracked by git). The
two `.tikz` files are pgfplots figures that read the `.dat` files next to them; `regressor_accuracy.tex` and
`regressor_budget.tex` are standalone wrappers (`pdflatex regressor_accuracy.tex` inside the output directory). Random
draws use NumPy's global generator and the generators of the gym spaces, seeded in a fixed order, so a run is
reproducible from its seeds up to floating-point rounding.

The results were first produced by exploratory scripts that this module consolidates. With the default settings and
the same version of torch it reproduces their numbers to rounding. Two groups of values are rounding error themselves
and change with the version of the linear-algebra library: the errors of ordinary least squares on the linear system
(between 1e-16 and 1e-10), and those of the shipped `ridge`, which inverts an ill-conditioned matrix explicitly, on
the linear system and at the smallest budgets.
"""

import contextlib
import io
import json
import os
import warnings

import gym
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from tap import Tap

torch.set_default_dtype(torch.float64)

import koopmanrl.environments  # noqa: F401,E402  (import registers the gym envs)
from koopmanrl.koopman_tensor.observables.numpy_observables import (  # noqa: E402
    monomials as numpy_monomials,
)
from koopmanrl.koopman_tensor.observables.torch_observables import (  # noqa: E402
    allMonomialPowers,
    monomials,
)
from koopmanrl.koopman_tensor.torch_tensor import KoopmanTensor, Regressor  # noqa: E402
from koopmanrl_utils.koopman_prediction_validation import (  # noqa: E402
    CONFIGS,
    dw_drift,
)

ENV_IDS = {cfg["label"]: env_id for env_id, cfg in CONFIGS.items()}
TITLES = {"Linear": "Linear", "Lorenz": "Lorenz", "FluidFlow": "Fluid flow", "DoubleWell": "Double well"}

SYSTEM_SEED = 0  # fixes the matrix A that the linear system draws at construction
HELDOUT_SEED = 10_000
HELDOUT_FRAC = 0.2
VALIDATION_FRAC = 0.2  # share of the identification transitions on which a hyperparameter is chosen

# key: (label, label in the figures, family, hyperparameter grid). Families: "paper" is the regressor behind the
# results of the paper, "shipped" the other regressors of the package as they were then, "alternative" those defined
# here.
REGRESSORS = {
    "ols": ("OLS, torch lstsq (paper)", "OLS (paper)", "paper", None),
    "ols_numpy": ("OLS, NumPy copy (normal equations)", "OLS, NumPy copy", "shipped", None),
    "ridge": ("ridge as shipped", "ridge (shipped)", "shipped", None),
    "sindy": ("SINDy as shipped", "SINDy (shipped)", "shipped", None),
    "rrr": ("reduced rank as shipped", "reduced rank (shipped)", "shipped", None),
    "ols_scaled": ("OLS, scaled columns", "OLS, scaled", "alternative", None),
    "tsvd": ("truncated SVD", "truncated SVD", "alternative", [1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4]),
    "ridge_cv": ("ridge, scaled increment", "ridge", "alternative", [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]),
    "stlsq_cv": ("thresholded LS, scaled increment", "thresholded LS", "alternative", [1e-4, 1e-3, 1e-2, 3e-2, 1e-1]),
    "lasso_cv": ("LASSO, scaled increment", "LASSO", "alternative", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
    "rrr_cv": ("reduced rank, scaled increment", "reduced rank", "alternative", [0.25, 0.5, 0.75, 0.9, 1.0]),
    "tls": ("total least squares", "total least squares", "alternative", None),
    "huber": ("Huber, scaled increment", "Huber", "alternative", None),
}
SHIPPED = ("ols", "ols_numpy", "ridge", "sindy", "rrr")

FLOOR = 1e-17  # values are floored here before they are written for a logarithmic axis


class ArgumentParser(Tap):
    study: str = "all"  # "accuracy", "budget" or "all"
    environments: list[str] = list(ENV_IDS)  # benchmarks, by the labels of koopman_prediction_validation.CONFIGS
    regressors: list[str] = list(REGRESSORS)  # regressors to run
    accuracy_seeds: int = 12  # seeds of the accuracy study: 0, 1, ...
    budget_seeds: int = 8  # seeds of the budget study: 0, 1, ...
    fractions: list[float] = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]  # fractions of the tuned budget
    output_dir: str = "koopman_regressor_comparison_results"  # where the results are written
    summarize_only: bool = False  # rewrite the tables and figures from the JSON files in output_dir


# --------------------------------------------------------------------------- #
# Dictionaries and the regression problem
# --------------------------------------------------------------------------- #
def powers(dim, order):
    """Exponents (dim, n_monomials) of the monomial dictionary, in the package's ordering."""
    return np.asarray(allMonomialPowers(dim, order)).astype(int)


def lift(x, order):
    """Monomial dictionary of samples x (n, dim) -> (n_monomials, n)."""
    return monomials(order)(torch.tensor(np.atleast_2d(x).T)).numpy()


def design(X, U, state_order, action_order):
    """Regressor matrix with rows kron(psi(u_n), phi(x_n)), in the package's column order, and phi(X)."""
    Phi, Psi = lift(X, state_order), lift(U, action_order)
    return np.einsum("zn,jn->zjn", Psi, Phi).reshape(-1, len(X)).T, Phi


def unfold(M, phi_dim, psi_dim):
    """The package's assembly of the tensor K (phi, phi, psi) from the regression matrix M."""
    return np.stack([M[i].reshape((phi_dim, psi_dim), order="F") for i in range(phi_dim)])


# --------------------------------------------------------------------------- #
# Shipped regressors, frozen
# --------------------------------------------------------------------------- #
# The package's NumPy least squares, ridge, SINDy and reduced-rank regressions as they were when the results of the
# paper were produced (up to commit b06d974), on the regressor matrix Z (n, m) and the targets T (n, p). They are
# copied here operation for operation, so that the comparison describes that code even after the package's
# implementations change. As in the classes, the matrices are transposes of arrays with one column per transition.
def _as_class(A):
    return torch.from_numpy(np.ascontiguousarray(A.T)).T


def shipped_ols_numpy(Z, T):
    return np.linalg.pinv(Z.T @ Z) @ Z.T @ T


def shipped_ridge(Z, T, penalty=0.05):
    Z, T = _as_class(Z), _as_class(T)
    return (torch.linalg.inv(Z.T @ Z + (penalty * torch.eye(Z.shape[1]))) @ Z.T @ T).numpy()


def shipped_sindy(Z, T, threshold=0.05, iterations=10):
    Z, T = _as_class(Z), _as_class(T)
    B = torch.linalg.lstsq(Z, T, rcond=None).solution
    for _ in range(iterations):
        small = torch.abs(B) < threshold
        B[small] = 0
        for j in range(T.shape[1]):
            big = small[:, j] == 0
            B[big, j] = torch.linalg.lstsq(Z[:, big], T[:, j].unsqueeze(0).T, rcond=None).solution[:, 0]
    return B.numpy()


def shipped_rrr(Z, T, rank=8):
    Z, T = _as_class(Z), _as_class(T)
    B = torch.linalg.lstsq(Z, T, rcond=None).solution
    _, _, V = torch.linalg.svd(T.T @ Z @ B)
    W = V[0:rank].T
    return (B @ W @ W.T).numpy()


SHIPPED_ESTIMATORS = {
    "ols_numpy": shipped_ols_numpy,
    "ridge": shipped_ridge,
    "sindy": shipped_sindy,
    "rrr": shipped_rrr,
}


# --------------------------------------------------------------------------- #
# Alternative regressors, on standardised regressors Z (n, m) and increments T (n, p)
# --------------------------------------------------------------------------- #
def gram(Z, T):
    return Z.T @ Z / len(Z), Z.T @ T / len(Z)


def solve_symmetric(G, C, ridge=0.0, tol=0.0):
    """(G + ridge I)^+ C through the eigendecomposition of G, dropping directions below tol * largest eigenvalue."""
    w, V = np.linalg.eigh(G)
    keep = w > max(tol * w[-1], 1e-300)
    inv = np.zeros_like(w)
    inv[keep] = 1.0 / (w[keep] + ridge)
    return (V * inv) @ (V.T @ C)


def least_squares(Z, T, hyper=None):
    return np.linalg.lstsq(Z, T, rcond=None)[0]


def truncated_svd(Z, T, hyper):
    """Principal-component regression; `hyper` is the relative eigenvalue cut-off of the Gram matrix."""
    return solve_symmetric(*gram(Z, T), tol=hyper)


def ridge(Z, T, hyper):
    return solve_symmetric(*gram(Z, T), ridge=hyper)


def thresholded_least_squares(Z, T, hyper, iterations=10):
    """Sequentially thresholded least squares (the SINDy regression) on the standardised coefficients."""
    G, C = gram(Z, T)
    B = solve_symmetric(G, C)
    for _ in range(iterations):
        small = np.abs(B) < hyper
        B[small] = 0.0
        for j in range(B.shape[1]):
            big = ~small[:, j]
            if big.any():
                B[big, j] = np.linalg.lstsq(G[np.ix_(big, big)], C[big, j], rcond=None)[0]
    return B


def lasso(Z, T, hyper):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = Lasso(alpha=hyper, fit_intercept=False, precompute=Z.T @ Z, max_iter=10000, tol=1e-8)
        model.fit(Z, T)
    return np.atleast_2d(model.coef_).T


def reduced_rank(Z, T, hyper):
    """Least squares projected on the leading directions of its fitted values; `hyper` is the fraction of the
    output dimension that is kept."""
    G, C = gram(Z, T)
    B = solve_symmetric(G, C)
    rank = max(1, int(round(hyper * T.shape[1])))
    w, V = np.linalg.eigh(B.T @ G @ B)
    Vr = V[:, np.argsort(w)[::-1][:rank]]
    return B @ Vr @ Vr.T


def total_least_squares(Z, T, hyper=None):
    """Errors in regressors and targets, by the eigendecomposition of [Z T]^T [Z T]."""
    live = (T**2).mean(axis=0) > 0  # an output that is identically zero carries no equation
    B = np.zeros((Z.shape[1], T.shape[1]))
    if not live.any():
        return B
    A = np.hstack([Z, T[:, live]])
    _, V = np.linalg.eigh(A.T @ A)
    V = V[:, ::-1]
    m = Z.shape[1]
    B[:, live] = -V[:m, m:] @ np.linalg.pinv(V[m:, m:])
    return B


def huber(Z, T, hyper=None, c=1.345, iterations=10):
    """Iteratively reweighted least squares with one Huber weight per transition, from the norm of its residual
    across outputs in units of a robust (median absolute deviation) scale."""
    B = least_squares(Z, T)
    for _ in range(iterations):
        R = T - Z @ B
        scale = 1.4826 * np.median(np.abs(R - np.median(R, axis=0)), axis=0)
        scale[scale == 0] = 1.0
        d = np.sqrt(((R / scale) ** 2).mean(axis=1))
        sw = np.sqrt(np.minimum(1.0, c / np.maximum(d, 1e-300)))[:, None]
        B = np.linalg.lstsq(Z * sw, T * sw, rcond=None)[0]
    return B


ESTIMATORS = {
    "ols_scaled": least_squares,
    "tsvd": truncated_svd,
    "ridge_cv": ridge,
    "stlsq_cv": thresholded_least_squares,
    "lasso_cv": lasso,
    "rrr_cv": reduced_rank,
    "tls": total_least_squares,
    "huber": huber,
}


def fit_increment(name, Z, T_inc):
    """Coefficients (n_regressors, n_outputs) of the regression of the increment, in the original units, and the
    hyperparameter selected on the last `VALIDATION_FRAC` of the rows."""
    sz = np.sqrt((Z**2).mean(axis=0))
    st = np.sqrt((T_inc**2).mean(axis=0))
    sz[sz == 0] = 1.0
    st[st == 0] = 1.0
    Zs, Ts = Z / sz, T_inc / st
    estimator, grid = ESTIMATORS[name], REGRESSORS[name][3]
    hyper = None
    if grid is not None:
        cut = int((1.0 - VALIDATION_FRAC) * len(Zs))
        errors = []
        for h in grid:
            B = estimator(Zs[:cut], Ts[:cut], h)
            errors.append(float(((Zs[cut:] @ B - Ts[cut:]) ** 2).mean()))
        hyper = grid[int(np.argmin(np.nan_to_num(np.array(errors), nan=np.inf)))]
    return estimator(Zs, Ts, hyper) * st[None, :] / sz[:, None], hyper


def fit_tensor(X, U, Y, state_order, action_order, regressor="ols"):
    """Identify the tensor K (phi, phi, psi) from transitions X, U, Y (n, dim); also returns the hyperparameter.

    Ordinary least squares runs through the package's torch class, as for the paper. The other shipped regressors are
    the frozen copies above, applied to the same regression. The alternatives give the regression matrix M of the
    increment. M is unfolded exactly as the package does."""
    if regressor == "ols":
        with contextlib.redirect_stdout(io.StringIO()):
            tensor = KoopmanTensor(
                torch.tensor(X.T),
                torch.tensor(Y.T),
                torch.tensor(U.T),
                phi=monomials(state_order),
                psi=monomials(action_order),
                regressor=Regressor(regressor),
            )
        return unfold(
            np.nan_to_num(tensor.M.numpy(), nan=0.0, posinf=0.0, neginf=0.0), tensor.phi_dim, tensor.psi_dim
        ), None
    Z, Phi = design(X, U, state_order, action_order)
    phi_dim, psi_dim = Phi.shape[0], Z.shape[1] // Phi.shape[0]
    if regressor == "ols_numpy":  # the NumPy class lifts with the NumPy dictionaries
        Phi, Psi = numpy_monomials(state_order)(X.T), numpy_monomials(action_order)(U.T)
        Z = np.einsum("zn,jn->zjn", Psi, Phi).reshape(-1, len(X)).T
        M, hyper = shipped_ols_numpy(Z, numpy_monomials(state_order)(Y.T).T).T, None
    elif regressor in SHIPPED_ESTIMATORS:
        M, hyper = SHIPPED_ESTIMATORS[regressor](Z, lift(Y, state_order).T).T, None
    else:
        B, hyper = fit_increment(regressor, Z, (lift(Y, state_order) - Phi).T)
        B[:phi_dim] += np.eye(phi_dim)  # back from the increment to phi(x')
        M = B.T
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    return unfold(M, phi_dim, psi_dim), hyper


# --------------------------------------------------------------------------- #
# Data and error
# --------------------------------------------------------------------------- #
_PATHS = {}


def random_agent_paths(env_id, seed, num_paths, steps):
    """Random-agent trajectories as in `generate_koopman_tensor`; arrays X, U, Y of shape (num_paths, steps, dim)."""
    key = (env_id, seed, num_paths, steps)
    if key not in _PATHS:
        np.random.seed(SYSTEM_SEED)
        with contextlib.redirect_stdout(io.StringIO()):
            env = gym.make(env_id)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.observation_space.seed(seed)
        env.action_space.seed(seed)
        X = np.zeros((num_paths, steps, env.observation_space.shape[0]))
        Y = np.zeros_like(X)
        U = np.zeros((num_paths, steps, env.action_space.shape[0]))
        for p in range(num_paths):
            state = env.reset()
            for t in range(steps):
                X[p, t] = state
                action = env.action_space.sample()
                U[p, t] = action
                state, _, _, _ = env.step(action)
                Y[p, t] = state
        _PATHS[key] = (X, U, Y)
    return _PATHS[key]


def transitions(paths, num_paths=None):
    """The first `num_paths` trajectories of (X, U, Y), flattened to (n, dim) each."""
    return tuple(a[:num_paths].reshape(-1, a.shape[-1]) for a in paths)


def conditional_mean_lifted(env_id, X, U, Y, state_order):
    """E[phi(x') | x, u] (n_monomials, n). Deterministic benchmarks: phi(x').

    Double well: exact for monomials of degree at most two under the environment's Euler-Maruyama step
    x' = m + S(x) xi sqrt(dt) with S(x) = [[0.7, x_1], [0, 0.5]], for which E[x'_i x'_j] = m_i m_j + (S S^T)_ij dt."""
    if not CONFIGS[env_id]["stochastic"]:
        return lift(Y, state_order)
    if state_order > 2:
        raise ValueError("the closed form holds for dictionaries of degree at most two")
    dt = gym.make(env_id).unwrapped.dt
    out = lift(X + dt * dw_drift(X.T, U.T).T, state_order)
    second_moment = {(2, 0): 0.49 + X[:, 0] ** 2, (1, 1): 0.5 * X[:, 0], (0, 2): 0.25 * np.ones(len(X))}
    for j, exponents in enumerate(powers(2, state_order).T):
        if tuple(exponents) in second_moment:
            out[j] += second_moment[tuple(exponents)] * dt
    return out


def one_step_error(K, X, U, target, state_order, action_order):
    """The normalised one-step error of the tensor and of the persistence predictor on transitions (X, U)."""
    Phi = lift(X, state_order)
    prediction = np.einsum("ijz,zn,jn->in", K, lift(U, action_order), Phi)
    scale = (target**2).mean(axis=1)
    rows = np.arange(1, target.shape[0])  # the first dictionary function is the constant
    error = np.sqrt((((prediction - target)[rows] ** 2) / scale[rows, None]).mean())
    persistence = np.sqrt((((Phi - target)[rows] ** 2) / scale[rows, None]).mean())
    return float(error), float(persistence)


def held_out(env_id):
    """Held-out transitions and their target: the same for every seed and regressor."""
    cfg = CONFIGS[env_id]
    paths = random_agent_paths(env_id, HELDOUT_SEED, max(1, round(HELDOUT_FRAC * cfg["num_paths"])), cfg["steps"])
    X, U, Y = transitions(paths)
    return X, U, conditional_mean_lifted(env_id, X, U, Y, cfg["so"])


# --------------------------------------------------------------------------- #
# The two studies
# --------------------------------------------------------------------------- #
def accuracy_study(args):
    results = {}
    for label in args.environments:
        env_id, cfg = ENV_IDS[label], CONFIGS[ENV_IDS[label]]
        Xh, Uh, target = held_out(env_id)
        results[label] = {}
        for regressor in args.regressors:
            rows = dict(relative_error=[], error=[], persistence=[], hyperparameter=[])
            for seed in range(args.accuracy_seeds):
                data = transitions(random_agent_paths(env_id, seed, cfg["num_paths"], cfg["steps"]))
                K, hyper = fit_tensor(*data, cfg["so"], cfg["ao"], regressor)
                error, persistence = one_step_error(K, Xh, Uh, target, cfg["so"], cfg["ao"])
                rows["relative_error"].append(error / persistence)
                rows["error"].append(error), rows["persistence"].append(persistence)
                rows["hyperparameter"].append(hyper)
            results[label][regressor] = rows
            print(f"accuracy {label:10s} {regressor:10s} median {np.median(rows['relative_error']):.4g}", flush=True)
    return dict(seeds=args.accuracy_seeds, results=results)


def budget_paths(num_paths, fractions):
    """Numbers of trajectories for the fractions of the budget: whole trajectories, at least two, no repeats."""
    return sorted({max(2, round(f * num_paths)) for f in fractions})


def budget_study(args):
    results = {}
    for label in args.environments:
        env_id, cfg = ENV_IDS[label], CONFIGS[ENV_IDS[label]]
        Xh, Uh, target = held_out(env_id)
        counts = budget_paths(cfg["num_paths"], args.fractions)
        results[label] = dict(transitions=[n * cfg["steps"] for n in counts], regressors={})
        for regressor in args.regressors:
            per_budget = []
            for n in counts:
                values = []
                for seed in range(args.budget_seeds):
                    paths = random_agent_paths(env_id, seed, cfg["num_paths"], cfg["steps"])
                    K, _ = fit_tensor(*transitions(paths, n), cfg["so"], cfg["ao"], regressor)
                    error, persistence = one_step_error(K, Xh, Uh, target, cfg["so"], cfg["ao"])
                    values.append(error / persistence)
                per_budget.append(values)
            results[label]["regressors"][regressor] = per_budget
            medians = [f"{median(v):.3g}" for v in per_budget]
            print(f"budget   {label:10s} {regressor:10s} medians {medians}", flush=True)
    return dict(seeds=args.budget_seeds, fractions=args.fractions, results=results)


def median(values):
    """Median across seeds, with a failed fit (not a number) counted as an infinite error."""
    return float(np.median(np.nan_to_num(np.asarray(values, dtype=float), nan=np.inf)))


# --------------------------------------------------------------------------- #
# Tables and figures
# --------------------------------------------------------------------------- #
def number(value):
    """A value for a pgfplots table on a logarithmic axis."""
    return f"{max(value, FLOOR):.6e}" if np.isfinite(value) else "nan"


def write_table(path, header, rows, separator=" "):
    with open(path, "w") as f:
        f.write(separator.join(header) + "\n")
        for row in rows:
            f.write(separator.join(number(v) if isinstance(v, float) else str(v) for v in row) + "\n")
    print(f"wrote {path}")


def accuracy_axis(label, medians):
    """Limits of one panel: (lower limit, value at which larger medians are drawn as arrows, upper limit)."""
    cap = 1.0 if label == "Linear" else 3.0
    low = max(min(medians), 1e-16) / (30.0 if label == "Linear" else 1.6)
    return low, cap, 1.3 * cap


def fixed_point_ticks(low, high, most=4):
    """At most `most` ticks from the 1-2-5 sequence inside (low, high], written without exponents."""
    ticks = [m * 10.0**k for k in range(-6, 3) for m in (1, 2, 5) if 1.05 * low <= m * 10.0**k <= high]
    if len(ticks) > most:
        ticks = sorted({ticks[0]} | {t for t in ticks if f"{t:e}".startswith("1.0")})
    return [f"{t:g}" for t in ticks]


TIKZ_COLOURS = r"""  \definecolor{krink}{HTML}{1A1A19}
  \definecolor{krblue}{HTML}{2A78D6}
  \definecolor{krorange}{HTML}{EB6834}
  \definecolor{kraqua}{HTML}{1BAF7A}
  \definecolor{kryellow}{HTML}{EDA100}
  \definecolor{krmagenta}{HTML}{E87BA4}
  \definecolor{krviolet}{HTML}{4A3AA7}
  \definecolor{krred}{HTML}{E34948}
  \definecolor{krgrid}{HTML}{DDDCD6}
  \definecolor{krmuted}{HTML}{6B6A63}"""

TIKZ_HEADER = r"""%% Written by koopmanrl_utils/koopman_regressor_comparison.py; reads the .dat files next to it.
%% Preamble: \usepackage{pgfplots} \usepgfplotslibrary{groupplots} \usetikzlibrary{calc} \pgfplotsset{compat=1.16}
%% When the data are kept elsewhere: \pgfplotsset{table/search path={<directory>}}"""

STANDALONE = r"""\documentclass[border=2pt]{standalone}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\usetikzlibrary{calc}
\pgfplotsset{compat=1.16}
\begin{document}
\input{%s}
\end{document}
"""

# family: (colour, mark of the median, description)
FAMILY_STYLE = {
    "paper": ("krink", "diamond*", "used for the paper"),
    "shipped": ("krorange", "square*", "shipped in the package, not used"),
    "alternative": ("krblue", "*", "alternatives"),
}


def scatter_classes(style):
    """A pgfplots `scatter/classes` argument with one class per family of regressors."""
    return ",\n          ".join(
        f"{family}={{{style(colour, mark)}}}" for family, (colour, mark, _) in FAMILY_STYLE.items()
    )


def write_accuracy(accuracy, output_dir):
    """Tables and the TikZ figure of the accuracy study: one row per regressor, one panel per benchmark."""
    results = accuracy["results"]
    labels = [label for label in ENV_IDS if label in results]
    regressors = [r for r in REGRESSORS if all(r in results[label] for label in labels)]
    write_table(
        os.path.join(output_dir, "accuracy_table.csv"),
        ["regressor"] + labels,
        [[r] + [median(results[label][r]["relative_error"]) for label in labels] for r in regressors],
        separator=",",
    )
    panels = []
    for label in labels:
        values = {r: np.asarray(results[label][r]["relative_error"], dtype=float) for r in regressors}
        medians = {r: median(values[r]) for r in regressors}
        low, cap, high = accuracy_axis(label, list(medians.values()))
        write_table(
            os.path.join(output_dir, f"accuracy_{label}.dat"),
            ["row", "regressor", "family", "median", "minimum", "maximum"],
            [
                [i + 1, r, REGRESSORS[r][2], medians[r], float(values[r].min()), float(values[r].max())]
                for i, r in enumerate(regressors)
            ],
        )
        write_table(
            os.path.join(output_dir, f"accuracy_seeds_{label}.dat"),
            ["row", "family", "value"],
            [[i + 1, REGRESSORS[r][2], float(v)] for i, r in enumerate(regressors) for v in values[r] if v <= cap],
        )
        offscale = [
            [i + 1, REGRESSORS[r][2], cap, f"{medians[r]:.3g}"] for i, r in enumerate(regressors) if medians[r] > cap
        ]
        if offscale:
            write_table(
                os.path.join(output_dir, f"accuracy_offscale_{label}.dat"),
                ["row", "family", "position", "median"],
                offscale,
            )
        panels.append(dict(label=label, low=low, cap=cap, high=high, offscale=bool(offscale)))

    n = len(regressors)
    seeds = scatter_classes(lambda colour, mark: f"mark=*, draw=none, fill={colour}")
    medians = scatter_classes(lambda colour, mark: f"mark={mark}, draw=white, fill={colour}, line width=0.3pt")
    arrows = scatter_classes(
        lambda colour, mark: f"mark=triangle*, mark options={{rotate=-90}}, draw=white, fill={colour}, line width=0.3pt"
    )
    ticklabels = ",".join("{" + REGRESSORS[r][1] + "}" for r in regressors)
    out = [TIKZ_HEADER, r"\begin{tikzpicture}", TIKZ_COLOURS]
    out.append(
        rf"""  \begin{{groupplot}}[
      group style={{group size={len(panels)} by 1, horizontal sep=0.45cm, y descriptions at=edge left}},
      scale only axis, width=2.2cm, height={0.4 * n:.1f}cm,
      xmode=log, y dir=reverse, ymin=0.4, ymax={n + 0.6},
      ytick={{1,...,{n}}}, yticklabels={{{ticklabels}}},
      tick label style={{font=\footnotesize}}, title style={{font=\footnotesize\bfseries, yshift=-0.6ex}},
      xtick align=outside, xtick pos=bottom, ytick style={{draw=none}},
      xmajorgrids, grid style={{krgrid, line width=0.4pt}}, axis line style={{krgrid}},
      max space between ticks=28pt, try min ticks=2,
    ]"""
    )
    for panel in panels:
        label = panel["label"]
        ticks = ""
        if panel["high"] / panel["low"] < 100.0:  # under two decades: ticks without exponents
            fixed = ",".join(fixed_point_ticks(panel["low"], panel["cap"]))
            ticks = f", xtick={{{fixed}}}, xticklabels={{{fixed}}}"
        out.append(
            rf"    \nextgroupplot[title={{{TITLES[label]}}}, xmin={panel['low']:.3e}, xmax={panel['high']:.3e}{ticks}]"
        )
        if panel["low"] < 1.0 < panel["high"]:
            out.append(
                rf"      \draw[krmuted, densely dashed, line width=0.5pt] (axis cs:1,0.4) -- (axis cs:1,{n + 0.6});"
            )
        out.append(
            rf"""      \addplot[scatter, only marks, mark size=0.9pt, opacity=0.35, scatter src=explicit symbolic,
        scatter/classes={{
          {seeds}}}]
        table[x=value, y=row, meta=family] {{accuracy_seeds_{label}.dat}};
      \addplot[scatter, only marks, mark size=2.4pt, scatter src=explicit symbolic,
        scatter/classes={{
          {medians}}}]
        table[x=median, y=row, meta=family] {{accuracy_{label}.dat}};"""
        )
        if panel["offscale"]:
            out.append(
                rf"""      \addplot[scatter, only marks, mark size=2.4pt, scatter src=explicit symbolic,
        scatter/classes={{
          {arrows}}}]
        table[x=position, y=row, meta=family] {{accuracy_offscale_{label}.dat}};
      \addplot[draw=none, mark=none, nodes near coords, point meta=explicit symbolic,
        every node near coord/.style={{anchor=east, xshift=-4pt, font=\scriptsize, text=krmuted, fill=white,
          inner sep=1pt}}]
        table[x=position, y=row, meta=median] {{accuracy_offscale_{label}.dat}};"""
            )
    out.append(r"  \end{groupplot}")
    out.append(
        rf"""  \coordinate (below) at ($(group c1r1.south west)!0.5!(group c{len(panels)}r1.south east)$);
  \node[anchor=north, font=\footnotesize] at ($(below)+(0,-0.6cm)$)
    {{held-out one-step error relative to persistence}};
  \matrix[anchor=north, ampersand replacement=\&, column sep=2pt] at ($(below)+(0,-1.05cm)$) {{"""
    )
    cells = []
    for colour, mark, text in FAMILY_STYLE.values():
        cells.append(
            rf"    \draw[draw=white, fill={colour}, line width=0.3pt] plot[only marks, mark={mark}, mark size=2.4pt]"
            rf" coordinates {{(0,0)}}; \& \node[anchor=west, inner sep=1pt, font=\footnotesize]"
            rf" {{{text}\hspace*{{1.2em}}}};"
        )
    out.append(" \\&\n".join(cells) + r" \\")
    out.append("  };")
    out.append(r"\end{tikzpicture}")
    write_text(os.path.join(output_dir, "regressor_accuracy.tikz"), "\n".join(out) + "\n")
    write_text(os.path.join(output_dir, "regressor_accuracy.tex"), STANDALONE % "regressor_accuracy.tikz")


# regressor: pgfplots style of its curve in the budget figure. The regressors that are not drawn coincide with
# ordinary least squares on the scale of the figure or leave its axes; `budget_<benchmark>.dat` has all of them.
BUDGET_STYLE = {
    "ols": "krink, line width=1.1pt, mark=diamond*, mark size=2.2pt",
    "ridge_cv": "krblue, line width=0.8pt, mark=*, mark size=1.7pt",
    "tsvd": "kraqua, line width=0.8pt, mark=triangle*, mark size=2.1pt",
    "stlsq_cv": "kryellow, line width=0.8pt, mark=square*, mark size=1.6pt",
    "lasso_cv": "krmagenta, line width=0.8pt, mark=triangle*, mark size=2.1pt, mark options={rotate=180}",
    "huber": "krviolet, line width=0.8pt, mark=pentagon*, mark size=1.9pt",
    "sindy": "krorange, line width=0.8pt, densely dashed, mark=x, mark size=2.4pt, mark options={solid}",
    "rrr": "krred, line width=0.8pt, densely dashed, mark=star, mark size=2.4pt, mark options={solid}",
}
BUDGET_LEGEND_COLUMNS = 4


def write_budget(budget, output_dir):
    """Tables and the TikZ figure of the budget study: one curve per regressor, one panel per benchmark."""
    results = budget["results"]
    labels = [label for label in ENV_IDS if label in results]
    regressors = [r for r in REGRESSORS if all(r in results[label]["regressors"] for label in labels)]
    drawn = [r for r in BUDGET_STYLE if r in regressors]
    panels = []
    for label in labels:
        medians = {r: [median(v) for v in results[label]["regressors"][r]] for r in regressors}
        write_table(
            os.path.join(output_dir, f"budget_{label}.dat"),
            ["transitions"] + regressors,
            [[n] + [medians[r][i] for r in regressors] for i, n in enumerate(results[label]["transitions"])],
        )
        if label == "Linear":
            limits = (1e-16, 10.0)
        else:  # around the errors that are drawn, and at most 30 times persistence
            shown = [v for r in drawn for v in medians[r] if np.isfinite(v)]
            limits = (min(shown) / 1.5, min(30.0, 1.6 * max(shown)))
        panels.append(dict(label=label, ymin=limits[0], ymax=limits[1]))

    out = [TIKZ_HEADER, r"\begin{tikzpicture}", TIKZ_COLOURS]
    out.append(
        rf"""  \begin{{groupplot}}[
      group style={{group size={len(panels)} by 1, horizontal sep=0.95cm}},
      scale only axis, width=2.15cm, height=3.3cm,
      xmode=log, ymode=log, unbounded coords=jump,
      tick label style={{font=\scriptsize}}, title style={{font=\footnotesize\bfseries, yshift=-0.6ex}},
      label style={{font=\footnotesize}},
      tick align=outside, tick pos=left, xtick pos=bottom,
      grid=major, grid style={{krgrid, line width=0.4pt}}, axis line style={{krmuted}},
      max space between ticks=30pt, try min ticks=3,
    ]"""
    )
    for i, panel in enumerate(panels):
        label = panel["label"]
        options = ", ylabel={held-out error rel.\\ to persistence}" if i == 0 else ""
        if panel["ymax"] / panel["ymin"] < 100.0:  # under two decades: ticks without exponents
            fixed = ",".join(fixed_point_ticks(panel["ymin"], panel["ymax"], most=5))
            options += f", ytick={{{fixed}}}, yticklabels={{{fixed}}}"
        limits = f"ymin={panel['ymin']:.3e}, ymax={panel['ymax']:.3e}"
        out.append(rf"    \nextgroupplot[title={{{TITLES[label]}}}, {limits}{options}]")
        out.append(
            r"      \draw[krmuted, densely dashed, line width=0.5pt] "
            r"({rel axis cs:0,0}|-{axis cs:1,1}) -- ({rel axis cs:1,0}|-{axis cs:1,1});"
        )
        for r in drawn:
            out.append(rf"      \addplot[{BUDGET_STYLE[r]}] table[x=transitions, y={r}] {{budget_{label}.dat}};")
    out.append(r"  \end{groupplot}")
    out.append(
        rf"""  \coordinate (below) at ($(group c1r1.south west)!0.5!(group c{len(panels)}r1.south east)$);
  \node[anchor=north, font=\footnotesize] at ($(below)+(0,-0.65cm)$) {{identification transitions}};
  \matrix[anchor=north, ampersand replacement=\&, column sep=2pt, row sep=2pt] at ($(below)+(0,-1.1cm)$) {{"""
    )
    rows = []
    for start in range(0, len(drawn), BUDGET_LEGEND_COLUMNS):
        cells = []
        for r in drawn[start : start + BUDGET_LEGEND_COLUMNS]:
            cells.append(
                rf"    \draw[{BUDGET_STYLE[r]}] (-0.3,0) -- (0.3,0); \draw[{BUDGET_STYLE[r]}] plot[only marks]"
                rf" coordinates {{(0,0)}}; \& \node[anchor=west, inner sep=1pt, font=\scriptsize]"
                rf" {{{REGRESSORS[r][1]}\quad}};"
            )
        rows.append(" \\&\n".join(cells) + r" \\")
    out.append("\n".join(rows))
    out.append("  };")
    out.append(r"\end{tikzpicture}")
    write_text(os.path.join(output_dir, "regressor_budget.tikz"), "\n".join(out) + "\n")
    write_text(os.path.join(output_dir, "regressor_budget.tex"), STANDALONE % "regressor_budget.tikz")


def write_text(path, text):
    with open(path, "w") as f:
        f.write(text)
    print(f"wrote {path}")


def summarize(output_dir):
    for name, writer in (("accuracy", write_accuracy), ("budget", write_budget)):
        path = os.path.join(output_dir, f"{name}.json")
        if os.path.exists(path):
            with open(path) as f:
                writer(json.load(f), output_dir)


def main():
    args = ArgumentParser().parse_args()
    for name in args.regressors:
        if name not in REGRESSORS:
            raise ValueError(f"unknown regressor {name!r}; choose from {list(REGRESSORS)}")
    for label in args.environments:
        if label not in ENV_IDS:
            raise ValueError(f"unknown environment {label!r}; choose from {list(ENV_IDS)}")
    if args.study not in ("accuracy", "budget", "all"):
        raise ValueError("--study is 'accuracy', 'budget' or 'all'")
    os.makedirs(args.output_dir, exist_ok=True)
    if not args.summarize_only:
        warnings.filterwarnings("ignore", category=UserWarning, module="gym")
        for name, study in (("accuracy", accuracy_study), ("budget", budget_study)):
            if args.study in (name, "all"):
                result = study(args)
                with open(os.path.join(args.output_dir, f"{name}.json"), "w") as f:
                    json.dump(result, f, indent=1)
    summarize(args.output_dir)


if __name__ == "__main__":
    main()
