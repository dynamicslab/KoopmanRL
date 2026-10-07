"""Prediction error of the fitted Koopman tensor on held-out random-agent data.

For each benchmark the Koopman tensor is identified from random-agent trajectories with the tuned SKVI dictionary
orders and identification budget of `configurations/`, on 80% of the trajectories, and evaluated on the remaining 20%.
The split is by whole trajectory, so no held-out transition comes from a trajectory used for the fit. Everything is
repeated over seeds and reported as the median (and quartiles) across seeds.

One-step error (`onestep_error.csv`, `onestep_by_degree*.csv`, `onestep_table.tex`)
    SKVI and SAKC use the tensor only through the one-step prediction K^u phi(x) of the dictionary; neither iterates
    it nor decodes it back to the state. The error is therefore measured in dictionary space, as the root mean square
    over transitions n and nonconstant dictionary functions j of

        (K^{u_n} phi(x_n) - E[phi(x'_n) | x_n, u_n])_j / s_j,

    where s_j is the root mean square of the target for function j, so that high-order monomials do not dominate. For
    the deterministic benchmarks the conditional mean is phi(x'_n). For the double well it is a Monte-Carlo average
    over the one-step noise, and the error of a single realisation phi(x'_n) about it is reported as the noise floor.
    Because the time step is small, x' is close to x, so the same error with phi(x_n) in place of the prediction (the
    persistence baseline) and the ratio of the two are reported too: the ratio is the fraction of the one-step change
    of the dictionary that the model misses. The by-degree files give the held-out error per monomial degree. The
    error of the decoded state, normalised by the root mean square of the target state, is kept in the unprefixed
    columns for reference.

Multi-step error (`multistep_<benchmark>.csv`)
    The error of the predicted state along fresh trajectories under the recorded actions, against the horizon, for
    two ways of rolling the model forward: iterating phi_{k+1} = K^{u_k} phi_k and decoding with B ("lifted"), and
    re-evaluating the dictionary at the predicted state after every step, x_{k+1} = B^T K^{u_k} phi(x_k)
    ("reproj"). For the double well the reference is a Monte-Carlo mean trajectory. This is a diagnostic of the model:
    the algorithms never roll it forward.

Training against held-out error (`train_vs_heldout_<benchmark>.csv`)
    The one-step error on the training and on the held-out transitions against the number of identification
    transitions, on nested subsamples (each smaller set is contained in the next). With the number of parameters
    fixed, the least-squares training error rises with the sample size towards the approximation floor while the
    held-out error falls towards it; overfitting would show as a gap that does not close.

Usage (from the repository root):

    uv run -m koopmanrl_utils.koopman_prediction_validation                      # four benchmarks, seeds 123-130
    uv run -m koopmanrl_utils.koopman_prediction_validation --seeds 2 --seed0 0  # two seeds starting at 0

The files are written into `--output_dir` (default `koopman_prediction_validation_results/`, not tracked by git).
Random draws use NumPy's global generator and a per-seed `RandomState`, seeded in a fixed order, so a run is
reproducible from its seed up to floating-point rounding (differences of order 1e-12 between runs, which reach the
leading digits only for the linear system, whose errors are at machine precision).

Reference: the results are in "Koopman-Assisted Reinforcement Learning" (Rozwood, Mehrez, Paehler, Sun and Brunton),
section "Evaluation" (one-step error), and in its electronic supplementary material, section "Koopman model
validation: additional results" (multi-step error, training against held-out error).
"""

import argparse
import os

import gym
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

import koopmanrl.environments  # noqa: F401,E402  (import registers the gym envs)
from koopmanrl.koopman_tensor.observables import torch_observables as obs  # noqa: E402
from koopmanrl.koopman_tensor.observables.torch_observables import (  # noqa: E402
    allMonomialPowers,
)
from koopmanrl.koopman_tensor.torch_tensor import KoopmanTensor, Regressor  # noqa: E402

# Tuned SKVI tensor settings, as in configurations/skvi_<benchmark>_hparams.json (num-paths, num-steps-per-path,
# state-order, action-order).
CONFIGS = {
    "LinearSystem-v0": dict(label="Linear", num_paths=75, steps=250, so=2, ao=3, stochastic=False),
    "Lorenz-v0": dict(label="Lorenz", num_paths=150, steps=250, so=3, ao=1, stochastic=False),
    "FluidFlow-v0": dict(label="FluidFlow", num_paths=200, steps=225, so=4, ao=2, stochastic=False),
    "DoubleWell-v0": dict(label="DoubleWell", num_paths=175, steps=100, so=2, ao=4, stochastic=True),
}

HORIZON = 500  # rollout horizon (steps)
N_ROLLOUT_PATHS = 40  # fresh held-out trajectories used for the rollout test
N_MC = 512  # Monte-Carlo samples for the stochastic mean-trajectory reference
N_ID_GRID = [250, 500, 1000, 2000, 4000, 8000, 12000]
HELDOUT_FRAC = 0.2
CLIP = 1e6  # guard against polynomial-model blow-up in long rollouts


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def make_env(env_id, seed):
    env = gym.make(env_id).unwrapped  # unwrap TimeLimit; we control the loop length
    env.action_space.seed(seed)
    return env


def collect_paths(env, num_paths, steps):
    """Random-agent rollouts; returns numpy arrays shaped (P, steps, dim)."""
    sd = env.observation_space.shape[0]
    ad = env.action_space.shape[0]
    X = np.zeros((num_paths, steps, sd))
    Y = np.zeros((num_paths, steps, sd))
    U = np.zeros((num_paths, steps, ad))
    for p in range(num_paths):
        s = np.asarray(env.reset(), dtype=float)
        for t in range(steps):
            a = np.asarray(env.action_space.sample(), dtype=float)
            X[p, t], U[p, t] = s, a
            s = np.asarray(env.step(a)[0], dtype=float)
            Y[p, t] = s
    return X, Y, U


def flat(A):
    """(P, steps, dim) -> (dim, P*steps)."""
    P, T, d = A.shape
    return A.reshape(P * T, d).T


# --------------------------------------------------------------------------- #
# DoubleWell SDE helpers (vectorised; checked against the env at start-up)
# --------------------------------------------------------------------------- #
def dw_drift(Xd, Ud):
    """Vectorised DoubleWell drift; Xd (2, N), Ud (1, N)."""
    return np.stack([4 * Xd[0] - 4 * Xd[0] ** 3 + Ud[0], -2 * Xd[1] + Ud[0]])


def dw_selfcheck(env):
    rng = np.random.RandomState(0)
    for _ in range(20):
        x, u = rng.uniform(-2, 2, 2), rng.uniform(-1, 1, 1)
        ref = np.asarray(env.continuous_f(u)(0.0, x), dtype=float).ravel()
        assert np.allclose(ref, dw_drift(x[:, None], u[:, None])[:, 0]), "dw_drift out of sync with env"


def dw_mc_mean_trajectory(x0, Useq, dt, rng):
    """Monte-Carlo E[x_h | x_0, u_0..u_{h-1}] by Euler-Maruyama (same scheme as the env).
    x0 (2, P), Useq (P, H, 1) -> (2, P, H)."""
    P, H, _ = Useq.shape
    Xc = np.repeat(x0[:, :, None], N_MC, axis=2)  # (2, P, M)
    out = np.empty((2, P, H))
    for h in range(H):
        u = Useq[:, h, 0][None, :, None]
        drift = np.stack([4 * Xc[0] - 4 * Xc[0] ** 3 + u[0], -2 * Xc[1] + u[0]])
        w = rng.normal(size=(2, P, N_MC))
        # sigma(x) = [[0.7, x], [0, 0.5]]
        diff = np.stack([0.7 * w[0] + Xc[0] * w[1], 0.5 * w[1]])
        Xc = np.clip(Xc + drift * dt + diff * np.sqrt(dt), -CLIP, CLIP)
        out[:, :, h] = Xc.mean(axis=2)
    return out


# --------------------------------------------------------------------------- #
# Model + metrics
# --------------------------------------------------------------------------- #
def rms(A):
    return float(np.sqrt(np.mean(np.sum(A**2, axis=0))))


def nrmse(pred, ref):
    return rms(ref - pred) / rms(ref)


def build_tensor(Xd, Yd, Ud, so, ao):
    return KoopmanTensor(
        torch.tensor(Xd),
        torch.tensor(Yd),
        torch.tensor(Ud),
        phi=obs.monomials(so),
        psi=obs.monomials(ao),
        regressor=Regressor("ols"),
    )


def predict(tensor, Xd, Ud):
    out = np.asarray(tensor.f(torch.tensor(np.ascontiguousarray(Xd)), torch.tensor(np.ascontiguousarray(Ud))))
    return np.clip(np.nan_to_num(out, nan=CLIP, posinf=CLIP, neginf=-CLIP), -CLIP, CLIP)


def phi_of(tensor, Xd):
    return np.asarray(tensor.phi(torch.tensor(np.ascontiguousarray(Xd))))


def predict_phi(tensor, Xd, Ud):
    """One-step prediction in dictionary space, K^u phi(x): exactly the quantity SKVI and SAKC consume."""
    out = np.asarray(tensor.phi_f(torch.tensor(np.ascontiguousarray(Xd)), torch.tensor(np.ascontiguousarray(Ud))))
    return np.clip(np.nan_to_num(out, nan=CLIP, posinf=CLIP, neginf=-CLIP), -CLIP, CLIP)


def ref_phi(tensor, env, Xd, Ud, Yd, stoch, rng, n_mc=128):
    """Reference E[phi(x') | x, u]. Deterministic: phi(x'). DoubleWell: Monte-Carlo over the
    one-step Gaussian increment (note E[phi(x')] != phi(E[x']) for nonlinear phi)."""
    if not stoch:
        return phi_of(tensor, Yd)
    mean = Xd + env.dt * dw_drift(Xd, Ud)
    acc = 0.0
    for _ in range(n_mc):
        w = rng.normal(size=Xd.shape)
        diff = np.stack([0.7 * w[0] + Xd[0] * w[1], 0.5 * w[1]]) * np.sqrt(env.dt)
        acc = acc + phi_of(tensor, mean + diff)
    return acc / n_mc


def lifted_metrics(pred, ref, base, realised=None):
    """Scaled dictionary-space errors. Each dictionary coordinate is divided by its RMS over the
    reference set so that high-order monomials do not dominate; constant coordinates are dropped."""
    keep = ref.std(axis=1) > 1e-12
    sc = np.sqrt(np.mean(ref[keep] ** 2, axis=1, keepdims=True))
    e = lambda A: float(np.sqrt(np.mean((A[keep] / sc) ** 2)))  # noqa: E731
    out = dict(err=e(ref - pred), persistence=e(ref - base), increment=e(ref - pred) / e(ref - base))
    out["noise_floor"] = e(ref - realised) if realised is not None else 0.0
    return out


def lifted_step(tensor, Phi, Ud):
    """phi_{k+1} = K^{u_k} phi_k for a batch; Phi (d, P), Ud (adim, P)."""
    K_u = np.asarray(tensor.K_(torch.tensor(np.ascontiguousarray(Ud))))
    if K_u.ndim == 2:
        K_u = K_u[None]
    out = np.einsum("pij,jp->ip", K_u, Phi)
    return np.clip(np.nan_to_num(out, nan=CLIP, posinf=CLIP, neginf=-CLIP), -CLIP, CLIP)


# --------------------------------------------------------------------------- #
# One seed of one environment
# --------------------------------------------------------------------------- #
def run_env(env_id, cfg, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    env = make_env(env_id, seed)
    stoch = cfg["stochastic"]
    if stoch:
        dw_selfcheck(env)

    X, Y, U = collect_paths(env, cfg["num_paths"], cfg["steps"])
    perm = rng.permutation(cfg["num_paths"])
    n_te = max(1, int(HELDOUT_FRAC * cfg["num_paths"]))
    te, tr = perm[:n_te], perm[n_te:]
    Xtr, Ytr, Utr = flat(X[tr]), flat(Y[tr]), flat(U[tr])
    Xte, Yte, Ute = flat(X[te]), flat(Y[te]), flat(U[te])

    ref_tr = Xtr + env.dt * dw_drift(Xtr, Utr) if stoch else Ytr
    ref_te = Xte + env.dt * dw_drift(Xte, Ute) if stoch else Yte

    tensor = build_tensor(Xtr, Ytr, Utr, cfg["so"], cfg["ao"])

    # one-step error of the decoded state
    pred_te = predict(tensor, Xte, Ute)
    onestep = dict(
        train=nrmse(predict(tensor, Xtr, Utr), ref_tr),
        heldout=nrmse(pred_te, ref_te),
        persistence=nrmse(Xte, ref_te),
        increment=rms(ref_te - pred_te) / rms(ref_te - Xte),
        noise_floor=nrmse(Yte, ref_te) if stoch else 0.0,
    )

    rphi_tr = ref_phi(tensor, env, Xtr, Utr, Ytr, stoch, rng)
    rphi_te = ref_phi(tensor, env, Xte, Ute, Yte, stoch, rng)
    L = lifted_metrics(
        predict_phi(tensor, Xte, Ute), rphi_te, phi_of(tensor, Xte), phi_of(tensor, Yte) if stoch else None
    )
    Ltr = lifted_metrics(predict_phi(tensor, Xtr, Utr), rphi_tr, phi_of(tensor, Xtr))
    onestep.update(
        phi_train=Ltr["err"],
        phi_heldout=L["err"],
        phi_persistence=L["persistence"],
        phi_increment=L["increment"],
        phi_noise_floor=L["noise_floor"],
    )

    # held-out dictionary-space error by monomial degree
    deg = np.asarray(allMonomialPowers(X.shape[2], cfg["so"])).sum(axis=0)
    pphi_te, bphi_te = predict_phi(tensor, Xte, Ute), phi_of(tensor, Xte)
    bydeg = np.full((4, 2), np.nan)
    for d in range(1, cfg["so"] + 1):
        k = deg == d
        m = lifted_metrics(pphi_te[k], rphi_te[k], bphi_te[k])
        bydeg[d - 1] = [m["err"], m["increment"]]
    onestep["bydeg"] = bydeg

    # multi-step error along fresh trajectories
    Xr, Yr, Ur = collect_paths(env, N_ROLLOUT_PATHS, HORIZON)  # fresh held-out trajectories
    x0 = Xr[:, 0, :].T  # (dim, P)
    if stoch:
        ref_traj = dw_mc_mean_trajectory(x0, Ur, env.dt, rng)  # (dim, P, H)
    else:
        ref_traj = np.transpose(Yr, (2, 0, 1))
    Xcur = x0.copy()
    Phi = np.asarray(tensor.phi(torch.tensor(np.ascontiguousarray(x0))))
    B = np.asarray(tensor.B)
    multistep = np.empty((HORIZON, 3))  # reprojected, lifted, persistence
    for h in range(HORIZON):
        Uk = Ur[:, h, :].T
        Xcur = predict(tensor, Xcur, Uk)
        Phi = lifted_step(tensor, Phi, Uk)
        ref_h = ref_traj[:, :, h]
        multistep[h] = [nrmse(Xcur, ref_h), nrmse(np.clip(B.T @ Phi, -CLIP, CLIP), ref_h), nrmse(x0, ref_h)]

    # training against held-out error on nested subsamples
    order = rng.permutation(Xtr.shape[1])
    sweep = np.full((len(N_ID_GRID), 2), np.nan)
    for j, n_id in enumerate(N_ID_GRID):
        if n_id > Xtr.shape[1]:
            continue
        idx = order[:n_id]
        t = build_tensor(Xtr[:, idx], Ytr[:, idx], Utr[:, idx], cfg["so"], cfg["ao"])
        a = lifted_metrics(predict_phi(t, Xtr[:, idx], Utr[:, idx]), rphi_tr[:, idx], phi_of(t, Xtr[:, idx]))
        b = lifted_metrics(predict_phi(t, Xte, Ute), rphi_te, phi_of(t, Xte))
        sweep[j] = [a["err"], b["err"]]

    return onestep, multistep, sweep, getattr(env, "dt", 1.0)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def q(a, axis=0):
    """median, lower quartile, upper quartile across seeds."""
    return np.nanmedian(a, axis), np.nanpercentile(a, 25, axis), np.nanpercentile(a, 75, axis)


def write_csv(path, header, rows):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(f"{v:.6e}" if isinstance(v, float) else str(v) for v in r) + "\n")
    print(f"wrote {path}")


def tex_num(v):
    if v == 0:
        return "---"
    m, e = f"{v:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="koopman_prediction_validation_results")
    ap.add_argument("--seeds", type=int, default=8, help="number of seeds")
    ap.add_argument("--seed0", type=int, default=123, help="first seed; seeds are seed0, seed0 + 1, ...")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    one_rows, tex_rows, deg_rows = [], [], []
    for env_id, cfg in CONFIGS.items():
        label = cfg["label"]
        print(f"\n=== {env_id} (orders s{cfg['so']}/a{cfg['ao']}, {cfg['num_paths']}x{cfg['steps']}) ===")
        ones, multis, sweeps = [], [], []
        for k in range(args.seeds):
            o, m, s, dt = run_env(env_id, cfg, args.seed0 + k)
            ones.append(o), multis.append(m), sweeps.append(s)
            print(f"  seed {args.seed0 + k}: held-out {o['heldout']:.3e}  increment-norm. {o['increment']:.3e}")

        keys = [
            "phi_train",
            "phi_heldout",
            "phi_persistence",
            "phi_increment",
            "phi_noise_floor",
            "train",
            "heldout",
            "persistence",
            "increment",
            "noise_floor",
        ]
        med = {k_: float(np.median([o[k_] for o in ones])) for k_ in keys}
        one_rows.append([label] + [med[k_] for k_ in keys])
        tex_rows.append(f"{label} & " + " & ".join(tex_num(med[k_]) for k_ in keys[:5]) + r" \\")

        D = np.nanmedian(np.stack([o["bydeg"] for o in ones]), axis=0)
        deg_rows.append([label] + [float(D[d, 0]) for d in range(4)] + [float(D[d, 1]) for d in range(4)])

        M = np.stack(multis)  # (seeds, H, 3)
        rows = []
        for h in range(HORIZON):
            r = [h + 1, float((h + 1) * dt)]
            for c in range(2):
                a, lo, hi = q(M[:, h, c])
                r += [float(a), float(lo), float(hi)]
            r.append(float(np.median(M[:, h, 2])))
            rows.append(r)
        write_csv(
            os.path.join(args.output_dir, f"multistep_{label}.csv"),
            ["h", "t", "reproj", "reproj_lo", "reproj_hi", "lifted", "lifted_lo", "lifted_hi", "persist"],
            rows,
        )

        S = np.stack(sweeps)  # (seeds, grid, 2)
        rows = []
        for j, n_id in enumerate(N_ID_GRID):
            if np.all(np.isnan(S[:, j, 0])):
                continue
            (a, alo, ahi), (b, blo, bhi) = q(S[:, j, 0]), q(S[:, j, 1])
            rows.append([n_id, float(a), float(alo), float(ahi), float(b), float(blo), float(bhi)])
        write_csv(
            os.path.join(args.output_dir, f"train_vs_heldout_{label}.csv"),
            ["n_id", "train", "train_lo", "train_hi", "heldout", "heldout_lo", "heldout_hi"],
            rows,
        )

    write_csv(
        os.path.join(args.output_dir, "onestep_error.csv"),
        [
            "env",
            "phi_train",
            "phi_heldout",
            "phi_persistence",
            "phi_increment",
            "phi_noise_floor",
            "train",
            "heldout",
            "persistence",
            "increment",
            "noise_floor",
        ],
        one_rows,
    )
    write_csv(
        os.path.join(args.output_dir, "onestep_by_degree.csv"),
        ["env"] + [f"err_deg{d}" for d in range(1, 5)] + [f"inc_deg{d}" for d in range(1, 5)],
        deg_rows,
    )
    # same data, one row per degree (pgfplots-friendly)
    write_csv(
        os.path.join(args.output_dir, "onestep_by_degree_long.csv"),
        ["degree"] + [r[0] for r in deg_rows],
        [[d + 1] + [r[1 + d] for r in deg_rows] for d in range(4)],
    )
    with open(os.path.join(args.output_dir, "onestep_table.tex"), "w") as f:
        f.write("\n".join(tex_rows) + "\n\\bottomrule\n")  # \bottomrule must live here: \input before it breaks tabular
    print(f"wrote {os.path.join(args.output_dir, 'onestep_table.tex')}")


if __name__ == "__main__":
    main()
