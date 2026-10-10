"""t-SNE embedding of Koopman tensors of the four benchmarks, written in a common dictionary basis.

For every benchmark a set of Koopman tensors is identified from random-agent data by ordinary least squares, the
regression of the package. Every tensor is written in the monomial dictionaries of the largest state order and the
largest action order of the set (the common basis), flattened, and all tensors of all benchmarks are embedded
together in two dimensions by t-distributed stochastic neighbour embedding. The output is the scatter of the
electronic supplementary material of the paper (Figure S2), as files that its TikZ source reads.

Provenance
    The code that produced the published figure is not in the history of this repository or of the upstream
    repository: both only hold the earlier version of this script, which embedded the rows of the regression matrix
    of one benchmark. The revised manuscript says, in its electronic supplementary material, that the tensors "were
    identified for each benchmark with state and action dictionaries of several orders, projected onto a common
    dictionary basis and embedded in two dimensions by t-distributed stochastic neighbour embedding". This module is
    a reconstruction, with two sweeps.

Sweeps (`--sweep`)
    `inferred_layout` (default), 161 tensors per benchmark from one data seed
        121 tensors of state order 2 and action order 2 on a grid of 11 numbers of trajectories by 11 numbers of steps
        per trajectory, 25 on a second such grid of 5 by 5, and 15 on the grid of state orders 1 to 4 by action
        orders 1 to 4 without (4, 4). This is the layout that the order of the published coordinates points to
        (161 points per benchmark: 11 blocks of 11, 5 blocks of 5, then groups of 4, 4, 4 and 3; within the blocks of
        11 the points of the Lorenz system and of the fluid flow are ordered like the names 50, 100, ..., 550 sorted
        as text). It is an inference, not recovered code. The published coordinates do not tell what the second
        variable of the two grids is, which seeds were used, or which of the 16 combinations of orders is absent;
        `INFERRED_LAYOUT` holds the assumptions (numbers of trajectories as the second variable, the last
        trajectories of the data for the second grid, (4, 4) absent). The numbers of trajectories are 10 to 110 where
        the published order suggests values five times larger, to keep the run short; with 50 to 550 the embedding
        has the same structure. The fluid flow of the paper has 157 points: four tensors of one block are absent.
    `orders`, the reading of the text of the paper
        The grid `--state_orders` by `--action_orders` (default 1 to 4 each) by `--seeds` data seeds (default 8), at
        the tuned SKVI identification budget of each benchmark or on the grid `--num_paths` by
        `--num_steps_per_path`. The budgets of a sweep are cut from one data set per seed (the first steps of the
        first trajectories); with one budget this is the data set of `generate_koopman_tensor`.

Common basis
    `largest` (default): every coefficient K[i, j, z] is placed at the positions of its three monomials in the
    dictionaries of the largest orders, and the other coefficients are zero. The tensor then gives the same
    prediction K^u phi(x) for its own dictionary functions through the common basis (checked by the tests). The
    double well has two state coordinates and the other benchmarks three: its coordinates are taken as the first two
    (`--double_well_coordinates`) of a state whose third coordinate does not appear, so all its coefficients of
    monomials with the third coordinate are zero. `smallest`: only the block of the smallest orders is kept.

Choices the paper leaves open, and their defaults
    The tensors are embedded as identified (`--units raw`, `--scaling none`, Euclidean `--metric`), with the defaults
    of scikit-learn for the t-SNE (`--perplexity 30`, PCA initialisation, Barnes-Hut) and `--tsne_seed 42`, the
    value of the earlier version of this script: the simplest reading of the text, and the one that gives the
    structure of the published figure. The alternatives test what the separation rests on: `--units box` (every
    state coordinate and the action divided by the largest absolute value of its bound in the observation and action
    boxes of the environment, which removes the scales of the benchmarks), `--scaling` (`standard`: z-score of every
    coefficient; `unit_norm`: every tensor divided by its norm), `--metric cosine`, `--subtract_persistence`
    (removes the coefficients K[i, i, 0] = 1 of "no change"), `--shared_coordinates_only` (drops the monomials of
    the third coordinate, which the double well does not have) and `--linear_system_seed` (the linear system draws
    its matrix A at construction: one fixed system by default; if negative, a new one per data seed, with the data
    of `generate_koopman_tensor`).

What the embedding shows
    `separation.csv` and `settings.json` give, per benchmark, the silhouette coefficient and the share of the ten
    nearest neighbours of a tensor that are of its benchmark ("purity"; see `separation`), in the embedding and among
    the flattened tensors. With the defaults the four benchmarks form four clusters in the embedding (on one numerical
    thread and on two: silhouette 0.63 to 0.82 and purity 0.95 to 0.98 per benchmark; published coordinates: 0.62 to
    0.69 and 0.95 to 0.96), and between 4 and 11 tensors per benchmark lie away from their cluster, as in the
    published figure. These
    are the tensors of the grid of orders: those of state order 1 of all four benchmarks lie at one common place,
    those of state orders 3 and 4 next to other clusters. The clusters are the tensors of orders (2, 2), which
    differ in the amount of data only. For the linear system, the fluid flow and the double well these are nearly
    one tensor (root-mean-square distance from their mean 0.0000, 0.02 and 0.03, against distances of 2 to 25
    between the benchmarks). For the Lorenz system they are not: its tensor depends strongly on the amount of data
    (root-mean-square distance from the mean 18.9, against a norm of the mean of 24.9). Among the flattened tensors
    the silhouettes of the linear system and of the Lorenz system are negative (-0.92 and -0.94) although their
    purity is 0.96 and 0.95, because a few tensors of state orders 3 and 4 lie far from all others. In the `orders`
    sweep the tensors form one cluster per benchmark and state order instead of four clusters (silhouette about
    0.1), although the nearest tensors of other orders are mostly of the same benchmark (about 0.9). The separation
    does not measure predictive accuracy, and part of it does not need dynamics: tensors of "no change" separate the
    double well from the other benchmarks because of its state dimension alone, and in the units of the environments
    most of the distance from the Lorenz system to the other benchmarks is in a few large coefficients.
    `koopmanrl_utils/TSNE.md` has the numbers.

Output (`--output_dir`, default `tsne_koopman_tensor_results/`, not tracked by git)
    `<benchmark>_tsne.csv` for linear_system, fluid_flow, lorenz and double_well: one row per tensor with the columns
    index, x-val, y-val (the three columns of the files of the paper), state_order, action_order, num_paths,
    num_steps_per_path, seed and grid (the number of the grid of the sweep). `tsne.csv`: all rows, with the benchmark
    and its class number in front. `t_sne_figure.tex`: the standalone pgfplots figure of the paper, reading the
    files next to it. `tsne_preview.pdf` and `.png`: the scatter with matplotlib, and the same points shaded by state
    order. `separation.csv`, `settings.json` (all arguments, numbers of numerical threads, run time, peak memory,
    scores) and `tensors_<benchmark>.npz` (the identified tensors, their rows and what they depend on besides: the
    size of the data set and the driver of the solver).

Usage (from the repository root):

    uv run -m koopmanrl_utils.tsne_koopman_tensor                    # inferred layout, 1.5 minutes on one core
    uv run -m koopmanrl_utils.tsne_koopman_tensor --sweep orders --output_dir tsne_koopman_tensor_results/orders
                                                                     # orders 1-4 x 1-4, 8 seeds, 5 minutes
    uv run -m koopmanrl_utils.tsne_koopman_tensor --embed_only --units box   # embed the stored tensors again
    uv run -m koopmanrl_utils.tsne_koopman_tensor --resume           # pick up an interrupted run
    uv run -m koopmanrl_utils.tsne_koopman_tensor --lstsq_driver gelsy --output_dir tsne_koopman_tensor_results/gelsy
                                                                     # the solver call of the package
    OMP_NUM_THREADS=1 uv run -m koopmanrl_utils.tsne_koopman_tensor --embed_only   # on one numerical thread
    uv run -m koopmanrl_utils.tsne_koopman_tensor --identify_only --environments lorenz
                                                                     # store the tensors of one benchmark, no embedding

The benchmarks are identified one after the other and stored as soon as they are done; only the data of one benchmark
are in memory (peak memory 0.8 GB for the default sweep and 0.9 GB for the `orders` sweep, most of it the
libraries). The times were measured on a shared two-core machine: 1 min 23 s for the default sweep pinned to one
core, 5 min for the `orders` sweep. With `--identify_only` the run ends when the tensors of `--environments` are
stored: nothing is embedded and no other file is written. One such call per benchmark and one call with
`--embed_only` are the two stages of the Snakemake workflow (`workflow/README.md`).

Reproducibility
    The data are drawn from NumPy's global generator and the generators of the gym spaces, seeded in a fixed order,
    and are the same in every run. What the regression returns from them depends on the LAPACK driver of
    `torch.linalg.lstsq` (`--lstsq_driver`), which is stored with the tensors and written into `settings.json`.

    `gelsd` (the default) and `gelss` solve the regression by a singular value decomposition and return the same
    tensor in every call. On one machine and with one number of numerical threads a run is then reproduced exactly,
    the tensors bit for bit and the tables byte for byte. With `gelsd` this was found for two runs of the default
    sweep on one thread, for a run with one process per benchmark (`--environments`) followed by `--embed_only`, for
    two runs on two threads (644 of 644 tensors each time) and for two runs of the `orders` sweep on one thread (512
    of 512); with `gelss` for two runs of the default sweep on one thread. The number of threads is one of the
    conditions: between one and two threads the tensors of the default sweep differed by a relative 2e-13 in the
    median and by up to 4e-6 (8e-6 in the `orders` sweep; both for the Lorenz system at orders (2, 4)). Another
    machine or another build of torch and its LAPACK was not tried and has to be expected to differ in the same way.

    `gelsy` is the driver of the solver call of `koopmanrl.koopman_tensor`, which names none; SKVI and SAKC solve
    their regressions with `gelsd`. Its result is not the same in every call, also within one process on one thread
    and with identical data: torch hands LAPACK a pivot array that it has not initialised (torch 2.9.1, and the
    sources of its releases 1.9.0 to 2.13.0), LAPACK keeps every column whose entry in this array is not zero out of
    the column pivoting, and the result follows what the memory held. Between runs of the default sweep the tensors
    of well-conditioned regressions then differ in the last digits (relative differences up to 1.2e-12).

    The solver estimates an effective rank with a threshold of the machine precision times the larger dimension of
    the regression matrix. In the default sweep the Lorenz system at orders (3, 4) and (4, 3) has singular values
    below this threshold (condition numbers 7e12 and 6e12); in the `orders` sweep the linear system at (4, 4), the
    Lorenz system at (3, 4), (4, 3) and (4, 4), and on some seeds the Lorenz system at (4, 2). For these `gelsd`
    returns the solution of smallest norm at the effective rank (in the default sweep rank 95 of 100 and 135 of 140,
    norms 44.7 and 2,058), on one thread and on two. With `gelsy` the estimate and the solution change with the
    pivot array: in 12 identifications on one thread the first of the two tensors had norm 506 six times, 81.7 four
    times, 1.47 and 0.41 once each, and the second 76,319 eleven times and 18,709 once. All other tensors of the
    default sweep agree between `gelsd` and `gelsy` to a relative 6e-6 (median 2e-13).

    The t-SNE is deterministic for identical tensors and an identical number of numerical threads, and sensitive to
    anything else. With `gelsd`, single points moved by up to 11% of the diagonal of the embedding (median 1.3%)
    between one and two threads from the same stored tensors, and by up to 14% (median 4%) between a run on one and
    a run on two threads. With `gelsy` they moved by up to 22% (median 3%) between two runs on one thread whose
    tensors differed in the last digits only, and by up to 36% (median 8%) when the two rank-deficient regressions
    came out differently. The clusters stayed the same in all cases (silhouette of all tensors 0.72 to 0.73, purity
    0.96 to 0.97). `settings.json` records the numbers of threads of a run; the number is fixed with
    `OMP_NUM_THREADS`. `--embed_only` embeds stored tensors again and refuses those that are not the tensors of the
    sweep and the driver of its arguments. The published coordinates were not made by this code and are not
    reproduced by it; `koopmanrl_utils/TSNE.md` has the comparisons.

Reference: "Koopman-Assisted Reinforcement Learning" (Rozwood, Mehrez, Paehler, Sun and Brunton), electronic
supplementary material, figure "t-distributed stochastic neighbour embedding of the Koopman tensors projected onto a
common basis".
"""

import contextlib
import io
import itertools
import json
import os
import resource
import sys
import time

import gym
import matplotlib
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors
from tap import Tap
from threadpoolctl import threadpool_info

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

torch.set_default_dtype(torch.float64)

import koopmanrl.environments  # noqa: F401,E402  (import registers the gym envs)
from koopmanrl.koopman_tensor.observables.torch_observables import (  # noqa: E402
    allMonomialPowers,
    monomials,
)
from koopmanrl_utils.koopman_prediction_validation import CONFIGS  # noqa: E402

# Benchmarks in the order of the legend of the figure. `name` is the stem of the per-benchmark file that the TikZ
# source of the paper reads (`<name>_tsne.csv`); `colour`, `rgb` and `mark` are the colour and the mark of that source
# and `marker` is the matplotlib marker of the preview. `scale` holds the units of `--units box`: for every state
# coordinate and for the action the largest absolute value of its bounds in the observation box and in the action box
# of the environment, so that the boxes lie within [-1, 1] in these units (the tests compare them with the
# environments). The boxes of the Lorenz system and of the fluid flow are not centred at zero in their third
# coordinate, [0, 50] and [0, 1].
BENCHMARKS = {
    "LinearSystem-v0": dict(
        name="linear_system",
        title="Linear System",
        colour="linear_sys_color",
        rgb=(251, 86, 7),
        mark="diamond*",
        marker="D",
        scale=((25.0, 25.0, 25.0), 10.0),
    ),
    "FluidFlow-v0": dict(
        name="fluid_flow",
        title="Fluid Flow",
        colour="fluid_flow_color",
        rgb=(255, 0, 110),
        mark="*",
        marker="o",
        scale=((1.0, 1.0, 1.0), 10.0),
    ),
    "Lorenz-v0": dict(
        name="lorenz",
        title="Lorenz 1963",
        colour="lorenz_color",
        rgb=(131, 56, 236),
        mark="square*",
        marker="s",
        scale=((20.0, 50.0, 50.0), 75.0),
    ),
    "DoubleWell-v0": dict(
        name="double_well",
        title="Stochastic Double Well",
        colour="double_well_color",
        rgb=(58, 134, 255),
        mark="triangle*",
        marker="^",
        scale=((2.0, 2.0), 25.0),
    ),
}
NAMES = {entry["name"]: env_id for env_id, entry in BENCHMARKS.items()}
COMMON_STATE_DIM = 3  # the state dimension of the common basis: the largest of the benchmarks
COLUMNS = ["index", "x-val", "y-val", "state_order", "action_order", "num_paths", "num_steps_per_path", "seed", "grid"]

# The sweep "inferred_layout" (see the module docstring): two grids of identification budgets at the default dictionary
# orders of `generate_tensor`, (2, 2), and one grid of dictionary orders at its default budget of 100 trajectories of
# 300 steps. Each entry is (state orders, action orders, numbers of paths, steps per path, whether the paths are taken
# from the end of the data set). The budgets of the second grid are budgets of the first one; it takes the last
# instead of the first trajectories, so that its tensors are not copies of tensors of the first grid (the published
# coordinates hold no such copies). The published files hold 15 tensors of the last grid, in groups of 4, 4, 4 and 3;
# the combination left out here is the last one, (4, 4).
INFERRED_LAYOUT = [
    ([2], [2], list(range(10, 120, 10)), list(range(50, 600, 50)), False),
    ([2], [2], list(range(20, 120, 20)), list(range(100, 600, 100)), True),
    ([1, 2, 3, 4], [1, 2, 3, 4], [100], [300], False),
]
INFERRED_LAYOUT_OMITTED_ORDERS = (4, 4)
INFERRED_LAYOUT_SEED = 123  # the default seed of `generate_tensor`

# LAPACK drivers of `torch.linalg.lstsq` for the regression (see "Reproducibility" in the module docstring). The
# package calls the solver without naming one, which is "gelsy" on the CPU.
LSTSQ_DRIVERS = ("gelsd", "gelss", "gelsy")
LSTSQ_DRIVER = "gelsd"


class ArgumentParser(Tap):
    environments: list[str] = list(NAMES)  # benchmarks: linear_system, fluid_flow, lorenz, double_well
    sweep: str = "inferred_layout"  # "inferred_layout" or "orders" (the grid of the next six arguments)
    state_orders: list[int] = [1, 2, 3, 4]  # orders of the state dictionaries ("orders" sweep)
    action_orders: list[int] = [1, 2, 3, 4]  # orders of the action dictionaries ("orders" sweep)
    num_paths: list[int] = []  # numbers of trajectories ("orders" sweep; default: the tuned SKVI number)
    num_steps_per_path: list[int] = []  # steps per trajectory ("orders" sweep; default: the tuned SKVI number)
    seeds: int = 8  # number of data seeds per configuration ("orders" sweep)
    seed0: int = 0  # first data seed; the seeds are seed0, seed0 + 1, ... ("orders" sweep)
    linear_system_seed: int = 0  # seed of the matrix A of the linear system; negative: drawn from each data seed
    lstsq_driver: str = LSTSQ_DRIVER  # LAPACK driver of the regression: "gelsd", "gelss" or "gelsy" (the package's)
    common_basis: str = "largest"  # "largest": the dictionaries of the largest orders; "smallest": of the smallest
    double_well_coordinates: list[int] = [0, 1]  # coordinates of the common state that the double well occupies
    units: str = "raw"  # "raw": the units of the environments; "box": states and actions in units of their boxes
    subtract_persistence: bool = False  # embed K minus the tensor of "no change" (K[i, i, 0] = 1)
    shared_coordinates_only: bool = False  # keep only the state monomials in the coordinates every benchmark has
    scaling: str = "none"  # "none", "standard" (z-score of every coefficient) or "unit_norm" (per tensor)
    metric: str = "euclidean"  # distance between the flattened tensors: "euclidean" or "cosine"
    perplexity: float = 30.0  # t-SNE perplexity (the default of scikit-learn)
    tsne_seed: int = 42  # random_state of the t-SNE (the value of the earlier version of this script)
    tsne_init: str = "pca"  # "pca" (the default of scikit-learn) or "random"
    tsne_method: str = "barnes_hut"  # "barnes_hut" (the default of scikit-learn) or "exact"
    neighbours: int = 10  # number of nearest neighbours of the purity scores
    output_dir: str = "tsne_koopman_tensor_results"  # where the results are written
    resume: bool = False  # reuse the tensors of the benchmarks that an interrupted run of this sweep has stored
    embed_only: bool = False  # embed the tensors stored in output_dir again instead of identifying them
    identify_only: bool = False  # identify and store the tensors of --environments, without embedding them


# --------------------------------------------------------------------------- #
# The common basis
# --------------------------------------------------------------------------- #
def powers(dim, order):
    """Exponents (dim, n_monomials) of the monomial dictionary, in the package's ordering."""
    return np.asarray(allMonomialPowers(dim, order)).astype(int)


def state_dimension(K, state_order):
    """The number of state variables of a tensor K (phi, phi, psi) of the monomials of `state_order`."""
    for dim in range(1, COMMON_STATE_DIM + 1):
        if powers(dim, state_order).shape[1] == K.shape[0]:
            return dim
    raise ValueError(f"{K.shape[0]} is not the number of monomials of order {state_order} of a state")


def monomial_positions(dim, order, common_dim, common_order, coordinates=None):
    """Positions in the dictionary of `common_order` in `common_dim` variables of the monomials of `order` in `dim`.

    Variable k of the smaller dictionary is variable `coordinates[k]` of the common one (default: the first `dim`);
    a monomial keeps its exponents and has exponent zero in the other variables."""
    coordinates = list(range(dim)) if coordinates is None else list(coordinates)
    if len(coordinates) != dim or len(set(coordinates)) != dim or not all(0 <= c < common_dim for c in coordinates):
        raise ValueError(f"{coordinates} does not place {dim} variables among {common_dim}")
    if order > common_order:
        raise ValueError(f"order {order} exceeds the order {common_order} of the common basis")
    padded = np.zeros((common_dim, powers(dim, order).shape[1]), dtype=int)
    padded[coordinates] = powers(dim, order)
    lookup = {tuple(exponents): j for j, exponents in enumerate(powers(common_dim, common_order).T)}
    return np.array([lookup[tuple(exponents)] for exponents in padded.T])


def project_onto_common_basis(K, state_order, action_order, common_state_order, common_action_order, coordinates=None):
    """The tensor K (phi, phi, psi) of monomial dictionaries, written in the dictionaries of the common orders.

    Every coefficient K[i, j, z] is placed at the positions of its three monomials (the predicted state monomial i,
    the state monomial j and the action monomial z); all other coefficients of the result are zero. `coordinates`
    places a state of fewer than three variables among the three of the common state (see `monomial_positions`)."""
    if K.shape != (K.shape[0], K.shape[0], action_order + 1):
        raise ValueError(f"a tensor of shape {K.shape} does not have action order {action_order}")
    state_dim = state_dimension(K, state_order)
    state = monomial_positions(state_dim, state_order, COMMON_STATE_DIM, common_state_order, coordinates)
    action = monomial_positions(1, action_order, 1, common_action_order)
    n = powers(COMMON_STATE_DIM, common_state_order).shape[1]
    common = np.zeros((n, n, common_action_order + 1))
    common[np.ix_(state, state, action)] = K
    return common


def restrict_to_common_basis(K, state_order, action_order, common_state_order, common_action_order):
    """The coefficients of K (phi, phi, psi) among the monomials of the (smaller) common orders.

    This is the block of K that predicts the state monomials of the common order from the state and action monomials
    of the common orders. The coefficients of the other monomials of K are dropped, so the result is a part of the
    model and not a model of its own: it does not predict on its own what K predicts."""
    state_dim = state_dimension(K, state_order)
    state = monomial_positions(state_dim, common_state_order, state_dim, state_order)
    action = monomial_positions(1, common_action_order, 1, action_order)
    return K[np.ix_(state, state, action)]


def change_units(K, state_order, action_order, state_scale, action_scale):
    """The tensor of the same model for the state x / state_scale and the action u / action_scale.

    `state_scale` is one number or one number per state coordinate. With phi(x) = D phi(x / s), D the diagonal matrix
    of the monomials of s, and likewise for psi, the prediction phi(x') = K (phi(x) x psi(u)) reads
    K~[i, j, z] = K[i, j, z] D_j D_z / D_i in the new units."""
    state_dim = state_dimension(K, state_order)
    scale = np.broadcast_to(np.asarray(state_scale, dtype=float), (state_dim,))
    d_state = np.prod(scale[:, None] ** powers(state_dim, state_order), axis=0)
    d_action = float(action_scale) ** powers(1, action_order)[0]
    return K * d_state[None, :, None] * d_action[None, None, :] / d_state[:, None, None]


# --------------------------------------------------------------------------- #
# Data and identification
# --------------------------------------------------------------------------- #
_PATHS = {}


def random_agent_paths(env_id, seed, num_paths, steps, linear_system_seed=0):
    """Random-agent trajectories as in `generate_koopman_tensor`; arrays X, U, Y of shape (num_paths, steps, dim).

    With a negative `linear_system_seed` the calls are those of `generate_koopman_tensor`, in its order, and the data
    are its data: the linear system draws its matrix A from NumPy's generator seeded with the data seed, and the
    initial states continue that stream. Otherwise A is drawn from `linear_system_seed` and the generator is seeded
    with the data seed afterwards, so that the data seed sets the initial states and the actions only. The other
    environments draw nothing at construction, so their data are those of `generate_koopman_tensor` in both cases
    (the tests compare the data)."""
    key = (env_id, seed, num_paths, steps, linear_system_seed)
    if key not in _PATHS:
        np.random.seed(seed if linear_system_seed < 0 else linear_system_seed)
        torch.manual_seed(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            env = gym.make(env_id)
        if linear_system_seed >= 0:
            np.random.seed(seed)
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


def least_squares_tensor(X, U, Y, state_order, action_order, driver=LSTSQ_DRIVER):
    """The Koopman tensor K (phi, phi, psi) of transitions X, U, Y (n, dim) by ordinary least squares.

    This is the regression of the package's class (`KoopmanTensor` with the regressor "ols"): the same monomial
    dictionaries, the same matrix of regressors kron(psi(u_n), phi(x_n)), `torch.linalg.lstsq` with the same
    threshold for the rank and the same unfolding of M into K. `driver` is the LAPACK routine of the solver: with
    "gelsy" the call is that of the package, whose result is not the same in every call; "gelsd" (the default) and
    "gelss" return the same tensor in every call (see "Reproducibility" in the module docstring). The class builds
    the regressors in a Python loop over the transitions and computes ranks and condition numbers that it only
    prints, which makes it several times slower on the largest budgets; the tests check that both give the same
    tensor up to the rounding of the solver."""
    if driver not in LSTSQ_DRIVERS:
        raise ValueError(f"unknown least-squares driver '{driver}'")
    phi, psi = monomials(state_order), monomials(action_order)
    Phi_X, Phi_Y = phi(torch.tensor(np.ascontiguousarray(X.T))), phi(torch.tensor(np.ascontiguousarray(Y.T)))
    Psi_U = psi(torch.tensor(np.ascontiguousarray(U.T)))
    regressors = (Psi_U[:, None, :] * Phi_X[None, :, :]).reshape(-1, len(X))
    M = torch.linalg.lstsq(regressors.T, Phi_Y.T, rcond=None, driver=driver).solution.T.numpy()
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    phi_dim, psi_dim = Phi_X.shape[0], Psi_U.shape[0]
    return np.stack([M[i].reshape((phi_dim, psi_dim), order="F") for i in range(phi_dim)])


def identify(
    env_id,
    seed,
    state_order,
    action_order,
    num_paths,
    steps,
    data_set=None,
    linear_system_seed=0,
    from_end=False,
    driver=LSTSQ_DRIVER,
):
    """The Koopman tensor of one configuration of the sweep.

    The data are the first `steps` transitions of the first (`from_end`: the last) `num_paths` trajectories of a data
    set of `data_set = (trajectories, steps)`. The default is `(num_paths, steps)`, the data set of
    `generate_koopman_tensor`; a larger one lets the budgets of a sweep share the data of a seed. `driver` is the
    LAPACK driver of the regression (see `least_squares_tensor`)."""
    paths = random_agent_paths(env_id, seed, *(data_set or (num_paths, steps)), linear_system_seed)
    chosen = slice(-num_paths, None) if from_end else slice(num_paths)
    X, U, Y = (a[chosen, :steps].reshape(-1, a.shape[-1]) for a in paths)
    return least_squares_tensor(X, U, Y, state_order, action_order, driver)


def sweep_grids(args, name):
    """The grids of the sweep of one benchmark: (state orders, action orders, paths, steps, seeds, from the end)."""
    if args.sweep == "inferred_layout":
        return [(*grid[:4], [INFERRED_LAYOUT_SEED], grid[4]) for grid in INFERRED_LAYOUT]
    if args.sweep == "orders":
        tuned = CONFIGS[NAMES[name]]
        seeds = list(range(args.seed0, args.seed0 + args.seeds))
        num_paths = args.num_paths or [tuned["num_paths"]]
        steps = args.num_steps_per_path or [tuned["steps"]]
        return [(args.state_orders, args.action_orders, num_paths, steps, seeds, False)]
    raise ValueError(f"unknown sweep '{args.sweep}'")


def sweep_configurations(args):
    """One row (benchmark, state order, action order, num_paths, num_steps_per_path, seed, grid) per tensor, in the
    order of the output files, and for each benchmark the size of the data set that its budgets are cut from."""
    rows, data_sets = [], {}
    for name in args.environments:
        mine = []
        for index, grid in enumerate(sweep_grids(args, name)):
            mine += [(name, *row, index) for row in itertools.product(*grid[:5])]
        if args.sweep == "inferred_layout":
            mine = [row for row in mine if row[1:3] != INFERRED_LAYOUT_OMITTED_ORDERS]
        data_sets[name] = (max(row[3] for row in mine), max(row[4] for row in mine))
        rows += mine
    return rows, data_sets


def tensors_path(output_dir, name):
    return os.path.join(output_dir, f"tensors_{name}.npz")


def save_tensors(path, rows, tensors, settings=None):
    """The tensors K (as identified, before the projection) of one benchmark and their rows of the sweep."""
    arrays = {f"K{k}": K for k, K in enumerate(tensors)}
    np.savez_compressed(path, configurations=json.dumps(rows), settings=json.dumps(settings), **arrays)
    print(f"wrote {path}", flush=True)


def load_tensors(path):
    """Rows, tensors and settings stored by `save_tensors`."""
    with np.load(path) as stored:
        rows = [tuple(row) for row in json.loads(str(stored["configurations"]))]
        return rows, [stored[f"K{k}"] for k in range(len(rows))], json.loads(str(stored["settings"]))


def sweep_settings(args, name, data_sets):
    """What is stored with the tensors of a benchmark besides their rows: the settings that its tensors depend on."""
    if args.lstsq_driver not in LSTSQ_DRIVERS:
        raise ValueError(f"unknown least-squares driver '{args.lstsq_driver}'")
    settings = dict(data_set=list(data_sets[name]), lstsq_driver=args.lstsq_driver)
    if name == "linear_system":  # the only benchmark that draws something at construction
        settings["linear_system_seed"] = args.linear_system_seed
    return settings


def stored_mismatch(stored_rows, stored_settings, rows, settings):
    """Why stored tensors are not the tensors of the sweep of `rows` and `settings`; None if they are."""
    if stored_rows != rows:
        return f"it holds {len(stored_rows)} tensors of another sweep (the arguments give {len(rows)} other rows)"
    stored_settings = dict(stored_settings or {})
    for key in sorted(set(stored_settings) | set(settings)):
        if stored_settings.get(key) != settings.get(key):
            return f"its {key} is {stored_settings.get(key)}, the arguments give {settings.get(key)}"
    return None


def identify_sweep(args):
    """The tensors of the sweep, one benchmark after the other; each benchmark is stored as soon as it is done.

    Only the data set of the benchmark at hand is in memory. With `--resume`, a benchmark whose stored file holds
    the rows and settings of this sweep (data set, seed of the linear system, driver of the solver) is read instead
    of identified."""
    rows, data_sets = sweep_configurations(args)
    tensors, start = [], time.time()
    for name in args.environments:
        mine = [row for row in rows if row[0] == name]
        path = tensors_path(args.output_dir, name)
        settings = sweep_settings(args, name, data_sets)
        if args.resume and os.path.exists(path):
            stored_rows, stored, stored_settings = load_tensors(path)
            mismatch = stored_mismatch(stored_rows, stored_settings, mine, settings)
            if mismatch is None:
                tensors += stored
                print(f"read the tensors of {name} from {path}", flush=True)
                continue
            print(f"{path} is not reused: {mismatch}", flush=True)
        grids = sweep_grids(args, name)
        identified = [
            identify(
                NAMES[name],
                row[5],
                *row[1:5],
                data_sets[name],
                args.linear_system_seed,
                grids[row[6]][5],
                args.lstsq_driver,
            )
            for row in mine
        ]
        _PATHS.clear()  # the data of a benchmark are not needed again
        save_tensors(path, mine, identified, settings)
        tensors += identified
        print(f"identified the tensors of {name} ({time.time() - start:.0f} s since the start)", flush=True)
    return rows, tensors


def load_sweep(args):
    """The stored tensors of the benchmarks of `args.environments`, which have to be those of the sweep of `args`.

    The arguments of the identification are written into `settings.json` next to the embedding, so stored tensors of
    other rows, of another data set, of another linear system or of another driver of the solver are refused."""
    expected, data_sets = sweep_configurations(args)
    rows, tensors = [], []
    for name in args.environments:
        path = tensors_path(args.output_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"--embed_only: {path} does not exist; run without --embed_only to identify")
        stored_rows, stored, stored_settings = load_tensors(path)
        mine = [row for row in expected if row[0] == name]
        mismatch = stored_mismatch(stored_rows, stored_settings, mine, sweep_settings(args, name, data_sets))
        if mismatch is not None:
            raise ValueError(
                f"--embed_only: {path} does not hold the tensors of the sweep of the arguments: {mismatch}. Pass the "
                "arguments the tensors were identified with (--sweep and its arguments, --linear_system_seed, "
                "--lstsq_driver), or identify the sweep of the arguments without --embed_only."
            )
        rows += stored_rows
        tensors += stored
    return rows, tensors


# --------------------------------------------------------------------------- #
# Features and embedding
# --------------------------------------------------------------------------- #
def common_orders(rows, common_basis="largest"):
    """The state order and the action order of the common basis of the tensors of `rows`."""
    if common_basis not in ("largest", "smallest"):
        raise ValueError(f"unknown common basis '{common_basis}'")
    pick = max if common_basis == "largest" else min
    return pick(row[1] for row in rows), pick(row[2] for row in rows)


def feature_matrix(rows, tensors, args):
    """One row per tensor: the tensor in the common basis, flattened, after the transformations of the arguments."""
    common_state_order, common_action_order = common_orders(rows, args.common_basis)
    exponents = powers(COMMON_STATE_DIM, common_state_order)
    absent = [c for c in range(COMMON_STATE_DIM) if c not in args.double_well_coordinates]
    shared = np.all(exponents[absent] == 0, axis=0)
    features = []
    for (name, state_order, action_order, *_), K in zip(rows, tensors):
        env_id = NAMES[name]
        K = np.array(K, dtype=float)
        if args.subtract_persistence:  # phi(x') = phi(x) is K[i, i, 0] = 1, because psi_0 = 1
            K[np.arange(K.shape[0]), np.arange(K.shape[0]), 0] -= 1.0
        if args.units == "box":
            K = change_units(K, state_order, action_order, *BENCHMARKS[env_id]["scale"])
        elif args.units != "raw":
            raise ValueError(f"unknown units '{args.units}'")
        if args.common_basis == "smallest":
            K = restrict_to_common_basis(K, state_order, action_order, common_state_order, common_action_order)
            state_order, action_order = common_state_order, common_action_order
        coordinates = args.double_well_coordinates if env_id == "DoubleWell-v0" else None
        common = project_onto_common_basis(
            K, state_order, action_order, common_state_order, common_action_order, coordinates
        )
        if args.shared_coordinates_only:
            common = common[np.ix_(shared, shared, np.arange(common.shape[2]))]
        features.append(common.ravel())
    features = np.array(features)
    if args.scaling == "standard":
        # coefficients that are the same in every tensor up to the rounding of the solver would turn into noise of
        # unit variance; they are dropped
        spread = features.std(axis=0)
        varying = spread > 1e-9 * np.abs(features).max(axis=0)
        features = (features[:, varying] - features[:, varying].mean(axis=0)) / spread[varying]
    elif args.scaling == "unit_norm":
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
    elif args.scaling != "none":
        raise ValueError(f"unknown scaling '{args.scaling}'")
    return features


def embed(features, args):
    """The two-dimensional t-SNE of the rows of `features`."""
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        metric=args.metric,
        init=args.tsne_init,
        method=args.tsne_method,
        random_state=args.tsne_seed,
    )
    return tsne.fit_transform(features)


def benchmark_order(rows):
    """The benchmarks of `rows`, in the order of their first row."""
    return list(dict.fromkeys(row[0] for row in rows))


def separation(points, rows, neighbours=10, metric="euclidean"):
    """How well the benchmarks separate among `points` (one row per tensor), per benchmark and for all tensors.

    silhouette
        Mean silhouette coefficient with the benchmarks as the clusters: near 1 for compact clusters that are far
        apart, near 0 for overlapping ones, negative for tensors that are closer to another benchmark than to their
        own.
    purity
        Mean share of the `neighbours` nearest neighbours of a tensor that belong to its benchmark.
    purity_other_configuration
        The same after the tensors of its own benchmark with its dictionary orders and its identification budget have
        been removed from the candidates. These are its seed replicates, which are nearly the same tensor, so
        `purity` is close to 1 whenever a configuration has more seeds than `neighbours`, whatever the rest of the
        embedding looks like. This score asks whether a tensor is closer to the tensors of other configurations of
        its benchmark than to the tensors of the other benchmarks. Without separation it is about the share of the
        benchmark among the tensors (0.25 for four benchmarks of equal size). It is not defined (nan) for a benchmark
        with a single configuration.
    """
    names = benchmark_order(rows)
    labels = np.array([names.index(row[0]) for row in rows])
    configurations = np.array([hash(tuple(row[:5])) for row in rows])
    k = min(neighbours, len(points) - 1)
    if len(names) > 1:
        silhouette = silhouette_samples(points, labels, metric=metric)
    else:
        silhouette = np.full(len(points), np.nan)
    ranking = NearestNeighbors(n_neighbors=len(points), metric=metric).fit(points).kneighbors(points)[1]
    purity, purity_other = np.full(len(points), np.nan), np.full(len(points), np.nan)
    for n, ranked in enumerate(ranking):
        ranked = ranked[ranked != n]
        purity[n] = np.mean(labels[ranked[:k]] == labels[n])
        other = ranked[configurations[ranked] != configurations[n]]
        if np.any(labels[other] == labels[n]):
            purity_other[n] = np.mean(labels[other[:k]] == labels[n])

    def mean(values):
        return float(np.mean(values[np.isfinite(values)])) if np.isfinite(values).any() else float("nan")

    scores = {}
    for name, mine in [(name, labels == label) for label, name in enumerate(names)] + [("all", labels >= 0)]:
        scores[name] = dict(
            tensors=int(mine.sum()),
            silhouette=mean(silhouette[mine]),
            purity=mean(purity[mine]),
            purity_other_configuration=mean(purity_other[mine]),
        )
    return scores


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_coordinates(output_dir, rows, embedding):
    """`<benchmark>_tsne.csv`, the files that the TikZ source of the paper reads, and the combined `tsne.csv`."""
    combined = [",".join(["benchmark", "class"] + COLUMNS)]
    for name in benchmark_order(rows):
        lines = [",".join(COLUMNS)]
        mine = [(row, point) for row, point in zip(rows, embedding) if row[0] == name]
        for index, (row, point) in enumerate(mine):
            fields = [str(index), f"{point[0]:.6f}", f"{point[1]:.6f}"] + [str(value) for value in row[1:]]
            lines.append(",".join(fields))
            combined.append(",".join([name, str(list(NAMES).index(name))] + fields))
        path = os.path.join(output_dir, f"{name}_tsne.csv")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"wrote {path}")
    path = os.path.join(output_dir, "tsne.csv")
    with open(path, "w") as f:
        f.write("\n".join(combined) + "\n")
    print(f"wrote {path}")


def write_separation(output_dir, scores):
    """`separation.csv`: the scores of `separation` in the embedding and among the flattened tensors."""
    keys = ["silhouette", "purity", "purity_other_configuration"]
    lines = [",".join(["benchmark", "tensors"] + [f"{key}_{space}" for space in scores for key in keys])]
    for name in scores["embedding"]:
        fields = [name, str(scores["embedding"][name]["tensors"])]
        lines.append(",".join(fields + [f"{scores[space][name][key]:.4f}" for space in scores for key in keys]))
    path = os.path.join(output_dir, "separation.csv")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


TIKZ_TEMPLATE = r"""\documentclass[crop,tikz]{standalone}

\usepackage{pgfplots}
\pgfplotsset{compat=1.17}

% Define colors for the individual clusters
<colours>

\begin{document}

\begin{tikzpicture}
    \begin{axis}[
        enlarge y limits=true,
        enlarge x limits=true,
        yticklabel={\empty},
        xticklabel={\empty},
        legend columns=<columns>,
        axis lines=left,
        width=20cm,
        height=10cm,
        legend image post style={scale=1.5},
        legend style={
            at={(0.5,1.15)},
            anchor=north,
            column sep=0.3cm,
            font=\large,
            draw=white
        }
    ]

<plots>

    \end{axis}
\end{tikzpicture}

\end{document}
"""


def write_tikz(output_dir, names):
    """`t_sne_figure.tex`: the standalone pgfplots figure of the paper, reading the files next to it.

    It is the source of the figure of the paper (`figures/sources/t_sne_figure.tex` of the paper project) with three
    changes: the tables are read from the directory of the file instead of `data/tsne/`; the axis limits, which the
    paper sets for its own coordinates, are left to pgfplots; and the comment lines around the plots are left out."""
    colours, plots = [], []
    for name in names:
        entry = BENCHMARKS[NAMES[name]]
        colours.append("\\definecolor{%s}{RGB}{%d, %d, %d}" % (entry["colour"], *entry["rgb"]))
        plots.append(
            "    \\addplot[only marks, thick, %s, mark=%s] " % (entry["colour"], entry["mark"])
            + "table [x=x-val, y=y-val, col sep=comma] {%s_tsne.csv};\n" % name
            + "    \\addlegendentry{%s}" % entry["title"]
        )
    text = TIKZ_TEMPLATE.replace("<colours>", "\n".join(colours)).replace("<plots>", "\n\n".join(plots))
    path = os.path.join(output_dir, "t_sne_figure.tex")
    with open(path, "w") as f:
        f.write(text.replace("<columns>", str(len(names))))
    print(f"wrote {path}")


def write_preview(output_dir, rows, embedding):
    """`tsne_preview.pdf` and `.png`: the scatter by benchmark, and the same points shaded by state order."""
    state_orders = np.array([row[1] for row in rows])
    shades = plt.get_cmap("viridis")(np.linspace(0.0, 0.9, max(2, int(state_orders.max()))))
    figure, (left, right) = plt.subplots(1, 2, figsize=(15, 6.2))
    names = benchmark_order(rows)
    for name in names:
        entry = BENCHMARKS[NAMES[name]]
        mine = np.array([row[0] == name for row in rows])
        colour = [c / 255 for c in entry["rgb"]]
        left.scatter(*embedding[mine].T, marker=entry["marker"], s=26, color=colour, linewidths=0, label=entry["title"])
        right.scatter(
            *embedding[mine].T, marker=entry["marker"], s=26, color=shades[state_orders[mine] - 1], linewidths=0
        )
    left.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(names), frameon=False)
    handles = [
        plt.Line2D([], [], linestyle="none", marker="o", color=shades[order - 1], label=f"state order {order}")
        for order in sorted(set(state_orders.tolist()))
    ]
    right.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles), frameon=False)
    for axis in (left, right):
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    for extension in ("pdf", "png"):
        path = os.path.join(output_dir, f"tsne_preview.{extension}")
        figure.savefig(path, dpi=150)
        print(f"wrote {path}")
    plt.close(figure)


def peak_memory_mb():
    """Peak resident memory of the process in MB (`ru_maxrss` is in kilobytes on Linux and in bytes on macOS)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024**2 if sys.platform == "darwin" else 1024))


def numerical_threads():
    """Numbers of threads of the numerical libraries of the process, by interface, and of torch."""
    threads = {"torch": torch.get_num_threads()}
    for pool in threadpool_info():
        threads[pool["user_api"]] = max(threads.get(pool["user_api"], 0), pool["num_threads"])
    return threads


def run(args):
    """Identify (or load) the tensors, embed them and write the output files; returns rows, embedding and scores.

    With `--identify_only` the tensors are identified and stored, and the embedding and the scores are None."""
    if args.identify_only and args.embed_only:
        raise ValueError("--identify_only and --embed_only exclude each other")
    os.makedirs(args.output_dir, exist_ok=True)
    start = time.time()
    rows, tensors = load_sweep(args) if args.embed_only else identify_sweep(args)
    identification_seconds = time.time() - start
    if args.identify_only:
        return rows, None, None
    features = feature_matrix(rows, tensors, args)
    embedding = embed(features, args)
    scores = dict(
        embedding=separation(embedding, rows, args.neighbours),
        tensors=separation(features, rows, args.neighbours, args.metric),
    )
    write_coordinates(args.output_dir, rows, embedding)
    write_separation(args.output_dir, scores)
    write_tikz(args.output_dir, benchmark_order(rows))
    write_preview(args.output_dir, rows, embedding)
    settings = dict(
        arguments=args.as_dict(),
        tensors=len(rows),
        coefficients_per_tensor=int(features.shape[1]),
        common_state_order=common_orders(rows, args.common_basis)[0],
        common_action_order=common_orders(rows, args.common_basis)[1],
        identification_seconds=None if args.embed_only else round(identification_seconds, 1),
        total_seconds=round(time.time() - start, 1),
        peak_memory_mb=peak_memory_mb(),
        threads=numerical_threads(),
        separation=scores,
    )
    path = os.path.join(args.output_dir, "settings.json")
    with open(path, "w") as f:
        json.dump(settings, f, indent=2)
    print(f"wrote {path}")
    for name, entry in scores["embedding"].items():
        print(
            f"{name:14s} silhouette {entry['silhouette']:6.3f}  purity {entry['purity']:.3f}  "
            f"purity among other configurations {entry['purity_other_configuration']:.3f}"
        )
    return rows, embedding, scores


def main():
    run(ArgumentParser().parse_args())


if __name__ == "__main__":
    main()
