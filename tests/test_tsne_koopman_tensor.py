"""koopmanrl_utils.tsne_koopman_tensor: the projection onto the common basis keeps the model, the tensors are those of
the package, and the output files are what the module says they are."""

import contextlib
import inspect
import io
import json
import os
import re
import subprocess
import sys

import gym
import numpy as np
import pytest
import torch

import koopmanrl.environments  # noqa: F401
import koopmanrl_utils.tsne_koopman_tensor as tsne
from koopmanrl.koopman_tensor.observables.torch_observables import monomials
from koopmanrl.soft_koopman_value_iteration import generate_koopman_tensor

TINY = ["--sweep", "orders", "--state_orders", "1", "2", "--action_orders", "1", "2", "--seeds", "2"]
TINY += ["--num_paths", "4", "--num_steps_per_path", "30", "--perplexity", "5"]
REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(tsne.__file__)))


def lift(x, order):
    """Monomial dictionary of the package for samples x (n, dim) -> (n_monomials, n)."""
    return monomials(order)(torch.tensor(np.atleast_2d(x).T)).numpy()


def predict(K, x, u, state_order, action_order):
    """K^u phi(x) for samples x (n, dim), u (n, 1) -> (n_monomials, n)."""
    return np.einsum("ijz,zn,jn->in", K, lift(u, action_order), lift(x, state_order))


def small_tensor(env_id, state_order, action_order, seed=0):
    return tsne.identify(env_id, seed, state_order, action_order, 6, 40)


def tiny_run(tmp_path, *extra):
    tsne._PATHS.clear()
    args = tsne.ArgumentParser().parse_args(TINY + ["--output_dir", str(tmp_path)] + list(extra))
    with contextlib.redirect_stdout(io.StringIO()):
        return args, tsne.run(args)


def test_monomials_keep_their_exponents_in_the_common_basis():
    common = tsne.powers(3, 4)
    for dim, order, coordinates in [(3, 2, None), (2, 3, None), (2, 3, [2, 0])]:
        positions = tsne.monomial_positions(dim, order, 3, 4, coordinates)
        assert len(set(positions.tolist())) == len(positions) == tsne.powers(dim, order).shape[1]
        placed = np.zeros((3, len(positions)), dtype=int)
        placed[list(range(dim)) if coordinates is None else coordinates] = tsne.powers(dim, order)
        assert np.array_equal(common[:, positions], placed)
    assert np.array_equal(tsne.monomial_positions(3, 4, 3, 4), np.arange(35))
    assert np.array_equal(tsne.monomial_positions(1, 2, 1, 4), [0, 1, 2])
    with pytest.raises(ValueError):
        tsne.monomial_positions(3, 4, 3, 2)
    with pytest.raises(ValueError):
        tsne.monomial_positions(2, 2, 3, 4, [0, 0])


@pytest.mark.parametrize(
    "env_id, state_order, action_order, coordinates",
    [
        ("FluidFlow-v0", 2, 1, None),
        ("Lorenz-v0", 1, 3, None),
        ("DoubleWell-v0", 2, 2, [0, 1]),
        ("DoubleWell-v0", 3, 1, [2, 0]),
    ],
)
def test_projected_tensor_predicts_what_the_tensor_predicts(env_id, state_order, action_order, coordinates):
    # a tensor identified at low orders, evaluated through the dictionaries of the common basis, gives the same
    # prediction K^u phi(x) of its own dictionary functions, and predicts zero for the functions it does not have
    K = small_tensor(env_id, state_order, action_order)
    state_dim = tsne.state_dimension(K, state_order)
    common = tsne.project_onto_common_basis(K, state_order, action_order, 4, 4, coordinates)
    assert common.shape == (35, 35, 5)
    assert np.count_nonzero(common) == np.count_nonzero(K)

    rng = np.random.RandomState(0)
    x_common, u = rng.uniform(-1.5, 1.5, (25, 3)), rng.uniform(-2.0, 2.0, (25, 1))
    x = x_common[:, list(range(state_dim)) if coordinates is None else coordinates]
    positions = tsne.monomial_positions(state_dim, state_order, 3, 4, coordinates)
    assert np.allclose(lift(x_common, 4)[positions], lift(x, state_order))  # the shared dictionary functions

    prediction = predict(common, x_common, u, 4, 4)
    assert np.allclose(prediction[positions], predict(K, x, u, state_order, action_order), rtol=1e-12, atol=1e-12)
    others = np.setdiff1d(np.arange(35), positions)
    assert np.array_equal(prediction[others], np.zeros((len(others), 25)))


def test_projection_onto_its_own_basis_is_the_tensor():
    K = small_tensor("FluidFlow-v0", 2, 2)
    assert np.array_equal(tsne.project_onto_common_basis(K, 2, 2, 2, 2), K)
    with pytest.raises(ValueError):
        tsne.project_onto_common_basis(K, 2, 1, 4, 4)  # the action order does not match the tensor


def test_restriction_to_the_smallest_basis_is_a_block_of_the_tensor():
    K = small_tensor("DoubleWell-v0", 3, 2)
    block = tsne.restrict_to_common_basis(K, 3, 2, 1, 1)
    assert block.shape == (3, 3, 2)
    # restricting the projection is projecting the restriction
    common = tsne.project_onto_common_basis(K, 3, 2, 4, 4)
    positions = tsne.monomial_positions(2, 1, 3, 4)
    assert np.array_equal(common[np.ix_(positions, positions, [0, 1])], block)
    assert np.array_equal(tsne.restrict_to_common_basis(K, 3, 2, 3, 2), K)


@pytest.mark.parametrize(
    "env_id, scale", [("Lorenz-v0", (20.0, 50.0, 50.0)), ("Lorenz-v0", 25.0), ("DoubleWell-v0", (2.0, 0.5))]
)
def test_change_of_units_keeps_the_model(env_id, scale):
    K = small_tensor(env_id, 2, 2)
    state_dim = tsne.state_dimension(K, 2)
    scaled = tsne.change_units(K, 2, 2, scale, 75.0)
    rng = np.random.RandomState(1)
    x, u = rng.uniform(-20, 20, (15, state_dim)), rng.uniform(-50, 50, (15, 1))
    per_coordinate = np.broadcast_to(np.asarray(scale, dtype=float), (state_dim,))
    monomials_of_scale = np.prod(per_coordinate[:, None] ** tsne.powers(state_dim, 2), axis=0)
    prediction = predict(scaled, x / per_coordinate, u / 75.0, 2, 2) * monomials_of_scale[:, None]
    assert np.allclose(prediction, predict(K, x, u, 2, 2), rtol=1e-9, atol=1e-9)


def test_box_units_are_the_bounds_of_the_environments():
    # for every state coordinate and the action: the largest absolute value of its bounds
    for env_id, entry in tsne.BENCHMARKS.items():
        np.random.seed(0)
        with contextlib.redirect_stdout(io.StringIO()):
            env = gym.make(env_id)
        state_scale, action_scale = entry["scale"]
        observations, actions = env.observation_space, env.action_space
        assert np.array_equal(state_scale, np.maximum(np.abs(observations.low), np.abs(observations.high)))
        assert np.array_equal([action_scale], np.maximum(np.abs(actions.low), np.abs(actions.high)))
    assert tsne.BENCHMARKS["Lorenz-v0"]["scale"] == ((20.0, 50.0, 50.0), 75.0)


def packaged_data(env_id, seed, num_paths, steps):
    """Transitions (n, dim) and tensor of koopmanrl.soft_koopman_value_iteration.generate_koopman_tensor."""
    with contextlib.redirect_stdout(io.StringIO()):
        packaged = generate_koopman_tensor(env_id, seed, num_paths, steps, 2, 2, "ols")
    return [a.numpy().T for a in (packaged.X, packaged.U, packaged.Y)], packaged.K.numpy()


@pytest.mark.parametrize("env_id", ["LinearSystem-v0", "FluidFlow-v0", "Lorenz-v0", "DoubleWell-v0"])
def test_data_and_tensor_are_those_of_the_package(env_id):
    # with the matrix of the linear system drawn from the data seed, the data are those of generate_koopman_tensor,
    # value for value, and the tensor is its tensor up to the rounding of the solver: with "gelsy", the driver of
    # koopmanrl.koopman_tensor, and with "gelsd", the default driver of the script and the driver of SKVI and SAKC
    packaged, packaged_K = packaged_data(env_id, 3, 5, 40)
    tsne._PATHS.clear()
    paths = tsne.random_agent_paths(env_id, 3, 5, 40, linear_system_seed=-1)
    for mine, theirs in zip(paths, packaged):
        assert np.array_equal(mine.reshape(-1, mine.shape[-1]), theirs)
    assert tsne.LSTSQ_DRIVER == "gelsd"
    for driver in ("gelsy", tsne.LSTSQ_DRIVER):  # the driver of koopmanrl.koopman_tensor, and the default
        K = tsne.identify(env_id, 3, 2, 2, 5, 40, linear_system_seed=-1, driver=driver)
        assert K.shape == packaged_K.shape
        assert np.allclose(K, packaged_K, rtol=0.0, atol=1e-9 * np.abs(packaged_K).max())
    with pytest.raises(ValueError):
        tsne.identify(env_id, 3, 2, 2, 5, 40, linear_system_seed=-1, driver="gels")


@pytest.mark.parametrize("env_id", ["LinearSystem-v0", "FluidFlow-v0", "Lorenz-v0", "DoubleWell-v0"])
def test_fixed_linear_system_changes_the_data_of_the_linear_system_only(env_id):
    # the default draws the matrix of the linear system from seed 0 instead of the data seed; the environments that
    # draw nothing at construction keep the data of the package
    packaged, _ = packaged_data(env_id, 3, 5, 40)
    tsne._PATHS.clear()
    X, U, Y = (a.reshape(-1, a.shape[-1]) for a in tsne.random_agent_paths(env_id, 3, 5, 40))
    assert np.array_equal(U, packaged[1])  # the actions come from the generator of the action space
    if env_id == "LinearSystem-v0":
        assert not np.allclose(X, packaged[0]) and not np.allclose(Y, packaged[2])
    else:
        assert np.array_equal(X, packaged[0]) and np.array_equal(Y, packaged[2])


def test_fixed_linear_system_does_not_depend_on_the_data_seed():
    tsne._PATHS.clear()
    fixed = [tsne.identify("LinearSystem-v0", seed, 1, 1, 5, 40) for seed in (0, 1)]
    drawn = [tsne.identify("LinearSystem-v0", seed, 1, 1, 5, 40, linear_system_seed=-1) for seed in (0, 1)]
    assert np.allclose(fixed[0], fixed[1], atol=1e-9)  # x' = A x + B u is in the dictionary: the same tensor
    assert not np.allclose(drawn[0], drawn[1], atol=1e-3)


def test_budgets_of_a_sweep_are_cut_from_one_data_set():
    tsne._PATHS.clear()
    X, U, Y = tsne.random_agent_paths("DoubleWell-v0", 0, 6, 40)
    K = tsne.identify("DoubleWell-v0", 0, 2, 1, 3, 20, data_set=(6, 40))
    cut = [a[:3, :20].reshape(-1, a.shape[-1]) for a in (X, U, Y)]
    assert np.allclose(K, tsne.least_squares_tensor(*cut, 2, 1), rtol=1e-9, atol=1e-11)
    assert list(tsne._PATHS) == [("DoubleWell-v0", 0, 6, 40, 0)]
    last = [a[-3:, :20].reshape(-1, a.shape[-1]) for a in (X, U, Y)]
    from_end = tsne.identify("DoubleWell-v0", 0, 2, 1, 3, 20, data_set=(6, 40), from_end=True)
    assert np.allclose(from_end, tsne.least_squares_tensor(*last, 2, 1), rtol=1e-9, atol=1e-11)
    assert not np.allclose(from_end, K, atol=1e-6)


def test_inferred_layout_has_the_size_of_the_published_files():
    args = tsne.ArgumentParser().parse_args([])
    rows, data_sets = tsne.sweep_configurations(args)
    for name in tsne.NAMES:
        mine = [row for row in rows if row[0] == name]
        assert len(mine) == len(set(mine)) == 161
        assert [sum(row[6] == grid for row in mine) for grid in range(3)] == [121, 25, 15]
        assert all(row[1:3] == (2, 2) for row in mine if row[6] < 2)
        assert all(row[3:5] == (100, 300) for row in mine if row[6] == 2)
        assert (4, 4) not in {row[1:3] for row in mine} and len({row[1:3] for row in mine}) == 15
        assert {row[5] for row in mine} == {tsne.INFERRED_LAYOUT_SEED}
        assert data_sets[name] == (110, 550)
        assert [grid[5] for grid in tsne.sweep_grids(args, name)] == [False, True, False]
    assert tsne.common_orders(rows) == (4, 4)
    assert tsne.common_orders(rows, "smallest") == (1, 1)


def test_separation_scores_tell_benchmarks_from_configurations():
    rng = np.random.RandomState(0)
    rows = [(name, order, 1, 10, 10, seed, 0) for name in ("a", "b") for order in (1, 2) for seed in range(12)]
    noise = 0.01 * rng.randn(len(rows), 2)

    def points(centres):
        return np.array([[centres[row[0], row[1]], 0.0] for row in rows]) + noise

    # the two configurations of a benchmark lie next to each other, the benchmarks far apart
    by_benchmark = tsne.separation(points({("a", 1): 0.0, ("a", 2): 1.0, ("b", 1): 30.0, ("b", 2): 31.0}), rows)
    assert by_benchmark["all"]["tensors"] == 48 and by_benchmark["a"]["tensors"] == 24
    assert by_benchmark["all"]["silhouette"] > 0.8
    assert by_benchmark["all"]["purity"] == 1.0
    assert by_benchmark["all"]["purity_other_configuration"] == 1.0

    # every (benchmark, configuration) is a cluster of its own, next to a cluster of the other benchmark: the twelve
    # seed replicates still give a purity of 1, the nearest tensors of the other configuration are of the other
    # benchmark
    by_configuration = tsne.separation(points({("a", 1): 0.0, ("b", 2): 10.0, ("b", 1): 30.0, ("a", 2): 40.0}), rows)
    assert by_configuration["all"]["purity"] == 1.0
    assert by_configuration["all"]["silhouette"] < 0.5
    assert by_configuration["all"]["purity_other_configuration"] == 0.0
    assert by_configuration["b"]["purity_other_configuration"] == 0.0

    # replicates are those of the tensor's own benchmark: tensors of another benchmark with the same orders and
    # budget stay among the candidates, and a benchmark with a single configuration has no score
    single = [row for row in rows if row[1] == 1]
    centres = np.array([[0.0 if row[0] == "a" else 30.0, 0.0] for row in single])
    alone = tsne.separation(centres + noise[: len(single)], single)
    assert alone["all"]["purity"] == 1.0
    assert np.isnan(alone["all"]["purity_other_configuration"])


def test_output_files_of_a_small_sweep(tmp_path):
    args, (rows, embedding, scores) = tiny_run(tmp_path)
    assert len(rows) == 4 * 2 * 2 * 2 and embedding.shape == (32, 2)
    assert np.isfinite(embedding).all()

    for name in tsne.NAMES:  # the files that the TikZ source of the paper reads
        lines = (tmp_path / f"{name}_tsne.csv").read_text().splitlines()
        assert lines[0].split(",") == tsne.COLUMNS
        assert lines[0].split(",")[:3] == ["index", "x-val", "y-val"]
        table = np.array([[float(v) for v in line.split(",")] for line in lines[1:]])
        assert table.shape == (8, len(tsne.COLUMNS))
        assert np.array_equal(table[:, 0], np.arange(8))
        mine = np.array([row[0] == name for row in rows])
        assert np.allclose(table[:, 1:3], embedding[mine], atol=1e-6)
        assert [tuple(int(v) for v in line[3:]) for line in table] == [row[1:] for row in rows if row[0] == name]

    combined = (tmp_path / "tsne.csv").read_text().splitlines()
    assert combined[0].split(",") == ["benchmark", "class"] + tsne.COLUMNS
    assert len(combined) == 1 + 32
    assert [line.split(",")[:2] for line in combined[1::8]] == [[name, str(k)] for k, name in enumerate(tsne.NAMES)]

    separation = (tmp_path / "separation.csv").read_text().splitlines()
    assert len(separation) == 1 + 5 and separation[-1].startswith("all,32,")
    assert separation[0].split(",")[2:5] == [
        "silhouette_embedding",
        "purity_embedding",
        "purity_other_configuration_embedding",
    ]

    settings = json.loads((tmp_path / "settings.json").read_text())
    assert settings["arguments"]["state_orders"] == [1, 2] and settings["arguments"]["tsne_seed"] == 42
    assert settings["tensors"] == 32 and settings["coefficients_per_tensor"] == 10 * 10 * 3
    assert (settings["common_state_order"], settings["common_action_order"]) == (2, 2)
    assert settings["separation"]["embedding"].keys() == scores["embedding"].keys()

    figure = (tmp_path / "t_sne_figure.tex").read_text()
    for name, entry in ((name, tsne.BENCHMARKS[env_id]) for name, env_id in tsne.NAMES.items()):
        assert f"mark={entry['mark']}] table [x=x-val, y=y-val, col sep=comma] {{{name}_tsne.csv}};" in figure
        assert f"\\addlegendentry{{{entry['title']}}}" in figure
    assert (tmp_path / "tsne_preview.png").stat().st_size > 0 and (tmp_path / "tsne_preview.pdf").stat().st_size > 0


def test_stored_tensors_give_the_embedding_again(tmp_path):
    _, (rows, embedding, _) = tiny_run(tmp_path)
    for name in tsne.NAMES:
        stored_rows, stored, settings = tsne.load_tensors(tsne.tensors_path(tmp_path, name))
        assert stored_rows == [row for row in rows if row[0] == name] and len(stored) == 8
        seed_of_the_system = dict(linear_system_seed=0) if name == "linear_system" else {}
        assert settings == dict(data_set=[4, 30], lstsq_driver="gelsd", **seed_of_the_system)
    assert json.loads((tmp_path / "settings.json").read_text())["arguments"]["lstsq_driver"] == "gelsd"
    first = (tmp_path / "tsne.csv").read_text()
    _, (again_rows, again, _) = tiny_run(tmp_path, "--embed_only")
    assert again_rows == rows
    assert np.array_equal(again, embedding)
    assert (tmp_path / "tsne.csv").read_text() == first


def stored_tensors(folder):
    return [K for name in tsne.NAMES for K in tsne.load_tensors(tsne.tensors_path(folder, name))[1]]


def test_tensors_and_tables_of_a_small_sweep_repeat_exactly_with_the_default_driver(tmp_path):
    # with "gelsd" the solver returns the same tensor in every call, so a second run from the seeds gives the same
    # tensors bit for bit and, the t-SNE being deterministic for identical tensors, the same tables
    _, (rows, first, _) = tiny_run(tmp_path / "first")
    _, (again_rows, second, _) = tiny_run(tmp_path / "second")
    assert again_rows == rows and len(rows) == 32
    for a, b in zip(stored_tensors(tmp_path / "first"), stored_tensors(tmp_path / "second")):
        assert a.tobytes() == b.tobytes()
    assert np.array_equal(first, second)
    for file in ["tsne.csv", "separation.csv"] + [f"{name}_tsne.csv" for name in tsne.NAMES]:
        assert (tmp_path / "first" / file).read_bytes() == (tmp_path / "second" / file).read_bytes()

    # a benchmark identified by a call of its own (as a workflow with one job per benchmark does it) has the tensors
    # that it has in the sweep of all benchmarks
    tiny_run(tmp_path / "alone", "--environments", "lorenz")
    alone = tsne.load_tensors(tsne.tensors_path(tmp_path / "alone", "lorenz"))[1]
    joint = tsne.load_tensors(tsne.tensors_path(tmp_path / "first", "lorenz"))[1]
    assert len(alone) == 8 and all(a.tobytes() == b.tobytes() for a, b in zip(alone, joint))


def test_tensors_of_a_small_sweep_are_the_same_in_another_process(tmp_path):
    # the module as a script in a process of its own: the tensors are those of this process, bit for bit
    tiny_run(tmp_path / "here")
    command = [
        sys.executable,
        "-m",
        "koopmanrl_utils.tsne_koopman_tensor",
        *TINY,
        "--output_dir",
        str(tmp_path / "there"),
    ]
    subprocess.run(command, cwd=REPOSITORY, check=True, capture_output=True)
    here, there = stored_tensors(tmp_path / "here"), stored_tensors(tmp_path / "there")
    assert len(here) == len(there) == 32
    for a, b in zip(here, there):
        assert a.tobytes() == b.tobytes()


def test_tensors_of_the_driver_of_the_package_are_reproducible_to_rounding(tmp_path):
    # "gelsy", the driver of koopmanrl.koopman_tensor, returns a tensor that may differ in the last digits from one
    # call to the next (more for the few transitions of this sweep, which condition the regression badly), and the
    # t-SNE turns such differences into different coordinates: from the seeds, its tensors are reproducible to that
    # rounding; the embedding is reproducible from the stored tensors (test_stored_tensors_give_the_embedding_again)
    _, (rows, first, _) = tiny_run(tmp_path / "first", "--lstsq_driver", "gelsy")
    _, (again_rows, second, _) = tiny_run(tmp_path / "second", "--lstsq_driver", "gelsy")
    assert again_rows == rows and first.shape == second.shape
    for name in tsne.NAMES:
        stored = [tsne.load_tensors(tsne.tensors_path(tmp_path / run, name)) for run in ("first", "second")]
        assert stored[0][2]["lstsq_driver"] == stored[1][2]["lstsq_driver"] == "gelsy"
        for a, b in zip(stored[0][1], stored[1][1]):
            assert np.allclose(a, b, rtol=0.0, atol=1e-6 * np.abs(a).max())
    # the three drivers give the same tensors to that rounding
    tiny_run(tmp_path / "gelss", "--lstsq_driver", "gelss")
    tiny_run(tmp_path / "gelsd")
    for a, b, c in zip(*(stored_tensors(tmp_path / run) for run in ("first", "gelss", "gelsd"))):
        assert np.allclose(a, c, rtol=0.0, atol=1e-6 * np.abs(c).max())
        assert np.allclose(b, c, rtol=0.0, atol=1e-6 * np.abs(c).max())
    with pytest.raises(ValueError, match="unknown least-squares driver"):
        tiny_run(tmp_path / "unknown", "--lstsq_driver", "gels")


def test_embedding_of_stored_tensors_refuses_tensors_of_another_sweep(tmp_path):
    # --embed_only writes its arguments into settings.json next to the embedding: stored tensors that are not those
    # of the sweep of the arguments are refused, and nothing is written
    tiny_run(tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    for extra, reason in [
        (["--seed0", "5"], "tensors of another sweep"),
        (["--seeds", "1"], "tensors of another sweep"),
        (["--sweep", "inferred_layout"], "tensors of another sweep"),
        (["--linear_system_seed", "3"], "linear_system_seed is 0, the arguments give 3"),
        (["--lstsq_driver", "gelsy"], "lstsq_driver is gelsd, the arguments give gelsy"),
    ]:
        with pytest.raises(ValueError, match="does not hold the tensors of the sweep of the arguments") as error:
            tiny_run(tmp_path, "--embed_only", *extra)
        assert reason in str(error.value) and "tensors_linear_system.npz" in str(error.value)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
    tiny_run(tmp_path, "--embed_only", "--environments", "lorenz", "double_well")  # a part of the sweep is embedded
    (tmp_path / "tensors_lorenz.npz").unlink()
    with pytest.raises(FileNotFoundError, match="tensors_lorenz.npz does not exist"):
        tiny_run(tmp_path, "--embed_only")


def test_resume_identifies_again_with_another_driver(tmp_path, monkeypatch):
    # stored tensors are reused by --resume only with the driver they were identified with
    args, (rows, _, _) = tiny_run(tmp_path, "--lstsq_driver", "gelsy")
    _, (again_rows, _, _) = tiny_run(tmp_path, "--embed_only", "--lstsq_driver", "gelsy")
    assert again_rows == rows
    assert json.loads((tmp_path / "settings.json").read_text())["arguments"]["lstsq_driver"] == "gelsy"

    identified = []
    identify = tsne.identify

    def counted(env_id, *arguments):
        identified.append(arguments[-1])
        return identify(env_id, *arguments)

    monkeypatch.setattr(tsne, "identify", counted)
    tiny_run(tmp_path, "--resume", "--lstsq_driver", "gelsy")  # read, not identified
    assert identified == []
    tiny_run(tmp_path, "--resume")  # the default driver is another one: identified again, and recorded
    assert identified == ["gelsd"] * 32
    assert all(
        tsne.load_tensors(tsne.tensors_path(tmp_path, name))[2]["lstsq_driver"] == "gelsd" for name in tsne.NAMES
    )


def test_common_basis_and_scaling_options_change_the_features(tmp_path):
    args, (rows, _, _) = tiny_run(tmp_path)
    tensors = tsne.load_sweep(args)[1]
    largest = tsne.feature_matrix(rows, tensors, args)
    assert largest.shape == (32, 300)

    args.common_basis = "smallest"
    assert tsne.feature_matrix(rows, tensors, args).shape == (32, 4 * 4 * 2)
    args.common_basis = "largest"

    args.shared_coordinates_only = True
    shared = tsne.feature_matrix(rows, tensors, args)
    assert shared.shape == (32, 6 * 6 * 3)
    double_well = np.array([row[0] == "double_well" for row in rows])
    # the double well has no coefficient outside the shared coordinates, the other benchmarks do
    assert np.allclose(np.linalg.norm(shared[double_well], axis=1), np.linalg.norm(largest[double_well], axis=1))
    assert np.all(np.linalg.norm(shared[~double_well], axis=1) < np.linalg.norm(largest[~double_well], axis=1))
    args.shared_coordinates_only = False

    args.subtract_persistence = True
    removed = largest - tsne.feature_matrix(rows, tensors, args)
    assert np.all((removed == 0) | np.isclose(removed, 1.0))
    # one coefficient K[i, i, 0] = 1 per dictionary function of the tensor
    functions = [tsne.powers(2 if row[0] == "double_well" else 3, row[1]).shape[1] for row in rows]
    assert np.allclose(removed.sum(axis=1), functions)
    args.subtract_persistence = False

    args.scaling = "unit_norm"
    assert np.allclose(np.linalg.norm(tsne.feature_matrix(rows, tensors, args), axis=1), 1.0)
    args.scaling = "standard"
    standard = tsne.feature_matrix(rows, tensors, args)
    assert np.allclose(standard.mean(axis=0), 0.0, atol=1e-9) and np.allclose(standard.std(axis=0), 1.0)
    args.scaling = "nonsense"
    with pytest.raises(ValueError):
        tsne.feature_matrix(rows, tensors, args)


def test_interrupted_sweep_is_picked_up(tmp_path, monkeypatch):
    args, (rows, embedding, _) = tiny_run(tmp_path)
    (tmp_path / "tensors_lorenz.npz").unlink()  # as if the run had stopped before the third benchmark was stored
    identified = []
    identify = tsne.identify

    def counted(env_id, *arguments):
        identified.append(env_id)
        return identify(env_id, *arguments)

    monkeypatch.setattr(tsne, "identify", counted)
    _, (again_rows, again, _) = tiny_run(tmp_path, "--resume")
    assert set(identified) == {"Lorenz-v0"} and len(identified) == 8
    assert again_rows == rows and again.shape == embedding.shape
    # the stored tensors of another sweep are not reused, nor those of another driver of the solver
    identified.clear()
    tiny_run(tmp_path, "--resume", "--seed0", "5")
    assert len(identified) == 32
    identified.clear()
    tiny_run(tmp_path, "--resume", "--seed0", "5", "--lstsq_driver", "gelss")
    assert len(identified) == 32


def test_peak_memory_is_in_megabytes_on_linux_and_macos(monkeypatch):
    class Usage:
        ru_maxrss = 3 * 1024**2  # 3 GB in kilobytes (Linux), 3 MB in bytes (macOS)

    monkeypatch.setattr(tsne.resource, "getrusage", lambda who: Usage)
    monkeypatch.setattr(tsne.sys, "platform", "linux")
    assert tsne.peak_memory_mb() == 3 * 1024
    monkeypatch.setattr(tsne.sys, "platform", "darwin")
    assert tsne.peak_memory_mb() == 3


def test_identification_and_embedding_can_be_separate_calls(tmp_path):
    # the two steps of the Snakemake workflow: one call with --identify_only per benchmark, which stores the tensors
    # of that benchmark and writes nothing else, then one call with --embed_only for all of them
    stages = tmp_path / "stages"
    for name in tsne.NAMES:
        _, (rows, embedding, scores) = tiny_run(stages, "--identify_only", "--environments", name)
        assert [row[0] for row in rows] == [name] * 8 and embedding is None and scores is None
    stored = sorted(tsne.tensors_path("", name) for name in tsne.NAMES)
    assert sorted(path.name for path in stages.iterdir()) == stored
    before = {name: (stages / name).read_bytes() for name in stored}
    _, (rows, embedding, _) = tiny_run(stages, "--embed_only")
    assert {name: (stages / name).read_bytes() for name in stored} == before

    # the benchmarks do not depend on each other: these are the rows, the tensors (to the rounding of the solver, see
    # test_tensors_of_a_small_sweep_are_reproducible_from_its_seeds) and the files of one call for all benchmarks
    _, (joint_rows, joint, _) = tiny_run(tmp_path / "joint")
    assert rows == joint_rows and embedding.shape == joint.shape
    files = sorted(path.name for path in stages.iterdir())
    assert files == sorted(path.name for path in (tmp_path / "joint").iterdir())
    for name in tsne.NAMES:
        tensors = [tsne.load_tensors(tsne.tensors_path(tmp_path / run, name))[1] for run in ("stages", "joint")]
        for a, b in zip(*tensors):
            assert np.allclose(a, b, rtol=0.0, atol=1e-6 * np.abs(a).max())

    # every file is an output of the rules of the workflow, so that Snakemake notices a missing one
    with open(os.path.join(REPOSITORY, "workflow", "rules", "tsne.smk")) as f:
        rules = f.read()
    for file in files:
        for name in tsne.NAMES:
            file = file.replace(f"tensors_{name}.", "tensors_{benchmark}.").replace(
                f"{name}_tsne.", "{benchmark}_tsne."
            )
        assert f'"{file}"' in rules

    with pytest.raises(ValueError):
        tiny_run(stages, "--identify_only", "--embed_only")


def test_the_settings_of_the_snakemake_workflow_are_arguments_of_the_step_they_are_passed_to():
    with open(os.path.join(REPOSITORY, "configurations", "tsne.json")) as f:
        settings = json.load(f)
    assert list(settings) == ["tsne"]
    settings = settings["tsne"]
    with open(os.path.join(REPOSITORY, "workflow", "rules", "tsne.smk")) as f:
        rules = f.read()
    set_by_rules = set(re.findall(r'"(\w+)"', re.search(r"TS_SET_BY_RULES = \((.*?)\)", rules, re.S).group(1)))
    defaults = tsne.ArgumentParser().parse_args([]).as_dict()
    assert set_by_rules == {"environments", "output_dir", "identify_only", "embed_only", "resume"} <= set(defaults)
    assert all(f"--{name}" in rules for name in set_by_rules - {"resume"})

    # `identification` lists the arguments that the identification reads, and no other: the workflow refuses these
    # under `embedding`, where they would repeat the embedding of the same tensors and change nothing
    identification = (tsne.sweep_grids, tsne.sweep_configurations, tsne.identify_sweep)
    read = set(re.findall(r"\bargs\.(\w+)", "".join(inspect.getsource(function) for function in identification)))
    assert read <= set(defaults)
    assert set(settings["identification"]) == read - set_by_rules
    # without settings the workflow makes the default sweep of the script, embedded with its defaults
    assert settings["identification"].pop("sweep") == defaults["sweep"]
    assert set(settings["identification"].values()) == {None}
    assert settings["embedding"] == {} and settings["only_benchmarks"] is None

    # the workflow reads the benchmarks from the file of the episodic returns and passes them in that order, which
    # is the order of the rows of tsne.csv and of the legend
    with open(os.path.join(REPOSITORY, "configurations", "episodic_returns.json")) as f:
        benchmarks = json.load(f)["episodic_returns"]["benchmarks"]
    assert list(benchmarks.items()) == [(env_id, entry["name"]) for env_id, entry in tsne.BENCHMARKS.items()]
    assert list(benchmarks.values()) == defaults["environments"]
