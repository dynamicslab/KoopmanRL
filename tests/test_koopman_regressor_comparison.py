"""koopmanrl_utils.koopman_regressor_comparison builds the package's tensor, and its estimators, error and outputs
are what the module says they are."""

import contextlib
import io

import gym
import numpy as np
import pytest
import torch

import koopmanrl.environments  # noqa: F401
import koopmanrl_utils.koopman_regressor_comparison as comparison
from koopmanrl.koopman_tensor.observables.torch_observables import monomials
from koopmanrl.koopman_tensor.torch_tensor import KoopmanTensor, Regressor
from koopmanrl_utils.skvi_sensitivity_checks import double_well_conditional_mean


def small_data(env_id, seed=0, num_paths=6, steps=40):
    return comparison.transitions(comparison.random_agent_paths(env_id, seed, num_paths, steps))


def regression_problem(noise=0.0, seed=0, n=400, m=6, p=3):
    """Well-conditioned regressors Z, sparse coefficients B and targets T = Z B + noise."""
    rng = np.random.RandomState(seed)
    Z = rng.randn(n, m)
    B = np.zeros((m, p))
    B[0, 0], B[2, 1], B[4, 2], B[5, 0] = 1.0, -2.0, 0.5, 3.0
    return Z, B, Z @ B + noise * rng.randn(n, p)


def test_every_regressor_is_registered_once():
    assert len(comparison.REGRESSORS) == 13
    assert set(comparison.SHIPPED) | set(comparison.ESTIMATORS) == set(comparison.REGRESSORS)
    assert not set(comparison.SHIPPED) & set(comparison.ESTIMATORS)
    families = {key: entry[2] for key, entry in comparison.REGRESSORS.items()}
    assert [key for key, family in families.items() if family == "paper"] == ["ols"]
    assert all(families[key] == "alternative" for key in comparison.ESTIMATORS)
    assert set(comparison.BUDGET_STYLE) <= set(comparison.REGRESSORS)


@pytest.mark.parametrize("env_id", ["LinearSystem-v0", "FluidFlow-v0", "DoubleWell-v0"])
def test_scaled_least_squares_is_the_tensor_of_the_package(env_id):
    # the alternatives use their own regressor matrix, increment target and unfolding: with plain least squares
    # they must give the tensor that the package's class builds
    X, U, Y = small_data(env_id)
    packaged, _ = comparison.fit_tensor(X, U, Y, 2, 2, "ols")
    scaled, hyper = comparison.fit_tensor(X, U, Y, 2, 2, "ols_scaled")
    assert hyper is None
    assert packaged.shape == scaled.shape == (packaged.shape[0], packaged.shape[0], 3)
    assert np.allclose(packaged, scaled, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("regressor", ["ols", "ridge", "sindy", "rrr"])
def test_shipped_regressors_run_through_the_class_of_the_package(regressor):
    X, U, Y = small_data("FluidFlow-v0")
    with contextlib.redirect_stdout(io.StringIO()):
        tensor = KoopmanTensor(
            torch.tensor(X.T),
            torch.tensor(Y.T),
            torch.tensor(U.T),
            phi=monomials(2),
            psi=monomials(1),
            regressor=Regressor(regressor),
        )
    K, _ = comparison.fit_tensor(X, U, Y, 2, 1, regressor)
    assert np.allclose(K, tensor.K.numpy())


def test_numpy_copy_agrees_with_least_squares_on_a_well_posed_problem():
    X, U, Y = small_data("DoubleWell-v0")
    packaged, _ = comparison.fit_tensor(X, U, Y, 2, 1, "ols")
    numpy_copy, _ = comparison.fit_tensor(X, U, Y, 2, 1, "ols_numpy")
    assert np.allclose(packaged, numpy_copy, atol=1e-6)


def test_estimators_reduce_to_least_squares_without_regularisation():
    Z, _, T = regression_problem(noise=0.1)
    reference = np.linalg.lstsq(Z, T, rcond=None)[0]
    assert np.allclose(comparison.least_squares(Z, T), reference)
    assert np.allclose(comparison.ridge(Z, T, 0.0), reference)
    assert np.allclose(comparison.truncated_svd(Z, T, 0.0), reference)
    assert np.allclose(comparison.reduced_rank(Z, T, 1.0), reference)
    assert np.allclose(comparison.thresholded_least_squares(Z, T, 0.0), reference)
    assert np.allclose(comparison.lasso(Z, T, 1e-8), reference, atol=1e-5)


def test_estimators_regularise_as_named():
    Z, B, T = regression_problem(noise=0.01)
    reference = np.linalg.lstsq(Z, T, rcond=None)[0]
    assert np.linalg.norm(comparison.ridge(Z, T, 10.0)) < np.linalg.norm(reference)
    assert np.linalg.matrix_rank(comparison.reduced_rank(Z, T, 0.3)) == 1
    assert np.all(comparison.lasso(Z, T, 1e3) == 0.0)
    assert np.all(comparison.thresholded_least_squares(Z, T, 1e3) == 0.0)
    # thresholding and the l1 penalty recover the support of the sparse coefficients
    assert np.array_equal(comparison.thresholded_least_squares(Z, T, 0.1) != 0.0, B != 0.0)
    assert np.array_equal(np.abs(comparison.lasso(Z, T, 0.02)) > 1e-3, B != 0.0)
    # a cut-off above the smallest direction of the regressors removes what least squares puts on it
    scaled = Z * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1e-5])
    kept, dropped = comparison.truncated_svd(scaled, T, 0.0), comparison.truncated_svd(scaled, T, 1e-6)
    assert np.isclose(kept[5, 0], 3e5, rtol=1e-2) and np.abs(dropped[5]).max() < 1e-3


def test_total_least_squares_is_exact_without_noise_and_ignores_empty_outputs():
    Z, B, T = regression_problem()
    assert np.allclose(comparison.total_least_squares(Z, T), B)
    T[:, 1] = 0.0
    estimate = comparison.total_least_squares(Z, T)
    assert np.all(estimate[:, 1] == 0.0) and np.allclose(estimate[:, [0, 2]], B[:, [0, 2]])


def test_huber_resists_gross_outliers():
    Z, B, T = regression_problem(noise=0.01)
    assert np.allclose(comparison.huber(Z, T), np.linalg.lstsq(Z, T, rcond=None)[0], atol=1e-3)
    T[:8] += 50.0
    robust = np.abs(comparison.huber(Z, T) - B).max()
    least_squares = np.abs(np.linalg.lstsq(Z, T, rcond=None)[0] - B).max()
    assert robust < 0.1 * least_squares


def test_increment_fit_returns_original_units_and_a_grid_value():
    Z, B, T = regression_problem(noise=0.05)
    scales = np.array([1.0, 1e3, 1e-3, 10.0, 1.0, 1e2])
    estimate, hyper = comparison.fit_increment("ols_scaled", Z * scales, 7.0 * T)
    assert hyper is None
    assert np.allclose(estimate, 7.0 * np.linalg.lstsq(Z, T, rcond=None)[0] / scales[:, None])
    for name in ("tsvd", "ridge_cv", "stlsq_cv", "lasso_cv", "rrr_cv"):
        estimate, hyper = comparison.fit_increment(name, Z * scales, 7.0 * T)
        assert hyper in comparison.REGRESSORS[name][3]
        assert estimate.shape == B.shape and np.all(np.isfinite(estimate))


def test_one_step_error_is_relative_to_the_scale_of_each_function():
    X, U, Y = small_data("LinearSystem-v0")
    target = comparison.lift(Y, 2)
    exact, _ = comparison.fit_tensor(X, U, Y, 2, 2, "ols")
    error, persistence = comparison.one_step_error(exact, X, U, target, 2, 2)
    assert error < 1e-10 and persistence > 1e-2
    # the tensor of "no change" has the error of the persistence predictor
    identity = np.zeros_like(exact)
    identity[:, :, 0] = np.eye(exact.shape[0])
    error, persistence = comparison.one_step_error(identity, X, U, target, 2, 2)
    assert np.isclose(error, persistence)


def test_double_well_conditional_mean_is_the_exact_one():
    rng = np.random.RandomState(0)
    X, U = rng.uniform(-2, 2, (30, 2)), rng.uniform(-25, 25, (30, 1))
    with contextlib.redirect_stdout(io.StringIO()):
        env = gym.make("DoubleWell-v0")
    reference, _ = double_well_conditional_mean(env, X, U[:, 0], comparison.powers(2, 2))
    assert np.allclose(comparison.conditional_mean_lifted("DoubleWell-v0", X, U, None, 2), reference.T)
    with pytest.raises(ValueError):
        comparison.conditional_mean_lifted("DoubleWell-v0", X, U, None, 3)


def test_deterministic_target_is_the_dictionary_of_the_next_state():
    X, U, Y = small_data("Lorenz-v0", num_paths=2, steps=10)
    assert np.array_equal(comparison.conditional_mean_lifted("Lorenz-v0", X, U, Y, 3), comparison.lift(Y, 3))


def test_seeds_of_the_linear_system_are_data_sets_of_one_system():
    first, _ = comparison.fit_tensor(*small_data("LinearSystem-v0", seed=0), 1, 1, "ols")
    second, _ = comparison.fit_tensor(*small_data("LinearSystem-v0", seed=1), 1, 1, "ols")
    assert not np.array_equal(small_data("LinearSystem-v0", seed=0)[0], small_data("LinearSystem-v0", seed=1)[0])
    assert np.allclose(first, second, atol=1e-9)


def test_budgets_are_whole_trajectories_without_repeats():
    fractions = comparison.ArgumentParser().parse_args([]).fractions
    assert comparison.budget_paths(75, fractions) == [2, 4, 8, 15, 38, 75]
    assert comparison.budget_paths(150, fractions) == [2, 3, 8, 15, 30, 75, 150]


def test_double_well_reproduces_the_reported_errors():
    # seed 0 at the tuned settings: the values behind the first seed of the accuracy study
    env_id, cfg = "DoubleWell-v0", comparison.CONFIGS["DoubleWell-v0"]
    Xh, Uh, target = comparison.held_out(env_id)
    data = comparison.transitions(comparison.random_agent_paths(env_id, 0, cfg["num_paths"], cfg["steps"]))
    for regressor, reported in (("ols", 0.171137), ("sindy", 0.982233), ("huber", 0.160623)):
        K, _ = comparison.fit_tensor(*data, cfg["so"], cfg["ao"], regressor)
        error, persistence = comparison.one_step_error(K, Xh, Uh, target, cfg["so"], cfg["ao"])
        assert np.isclose(error / persistence, reported, rtol=1e-4)


def test_numbers_for_logarithmic_axes():
    assert comparison.number(0.0) == "1.000000e-17"
    assert comparison.number(float("inf")) == "nan"
    assert comparison.median([1.0, float("nan"), 3.0]) == 3.0
    assert comparison.fixed_point_ticks(0.16, 3.0) == ["0.2", "0.5", "1", "2"]
    assert comparison.fixed_point_ticks(0.012, 3.0) == ["0.02", "0.1", "1"]


def test_tables_and_figures_are_written(tmp_path):
    regressors = ["ols", "sindy", "lasso_cv"]
    errors = {"ols": [0.2, 0.3], "sindy": [5.0, 7.0], "lasso_cv": [0.25, 0.2]}
    accuracy = dict(seeds=2, results={"Lorenz": {r: dict(relative_error=errors[r]) for r in regressors}})
    budget = dict(
        seeds=2,
        results={"Lorenz": dict(transitions=[500, 1000], regressors={r: [errors[r], errors[r]] for r in regressors})},
    )
    with contextlib.redirect_stdout(io.StringIO()):
        comparison.write_accuracy(accuracy, str(tmp_path))
        comparison.write_budget(budget, str(tmp_path))

    rows = (tmp_path / "accuracy_Lorenz.dat").read_text().splitlines()
    assert rows[0] == "row regressor family median minimum maximum"
    assert rows[1].split()[:4] == ["1", "ols", "paper", "2.500000e-01"]
    # a median beyond the axis is drawn as an arrow with its value, and its seeds are left out
    assert (tmp_path / "accuracy_offscale_Lorenz.dat").read_text().splitlines()[1] == "2 shipped 3.000000e+00 6"
    assert len((tmp_path / "accuracy_seeds_Lorenz.dat").read_text().splitlines()) == 1 + 4
    assert (tmp_path / "budget_Lorenz.dat").read_text().splitlines()[0] == "transitions ols sindy lasso_cv"
    assert (tmp_path / "accuracy_table.csv").read_text().splitlines()[0] == "regressor,Lorenz"

    accuracy_figure = (tmp_path / "regressor_accuracy.tikz").read_text()
    budget_figure = (tmp_path / "regressor_budget.tikz").read_text()
    for name in ("accuracy_Lorenz.dat", "accuracy_seeds_Lorenz.dat", "accuracy_offscale_Lorenz.dat"):
        assert f"{{{name}}}" in accuracy_figure
    assert budget_figure.count("{budget_Lorenz.dat}") == 3
    assert accuracy_figure.count(r"\begin{tikzpicture}") == accuracy_figure.count(r"\end{tikzpicture}") == 1
    assert "regressor_budget.tikz" in (tmp_path / "regressor_budget.tex").read_text()
