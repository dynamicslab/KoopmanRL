"""The ridge, SINDy and reduced-rank regressors of the Koopman tensor (koopmanrl/koopman_tensor/regressors.py)."""

import contextlib
import io

import numpy as np
import pytest
import torch

import koopmanrl.environments  # noqa: F401
from koopmanrl import soft_actor_koopman_critic, soft_koopman_value_iteration
from koopmanrl.koopman_tensor import numpy_tensor, regressors, torch_tensor
from koopmanrl.koopman_tensor.observables import numpy_observables, torch_observables

METHODS = ("ridge", "sindy", "rrr")


def regression(n=400, p=12, q=5, noise=0.0, seed=0):
    """Regressors whose columns differ in scale by three orders of magnitude, and targets that are the first q
    columns plus a small, sparse increment: the structure of the tensor regression."""
    rng = np.random.default_rng(seed)
    scales = np.logspace(0, 3, p)
    Z = rng.standard_normal((n, p)) * scales
    increment = np.zeros((p, q))
    rows = rng.choice(p, q + 1, replace=False)
    columns = np.append(np.arange(q), rng.integers(0, q))
    increment[rows, columns] = 0.05 * rng.choice([-1, 1], q + 1) * (1 + rng.random(q + 1)) * scales[0] / scales[rows]
    T = Z[:, :q] + Z @ increment + noise * rng.standard_normal((n, q))
    return Z, T, increment


def increment_coefficients(Z, T):
    B = np.linalg.lstsq(Z, T, rcond=None)[0]
    B[: T.shape[1]] -= np.eye(T.shape[1])
    return B


def transitions(env_id="DoubleWell-v0", seed=0, num_paths=20, steps=100):
    with contextlib.redirect_stdout(io.StringIO()):
        tensor = soft_koopman_value_iteration.generate_koopman_tensor(env_id, seed, num_paths, steps, 2, 2, "ols")
    return tensor.X, tensor.Y, tensor.U


def build(module, X, Y, U, regressor, state_order=2, action_order=2, **kwargs):
    if module is numpy_tensor:
        X, Y, U = X.numpy(), Y.numpy(), U.numpy()
        observables = numpy_observables
    else:
        observables = torch_observables
    with contextlib.redirect_stdout(io.StringIO()):
        return module.KoopmanTensor(
            X,
            Y,
            U,
            phi=observables.monomials(state_order),
            psi=observables.monomials(action_order),
            regressor=regressor,
            **kwargs,
        )


def relative_error(tensor, X, Y, U):
    """One-step error of phi(x') relative to persistence, on the given transitions."""
    phi = torch_observables.monomials(2)
    K = torch.as_tensor(np.asarray(tensor.K))
    predicted = torch.einsum("ijz,zn,jn->in", K, torch_observables.monomials(2)(U), phi(X))
    return float(((predicted - phi(Y)) ** 2).sum() / ((phi(X) - phi(Y)) ** 2).sum())


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #
def test_estimators_reduce_to_least_squares():
    Z, T, _ = regression(noise=1e-3)
    Zs = Z / np.sqrt((Z**2).mean(axis=0))
    reference = np.linalg.lstsq(Zs, T, rcond=None)[0]
    assert np.allclose(regressors.ridge(Zs, T, 0.0), reference, rtol=1e-8, atol=1e-10)
    assert np.allclose(regressors.sindy(Zs, T, 0.0), reference, rtol=1e-8, atol=1e-10)
    assert np.allclose(regressors.rrr(Zs, T, T.shape[1]), reference, rtol=1e-8, atol=1e-10)


def test_ridge_shrinks_and_rrr_truncates():
    Z, T, _ = regression(noise=1e-3)
    Zs = Z / np.sqrt((Z**2).mean(axis=0))
    norms = [np.linalg.norm(regressors.ridge(Zs, T, penalty)) for penalty in (0.0, 1.0, 100.0)]
    assert norms[0] > norms[1] > norms[2]
    for rank in (1, 2, 3):
        assert np.linalg.matrix_rank(regressors.rrr(Zs, T, rank), tol=1e-10) == rank


@pytest.mark.parametrize("method", METHODS)
def test_persistence_is_recovered_exactly_from_noise_free_data(method):
    Z, T, increment = regression()
    B, _ = regressors.fit(method, Z, T, persistence=True)
    assert np.allclose(B[: T.shape[1]] - np.eye(T.shape[1]), increment[: T.shape[1]], atol=1e-9)
    assert np.allclose(Z @ B, T, rtol=1e-8, atol=1e-8 * np.abs(T).max())


def test_sindy_finds_the_support_of_the_increment():
    Z, T, increment = regression(noise=1e-6)
    B, threshold = regressors.fit("sindy", Z, T, persistence=True)
    B[: T.shape[1]] -= np.eye(T.shape[1])
    assert threshold in regressors.THRESHOLDS
    assert np.array_equal(B != 0, increment != 0)
    # thresholding the raw coefficients, which are dominated by the identity, cannot do this
    assert np.count_nonzero(increment_coefficients(Z, T)) > np.count_nonzero(increment)


def test_hyperparameters_given_or_chosen():
    Z, T, _ = regression(noise=1e-3)
    for method in METHODS:
        _, chosen = regressors.fit(method, Z, T, persistence=True)
        assert chosen in regressors.grid(method, T.shape[1])
    assert regressors.fit("ridge", Z, T, parameter=0.5)[1] == 0.5
    B, rank = regressors.fit("rrr", Z, T, parameter=2, persistence=True)
    assert rank == 2
    B[: T.shape[1]] -= np.eye(T.shape[1])
    assert np.linalg.matrix_rank(B, tol=1e-12 * np.abs(B).max()) == 2
    assert regressors.grid("rrr", 20) == [5, 10, 15, 18, 20]
    assert regressors.grid("rrr", 2) == [1, 2]


def test_unknown_method_is_rejected():
    Z, T, _ = regression()
    with pytest.raises(ValueError, match="lasso"):
        regressors.fit("lasso", Z, T)


# --------------------------------------------------------------------------- #
# Tensor classes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("env_id", ["FluidFlow-v0", "DoubleWell-v0"])
@pytest.mark.parametrize("method", METHODS)
def test_regularised_tensors_are_close_to_least_squares(method, env_id):
    # Before the correction, SINDy and the rank-8 regression predicted no better than persistence on these
    # systems (relative errors 1.01 and 2.19 on the fluid flow, 0.96 for SINDy on the double well), and the ridge
    # decoder missed the state by up to 1.5e-3.
    X, Y, U = transitions(env_id)
    Xh, Yh, Uh = transitions(env_id, seed=1)
    ols = relative_error(build(torch_tensor, X, Y, U, "ols"), Xh, Yh, Uh)
    tensor = build(torch_tensor, X, Y, U, method)
    assert relative_error(tensor, Xh, Yh, Uh) < min(2 * ols, 0.5)
    # the decoder reads the state off the dictionary, which contains it
    assert torch.allclose(tensor.B.T @ tensor.Phi_X, X, atol=1e-8)


def test_four_copies_of_the_class_agree():
    X, Y, U = transitions(num_paths=10)
    copies = (torch_tensor, numpy_tensor, soft_koopman_value_iteration, soft_actor_koopman_critic)
    for method in METHODS:
        tensors = [build(module, X, Y, U, method) for module in copies]
        reference = tensors[0].K.numpy()
        for tensor in tensors[1:]:
            assert np.allclose(np.asarray(tensor.K), reference, rtol=1e-8, atol=1e-10 * np.abs(reference).max())
            assert tensor.regressor_parameter == tensors[0].regressor_parameter


def test_hyperparameters_reach_the_regression():
    X, Y, U = transitions(num_paths=10)
    assert build(torch_tensor, X, Y, U, "ridge", penalty=0.25).regressor_parameter == 0.25
    assert build(torch_tensor, X, Y, U, "sindy", threshold=0.02).regressor_parameter == 0.02
    tensor = build(torch_tensor, X, Y, U, "rrr", rank=2)
    increment = tensor.M.numpy().copy()
    increment[:, : tensor.phi_dim] -= np.eye(tensor.phi_dim)
    assert np.linalg.matrix_rank(increment, tol=1e-10 * np.abs(increment).max()) == 2


def test_ordinary_least_squares_is_unchanged():
    X, Y, U = transitions(num_paths=10)
    tensor = build(torch_tensor, X, Y, U, "ols")
    assert torch.equal(tensor.M, torch.linalg.lstsq(tensor.kron_matrix.T, tensor.regression_Y.T).solution.T)
    assert not hasattr(tensor, "regressor_parameter")
    numpy_ols = build(numpy_tensor, X, Y, U, "ols")
    assert np.allclose(numpy_ols.K, tensor.K.numpy(), rtol=1e-8, atol=1e-10 * np.abs(numpy_ols.K).max())
