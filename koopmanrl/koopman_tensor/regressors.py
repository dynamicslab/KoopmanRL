"""
Ridge, sequentially thresholded least squares (SINDy) and reduced-rank regression for the Koopman tensor.

The tensor is identified from the regression  phi(x') ~ M kron(psi(u), phi(x)),  with one column per transition, and
the decoder from  x ~ B^T phi(x).  Ordinary least squares solves these regressions directly. The three regularised
estimators here cannot be applied to them in that form, for two reasons:

* the identity dominates M. With a time step of 0.01, M is close to [I 0 ... 0], and every coefficient that carries
  the dynamics is small. A threshold or penalty on the raw coefficients removes the dynamics and keeps persistence,
  and the leading directions of the fitted phi(x') are those of the largest monomials, not of the state;
* the columns of the regression differ in scale by many orders of magnitude, so a fixed penalty, threshold or rank
  means something different on every system.

Both are removed by two changes of coordinates, which leave ordinary least squares unchanged:

* increment target. The regression target is phi(x') - phi(x). Since psi_0(u) = 1, phi(x) is the first block of the
  regressors, so  M = M_inc + [I 0 ... 0]  and the penalty, threshold or rank acts on the departure from persistence;
* standardisation. Every regressor column and every target column is divided by its root-mean-square value, so that
  the penalty and the threshold are dimensionless.

The hyperparameter (ridge penalty, SINDy threshold, rank) is either given or chosen from a short grid by fitting on
the first 80% of the transitions and scoring the squared error on the last 20%, after which the model is refitted on
all transitions. The data of the package are stored path by path, so the last 20% of the transitions are the last 20%
of the paths.
"""

import numpy as np

VALIDATION_FRACTION = 0.2  # share of the transitions, taken from the end, on which the hyperparameter is chosen
PENALTIES = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)  # ridge penalties, relative to the standardised Gram matrix
THRESHOLDS = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1)  # SINDy thresholds on the standardised coefficients
RANK_FRACTIONS = (0.25, 0.5, 0.75, 0.9, 1.0)  # ranks of the reduced-rank regression, as fractions of the outputs
ITERATIONS = 10  # threshold-and-refit rounds of SINDy, as in Brunton, Proctor and Kutz (2016)

METHODS = ("ridge", "sindy", "rrr")


def _solve(G, C, penalty=0.0):
    """(G + penalty I)^+ C for a symmetric positive semi-definite G, through its eigendecomposition."""
    w, V = np.linalg.eigh(G)
    keep = w > 1e-300
    inverse = np.zeros_like(w)
    inverse[keep] = 1.0 / (w[keep] + penalty)
    return (V * inverse) @ (V.T @ C)


def _gram(Z, T):
    return Z.T @ Z / len(Z), Z.T @ T / len(Z)


def ridge(Z, T, penalty):
    """Tikhonov-regularised least squares: argmin ||Z B - T||^2 / n + penalty ||B||^2."""
    G, C = _gram(Z, T)
    return _solve(G, C, penalty)


def sindy(Z, T, threshold, iterations=ITERATIONS):
    """Sequentially thresholded least squares: zero the coefficients below the threshold and refit, repeatedly."""
    G, C = _gram(Z, T)
    B = _solve(G, C)
    for _ in range(iterations):
        small = np.abs(B) < threshold
        B[small] = 0.0
        for j in range(B.shape[1]):
            big = ~small[:, j]
            if big.any():
                B[big, j] = np.linalg.lstsq(G[np.ix_(big, big)], C[big, j], rcond=None)[0]
    return B


def rrr(Z, T, rank):
    """Reduced-rank regression: least squares projected onto the leading `rank` directions of the fitted values."""
    G, C = _gram(Z, T)
    B = _solve(G, C)
    w, V = np.linalg.eigh(B.T @ G @ B)
    leading = V[:, np.argsort(w)[::-1][: max(1, min(int(rank), T.shape[1]))]]
    return B @ leading @ leading.T


ESTIMATORS = {"ridge": ridge, "sindy": sindy, "rrr": rrr}


def grid(method, num_outputs):
    """Candidate hyperparameters of a method for a regression with `num_outputs` targets."""
    if method == "ridge":
        return list(PENALTIES)
    if method == "sindy":
        return list(THRESHOLDS)
    return sorted({max(1, int(round(f * num_outputs))) for f in RANK_FRACTIONS})


def fit(method, Z, T, parameter=None, persistence=False):
    """
    Coefficients B of the regression T ~ Z B with a regularised estimator, in the original units.

    Parameters
    ----------
    method : {'ridge', 'sindy', 'rrr'}
        Estimator.
    Z : array_like
        Regressors, one row per transition, shape (N, p).
    T : array_like
        Targets, one row per transition, shape (N, q).
    parameter : float or int, optional
        Ridge penalty or SINDy threshold, both in standardised units, or rank. None chooses it on the last
        VALIDATION_FRACTION of the transitions.
    persistence : bool, optional
        Whether the first q columns of Z are the targets' values at the current step (phi(x) for the tensor). The
        regression is then solved for the increment T - Z[:, :q], and the identity is added back. Default is False.

    Returns
    -------
    B : ndarray
        Coefficients, shape (p, q).
    parameter : float or int
        Hyperparameter used.
    """
    method = getattr(method, "value", method).lower()
    if method not in ESTIMATORS:
        raise ValueError(f"Unknown regularised regressor '{method}'; expected one of {METHODS}.")
    Z = np.asarray(Z, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    q = T.shape[1]
    if persistence:
        T = T - Z[:, :q]

    z_scale = np.sqrt((Z**2).mean(axis=0))
    t_scale = np.sqrt((T**2).mean(axis=0))
    z_scale[z_scale == 0] = 1.0
    t_scale[t_scale == 0] = 1.0
    Zs, Ts = Z / z_scale, T / t_scale

    estimator = ESTIMATORS[method]
    if parameter is None:
        cut = int((1.0 - VALIDATION_FRACTION) * len(Zs))
        candidates = grid(method, q)
        errors = []
        for candidate in candidates:
            B = estimator(Zs[:cut], Ts[:cut], candidate)
            errors.append(float(((Zs[cut:] @ B - Ts[cut:]) ** 2).mean()))
        parameter = candidates[int(np.argmin(np.nan_to_num(errors, nan=np.inf)))]

    B = estimator(Zs, Ts, parameter) * t_scale[None, :] / z_scale[:, None]
    if persistence:
        B[:q] += np.eye(q)
    return B, parameter


def tensor_regression(
    method, kron_matrix, regression_Y, Phi_X, X, rank=None, penalty=None, threshold=None, persistence=False
):
    """
    Regression matrix M and decoder B of a Koopman tensor with a regularised estimator.

    Parameters
    ----------
    method : {'ridge', 'sindy', 'rrr'}
        Estimator.
    kron_matrix : array_like
        Columns kron(psi(u_i), phi(x_i)), shape (psi_dim * phi_dim, N).
    regression_Y : array_like
        Regression target, phi(x'_i) or the generator applied to phi, shape (phi_dim, N).
    Phi_X : array_like
        Columns phi(x_i), shape (phi_dim, N).
    X : array_like
        States, shape (x_dim, N).
    rank, penalty, threshold : optional
        Hyperparameter of M for 'rrr', 'ridge' and 'sindy'; None chooses it on held-out transitions. The decoder's
        hyperparameter is always chosen on held-out transitions.
    persistence : bool, optional
        Whether regression_Y is phi(x') and the first phi_dim rows of kron_matrix are phi(x), which holds when
        psi_0(u) = 1. Default is False.

    Returns
    -------
    M : ndarray
        Shape (phi_dim, psi_dim * phi_dim).
    B : ndarray
        Decoder, shape (phi_dim, x_dim).
    parameter : float or int
        Hyperparameter used for M.
    """
    method = getattr(method, "value", method).lower()
    parameter = {"ridge": penalty, "sindy": threshold, "rrr": rank}.get(method)
    M, parameter = fit(method, np.asarray(kron_matrix).T, np.asarray(regression_Y).T, parameter, persistence)
    B, _ = fit(method, np.asarray(Phi_X).T, np.asarray(X).T)
    return M.T, B, parameter
