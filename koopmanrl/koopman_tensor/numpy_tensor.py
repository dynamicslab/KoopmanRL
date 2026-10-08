from enum import Enum

import numpy as np

from koopmanrl.koopman_tensor.regressors import tensor_regression

""" Helper functions """


def checkMatrixRank(X, name):
    rank = np.linalg.matrix_rank(X)
    print(f"{name} matrix rank: {rank}")
    if rank != X.shape[0]:
        # raise ValueError(f"{name} matrix is not full rank ({rank} / {X.shape[0]})")
        pass


def checkConditionNumber(X, name, threshold=200):
    cond_num = np.linalg.cond(X)
    print(f"Condition number of {name}: {cond_num}")
    if cond_num > threshold:
        # raise ValueError(f"Condition number of {name} is too large ({cond_num} > {threshold})")
        pass


def ols(X, Y, pinv=True):
    if pinv:  # minimum-norm least squares by an orthogonal factorisation, as in the torch class
        return np.linalg.lstsq(X, Y, rcond=None)[0]
    return np.linalg.solve(X.T @ X, X.T @ Y)


def OLS(X, Y, pinv=True):
    return ols(X, Y, pinv)


""" Regressor enum """


class Regressor(str, Enum):
    OLS = "ols"
    RRR = "rrr"
    SINDy = "sindy"
    RIDGE = "ridge"


""" Koopman Tensor """


class KoopmanTensor:
    def __init__(
        self,
        X,
        Y,
        U,
        phi,
        psi,
        regressor=Regressor.OLS,
        p_inv=True,
        rank=None,
        is_generator=False,
        dt=0.01,
        penalty=None,
        threshold=None,
    ):
        """
        Create an instance of the KoopmanTensor class.

        Parameters
        ----------
        X : array_like
            States dataset used for training.
        Y : array_like
            Single-step forward states dataset used for training.
        U : array_like
            Actions dataset used for training.
        phi : callable
            Dictionary space representing the states.
        psi : callable
            Dictionary space representing the actions.
        regressor : {'ols', 'ridge', 'sindy', 'rrr'}, optional
            Regression method. Default is 'ols'. 'ridge', 'sindy' and 'rrr' are solved for the standardised
            increment phi(x') - phi(x) (see koopmanrl/koopman_tensor/regressors.py).
        p_inv : bool, optional
            Whether 'ols' returns the minimum-norm least-squares solution instead of solving the normal
            equations. Default is True.
        rank : int, optional
            Rank of the regression for 'rrr'. Default is None: chosen on the last 20% of the transitions.
        is_generator : bool, optional
            Boolean indicating whether the model is a Koopman generator tensor. Default is False.
        dt : float, optional
            The time step of the system. Default is 0.01.
        penalty : float, optional
            Penalty for 'ridge', relative to the standardised Gram matrix. Default is None: chosen on the last
            20% of the transitions.
        threshold : float, optional
            Threshold for 'sindy' on the standardised coefficients. Default is None: chosen on the last 20% of
            the transitions.

        Returns
        -------
        KoopmanTensor
            An instance of the KoopmanTensor class.
        """

        # Save datasets
        self.X = X
        self.Y = Y
        self.U = U

        # Extract dictionary/observable functions
        self.phi = phi
        self.psi = psi

        # Get number of data points
        self.N = self.X.shape[1]

        # Construct Phi and Psi matrices
        self.Phi_X = self.phi(X)
        self.Phi_Y = self.phi(Y)
        self.Psi_U = self.psi(U)

        # Get dimensions
        self.x_dim = self.X.shape[0]
        self.u_dim = self.U.shape[0]
        self.phi_dim = self.Phi_X.shape[0]
        self.psi_dim = self.Psi_U.shape[0]
        self.x_column_dim = (self.x_dim, 1)
        self.u_column_dim = (self.u_dim, 1)
        self.phi_column_dim = (self.phi_dim, 1)

        # Update regression matrices if dealing with Koopman generator
        if is_generator:
            # Save delta time
            self.dt = dt

            # Update regression target
            finite_differences = self.Y - self.X  # (self.x_dim, self.N)
            phi_derivative = self.phi.diff(self.X)  # (self.phi_dim, self.x_dim, self.N)
            phi_double_derivative = self.phi.ddiff(self.X)  # (self.phi_dim, self.x_dim, self.x_dim, self.N)
            self.regression_Y = np.einsum("os,pos->ps", finite_differences / self.dt, phi_derivative)
            self.regression_Y += np.einsum(
                "ot,pots->ps",
                0.5 * (finite_differences @ finite_differences.T) / self.dt,  # (state_dim, state_dim)
                phi_double_derivative,
            )
        else:
            # Set regression target to phi(Y)
            self.regression_Y = self.Phi_Y

        # Make sure data is full rank
        checkMatrixRank(self.Phi_X, "Phi_X")
        checkMatrixRank(self.regression_Y, "dPhi_Y" if is_generator else "Phi_Y")
        checkMatrixRank(self.Psi_U, "Psi_U")
        print("\n")

        # Make sure condition numbers are small
        checkConditionNumber(self.Phi_X, "Phi_X")
        checkConditionNumber(self.regression_Y, "dPhi_Y" if is_generator else "Phi_Y")
        checkConditionNumber(self.Psi_U, "Psi_U")
        print("\n")

        # Build matrix of kronecker products between u_i and x_i for all 0 <= i <= N
        self.kron_matrix = np.empty([self.psi_dim * self.phi_dim, self.N])
        for i in range(self.N):
            self.kron_matrix[:, i] = np.kron(self.Psi_U[:, i], self.Phi_X[:, i])

        # Solve for M and B
        lowercase_regressor = regressor.lower()
        if lowercase_regressor == Regressor.OLS:
            self.M = ols(self.kron_matrix.T, self.regression_Y.T, p_inv).T
            self.B = ols(self.Phi_X.T, self.X.T, p_inv)
        elif lowercase_regressor in (Regressor.RIDGE, Regressor.SINDy, Regressor.RRR):
            # With psi_0(u) = 1 the first block of kron_matrix is phi(x), and the regularised regressors act on
            # the departure from persistence, phi(x') - phi(x).
            persistence = not is_generator and bool(np.all(self.Psi_U[0] == 1))
            M, B, self.regressor_parameter = tensor_regression(
                lowercase_regressor,
                self.kron_matrix,
                self.regression_Y,
                self.Phi_X,
                self.X,
                rank=rank,
                penalty=penalty,
                threshold=threshold,
                persistence=persistence,
            )
            self.M = M
            self.B = B
        else:
            raise Exception("Did not pick a supported regression algorithm.")

        # reshape M into tensor K
        self.K = np.empty([self.phi_dim, self.phi_dim, self.psi_dim])
        for i in range(self.phi_dim):
            self.K[i] = self.M[i].reshape((self.phi_dim, self.psi_dim), order="F")

    def K_(self, u):
        """
        Compute the Koopman operator associated with a given action.

        Parameters
        ----------
        u : array_like
            Action for which the Koopman operator is computed.

        Returns
        -------
        ndarray
            Koopman operator corresponding to the given action.
        """

        K_u = np.einsum("ijz,zk->kij", self.K, self.psi(u))

        if K_u.shape[0] == 1:
            return K_u[0]

        return K_u

    def phi_f(self, x, u):
        """
        Apply the Koopman tensor to push forward phi(x) x psi(u) to phi(x').

        Parameters
        ----------
        x : array_like
            State column vector(s).
        u : array_like
            Action column vector(s).

        Returns
        -------
        ndarray
            Transformed phi(x') column vector(s).
        """

        # return self.K_(u) @ self.phi(x)

        K_us = self.K_(u)  # (batch_size, phi_dim, phi_dim)
        phi_x = self.phi(x)  # (phi_dim, batch_size)

        if len(K_us.shape) == 2:
            return K_us @ phi_x

        phi_x_primes = (K_us @ phi_x.T[:, :, np.newaxis]).squeeze()
        return phi_x_primes.T

    def f(self, x, u):
        """
        Utilize the Koopman tensor to approximate the true dynamics f(x, u) and predict x'.

        Parameters
        ----------
        x : array_like
            State column vector(s).
        u : array_like
            Action column vector(s).

        Returns
        -------
        ndarray
            Predicted state column vector(s).
        """

        return self.B.T @ self.phi_f(x, u)
