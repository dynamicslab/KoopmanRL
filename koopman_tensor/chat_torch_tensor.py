""" THIS CODE WAS WRITTEN BY CHATGPT """


import torch

def checkMatrixRank(X, name):
    rank = torch.linalg.matrix_rank(X)
    print(f"{name} matrix rank: {rank}")
    if rank != X.shape[0]:
        pass

def checkConditionNumber(X, name, threshold=200):
    cond_num = torch.linalg.cond(X)
    print(f"Condition number of {name}: {cond_num}")
    if cond_num > threshold:
        pass

def SINDy(Theta, dXdt, lamb=0.05):
    d = dXdt.shape[1]
    Xi = torch.lstsq(dXdt, Theta, rcond=None)[0]

    for _ in range(10):
        smallinds = torch.abs(Xi) < lamb
        Xi[smallinds] = 0
        for ind in range(d):
            biginds = smallinds[:, ind] == 0
            Xi[biginds, ind], _ = torch.lstsq(dXdt[:, ind], Theta[:, biginds], rcond=None)

    L = Xi
    return L

def ols(X, Y, pinv=True):
    if pinv:
        return torch.linalg.pinv(X.T @ X) @ X.T @ Y
    return torch.linalg.inv(X.T @ X) @ X.T @ Y

def rrr(X, Y, rank=8):
    B_ols = ols(X, Y)
    U, S, V = torch.linalg.svd(Y.T @ X @ B_ols)
    W = V[:, :rank]

    B_rr = B_ols @ W @ W.T
    L = B_rr
    return L

def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

""" Koopman Tensor """

class KoopmanTensor:
    def __init__(
        self,
        X,
        Y,
        U,
        phi,
        psi,
        regressor='ols',
        p_inv=True,
        rank=8,
        is_generator=False,
        dt=0.01
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
        regressor : {'ols', 'sindy', 'rrr'}, optional
            String indicating the regression method to use. Default is 'ols'.
        p_inv : bool, optional
            Boolean indicating whether to use pseudo-inverse instead of regular inverse. Default is True.
        rank : int, optional
            Rank of the Koopman tensor when applying reduced rank regression. Default is 8.
        is_generator : bool, optional
            Boolean indicating whether the model is a Koopman generator tensor. Default is False.
        dt : float, optional
            The time step of the system. Default is 0.01.

        Returns
        -------
        KoopmanTensor
            An instance of the KoopmanTensor class.
        """

        self.X = X
        self.Y = Y
        self.U = U
        self.phi = phi
        self.psi = psi
        self.N = self.X.shape[1]
        self.Phi_X = self.phi(X)
        self.Phi_Y = self.phi(Y)
        self.Psi_U = self.psi(U)
        self.x_dim = self.X.shape[0]
        self.u_dim = self.U.shape[0]
        self.phi_dim = self.Phi_X.shape[0]
        self.psi_dim = self.Psi_U.shape[0]
        self.x_column_dim = [self.x_dim, 1]
        self.u_column_dim = [self.u_dim, 1]
        self.phi_column_dim = [self.phi_dim, 1]

        if is_generator:
            self.dt = dt
            finite_differences = (self.Y - self.X)
            phi_derivative = self.phi.diff(self.X)
            phi_double_derivative = self.phi.ddiff(self.X)
            self.regression_Y = (finite_differences / self.dt) @ phi_derivative
            self.regression_Y += (0.5 * (finite_differences @ finite_differences.T) / self.dt) @ phi_double_derivative
        else:
            self.regression_Y = self.Phi_Y

        checkMatrixRank(self.Phi_X, "Phi_X")
        checkMatrixRank(self.regression_Y, "dPhi_Y" if is_generator else "Phi_Y")
        checkMatrixRank(self.Psi_U, "Psi_U")

        checkConditionNumber(self.Phi_X, "Phi_X")
        checkConditionNumber(self.regression_Y, "dPhi_Y" if is_generator else "Phi_Y")
        checkConditionNumber(self.Psi_U, "Psi_U")

        self.kron_matrix = torch.empty([self.psi_dim * self.phi_dim, self.N])
        for i in range(self.N):
            self.kron_matrix[:, i] = torch.kron(self.Psi_U[:, i], self.Phi_X[:, i])

        if regressor.lower() == 'rrr':
            self.M = rrr(self.kron_matrix.T, self.regression_Y.T, rank).T
            self.B = rrr(self.Phi_X.T, self.X.T, rank)
        elif regressor.lower() == 'sindy':
            self.M = SINDy(self.kron_matrix.T, self.regression_Y.T).T
            self.B = SINDy(self.Phi_X.T, self.X.T)
        elif regressor.lower() == 'ols':
            self.M = ols(self.kron_matrix.T, self.regression_Y.T, p_inv).T
            self.B = ols(self.Phi_X.T, self.X.T, p_inv)

        self.K = torch.empty([self.phi_dim, self.phi_dim, self.psi_dim])
        for i in range(self.phi_dim):
            # self.K[i] = self.M[i].reshape([self.phi_dim, self.psi_dim], order='F')
            self.K[i] = reshape_fortran(self.M[i], (self.phi_dim, self.psi_dim))

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

        torch_K = torch.Tensor(self.K)
        K_u = torch.einsum('ijz,zk->kij', torch_K, self.psi(u))
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

        return self.K_(u) @ self.phi(x)

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