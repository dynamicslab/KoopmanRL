"""
Example usage:

python -m cleanrl.value_based_sac_continuous_action --env-id=FluidFlow-v0 --alpha=1 --autotune=false --total-timesteps=50000 --
"""

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool
from enum import Enum

import gym
import numpy as np
import torch

torch.set_default_dtype(torch.float64)  # TODO: Test if this is truly necessary
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from analysis.utils import create_folder
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from cleanrl_utils.koopman_observables import monomials

from custom_envs import *


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment (default: 1)")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False` (default: True)")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default (default: True)")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases (default: False)")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name (default: \"cleanRL\")")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project (default: None)")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder; default: False)")
    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="LinearSystem-v0",
        help="the id of the environment (default: LinearSystem-v0)")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments (default: 1000000)")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size (default: 1000000)")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma (default: 0.99)")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory (default: 256)")
    parser.add_argument("--learning-starts", type=int, default=5e3,
        help="timestep to start learning (default: 5000)")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer (default: 0.0003)")
    parser.add_argument("--v-lr", type=float, default=1e-3,
        help="the learning rate of the V network optimizer (default: 0.001)")
    parser.add_argument("--q-lr", type=float, default=1e-3,
        help="the learning rate of the Q network optimizer (default: 0.001)")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed; default: 2)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks (default: 1)")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization (default: 0.5)")
    parser.add_argument("--alpha", type=float, default=0.2,
        help="Entropy regularization coefficient (default: 0.2)")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient (default: True)")
    parser.add_argument("--alpha-lr", type=float, default=1e-3,
        help="the learning rate of the alpha network optimizer (default: 0.001)")
    # Koopman specific arguments
    parser.add_argument('--num-paths', type=int, default=100,
        help='Number of paths for the dataset (default: 100)')
    parser.add_argument('--num-steps-per-path', type=int, default=300,
        help='Number of steps per path for the dataset (default: 300)')
    parser.add_argument('--state-order', type=int, default=2,
        help='Order of monomials to use for state dictionary (default: 2)')
    parser.add_argument('--action-order', type=int, default=2,
        help='Order of monomials to use for action dictionary (default: 2)')
    parser.add_argument('--regressor', type=str, default='ols', choices=['ols', 'sindy', 'rrr', 'ridge'], nargs="?", const=True,
        help='Which regressor to use to build the Koopman tensor (default: \'ols\')')
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def checkMatrixRank(X, name):
    rank = torch.linalg.matrix_rank(X)
    print(f"{name} matrix rank: {rank}")
    if rank != X.shape[0]:
        # raise ValueError(f"{name} matrix is not full rank ({rank} / {X.shape[0]})")
        pass

def checkConditionNumber(X, name, threshold=200):
    cond_num = torch.linalg.cond(X)
    print(f"Condition number of {name}: {cond_num}")
    if cond_num > threshold:
        # raise ValueError(f"Condition number of {name} is too large ({cond_num} > {threshold})")
        pass

def ols(X, Y):
    return torch.linalg.lstsq(X, Y, rcond=None).solution

def OLS(X, Y):
    return ols(X, Y)

def SINDy(Theta, dXdt, lamb=0.05):
    d = dXdt.shape[1]
    Xi = torch.linalg.lstsq(Theta, dXdt, rcond=None).solution # Initial guess: Least-squares

    for _ in range(10):
        smallinds = torch.abs(Xi) < lamb # Find small coefficients
        Xi[smallinds] = 0                # and threshold
        for ind in range(d):             # n is state dimension
            biginds = smallinds[:, ind] == 0
            # Regress dynamics onto remaining terms to find sparse Xi
            Xi[biginds, ind] = torch.linalg.lstsq(
                Theta[:, biginds],
                dXdt[:, ind].unsqueeze(0).T,
                rcond=None
            ).solution[:, 0]

    L = Xi
    return L

def rrr(X, Y, rank=8):
    B_ols = ols(X, Y) # if infeasible use GD (numpy CG)
    U, S, V = torch.linalg.svd(Y.T @ X @ B_ols)
    W = V[0:rank].T

    B_rr = B_ols @ W @ W.T
    L = B_rr#.T
    return L

def RRR(X, Y, rank=8):
    return rrr(X, Y, rank)

def ridgeRegression(X, y, lamb=0.05):
    return torch.linalg.inv(X.T @ X + (lamb * torch.eye(X.shape[1]))) @ X.T @ y


class Regressor(str, Enum):
    OLS = 'ols'
    SINDy = 'sindy'
    RRR = 'rrr'
    RIDGE = 'ridge'


class KoopmanTensor:
    def __init__(
        self,
        X,
        Y,
        U,
        phi,
        psi,
        regressor=Regressor.OLS,
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
            finite_differences = (self.Y - self.X) # (self.x_dim, self.N)
            phi_derivative = self.phi.diff(self.X) # (self.phi_dim, self.x_dim, self.N)
            phi_double_derivative = self.phi.ddiff(self.X) # (self.phi_dim, self.x_dim, self.x_dim, self.N)
            self.regression_Y = torch.einsum(
                'os,pos->ps',
                finite_differences / self.dt,
                phi_derivative
            )
            self.regression_Y += torch.einsum(
                'ot,pots->ps',
                0.5 * ( finite_differences @ finite_differences.T ) / self.dt, # (state_dim, state_dim)
                phi_double_derivative
            )
        else:
            # Set regression target to phi(Y)
            self.regression_Y = self.Phi_Y

        # Make sure data is full rank
        checkMatrixRank(self.Phi_X, "Phi_X")
        checkMatrixRank(self.regression_Y, "dPhi_Y" if is_generator else "Phi_Y")
        checkMatrixRank(self.Psi_U, "Psi_U")
        print('\n')

        # Make sure condition numbers are small
        checkConditionNumber(self.Phi_X, "Phi_X")
        checkConditionNumber(self.regression_Y, "dPhi_Y" if is_generator else "Phi_Y")
        checkConditionNumber(self.Psi_U, "Psi_U")
        print('\n')

        # Build matrix of kronecker products between u_i and x_i for all 0 <= i <= N
        self.kron_matrix = torch.empty([
            self.psi_dim * self.phi_dim,
            self.N
        ])
        for i in range(self.N):
            self.kron_matrix[:,i] = torch.kron(
                self.Psi_U[:,i],
                self.Phi_X[:,i]
            )

        # Solve for M and B
        if regressor == Regressor.RRR:
            self.M = rrr(self.kron_matrix.T, self.regression_Y.T, rank).T
            self.B = rrr(self.Phi_X.T, self.X.T, rank)
        elif regressor == Regressor.SINDy:
            self.M = SINDy(self.kron_matrix.T, self.regression_Y.T).T
            self.B = SINDy(self.Phi_X.T, self.X.T)
        elif regressor == Regressor.OLS:
            self.M = ols(self.kron_matrix.T, self.regression_Y.T).T
            self.B = ols(self.Phi_X.T, self.X.T)
        elif regressor == Regressor.RIDGE:
            self.M = ridgeRegression(self.kron_matrix.T, self.regression_Y.T).T
            self.B = ridgeRegression(self.Phi_X.T, self.X.T)
        else:
            raise Exception("Did not pick a supported regression algorithm.")

        # reshape M into tensor K
        self.K = np.empty([
            self.phi_dim,
            self.phi_dim,
            self.psi_dim
        ])
        self.M = self.M.numpy()
        for i in range(self.phi_dim):
            self.K[i] = self.M[i].reshape(
                (self.phi_dim, self.psi_dim),
                order='F'
            )

        # Cast to tensors
        self.M = torch.tensor(self.M, dtype=torch.float64)
        self.K = torch.tensor(self.K, dtype=torch.float64)

    def K_(self, u):
        """
        Compute the Koopman operator associated with a given action.

        Parameters
        ----------
        u : array_like
            Action as a column vector or matrix of column vectors for which the Koopman operator is computed.

        Returns
        -------
        ndarray
            Koopman operator corresponding to the given action.
        """

        K_u = torch.einsum('ijz,zk->kij', self.K, self.psi(u))

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

        K_us = self.K_(u) # (batch_size, phi_dim, phi_dim)
        phi_x = self.phi(x) # (phi_dim, batch_size)

        if len(K_us.shape) == 2:
            return K_us @ phi_x

        phi_x_primes = (K_us @ phi_x.T.unsqueeze(-1)).squeeze(-1)
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


# TODO: Need to get the observables into this file
def generate_koopman_tensor(
        env_id,
        seed,
        num_paths,
        num_steps_per_path,
        state_order,
        action_order,
        regressor):

    # Set seeds and create environment
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = gym.make(env_id)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)

    # Collect data
    state_dim = env.observation_space.shape
    state_dim = 1 if len(state_dim) == 0 else state_dim[0]
    action_dim = env.action_space.shape
    action_dim = 1 if len(action_dim) == 0 else action_dim[0]
    X = torch.zeros((num_paths, num_steps_per_path, state_dim))
    Y = torch.zeros_like(X)
    U = torch.zeros((num_paths, num_steps_per_path, action_dim))

    for path_num in range(num_paths):
        state = env.reset()
        for step_num in range(num_steps_per_path):
            X[path_num, step_num] = torch.tensor(state)
            action = env.action_space.sample()
            U[path_num, step_num] = torch.tensor(action)

            state, _, _, _ = env.step(action)
            Y[path_num, step_num] = torch.tensor(state)
    
    # Reshape data into matrices
    total_num_datapoints = num_paths * num_steps_per_path
    X = X.reshape(total_num_datapoints, state_dim).T
    Y = Y.reshape(total_num_datapoints, state_dim).T
    U = U.reshape(total_num_datapoints, action_dim).T

    # Construct the Koopman tensor
    try:
        path_based_tensor = KoopmanTensor(
            X,
            Y,
            U,
            phi=monomials(state_order),
            psi=monomials(action_order),
            regressor=Regressor(regressor),
            dt=env.dt,
        )
    except:
        # Assume the error was because there is no dt for LinearSystem
        path_based_tensor = KoopmanTensor(
            X,
            Y,
            U,
            phi=monomials(state_order),
            psi=monomials(action_order),
            regressor=Regressor(regressor),
        )

    return path_based_tensor


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class SoftVNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class SoftKoopmanVNetwork(nn.Module):
    def __init__(self, koopman_tensor):
        super().__init__()

        self.koopman_tensor = koopman_tensor
        self.phi_state_dim = self.koopman_tensor.Phi_X.shape[0]

        self.linear = nn.Linear(self.phi_state_dim, 1, bias=False)

    def forward(self, state):
        """Linear in the phi(x)s"""

        phi_xs = self.koopman_tensor.phi(state.T).T

        output = self.linear(phi_xs)

        return output


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))

        # action rescaling
        high_action = env.action_space.high
        low_action = env.action_space.low
        # high_action = np.clip(env.action_space.high, a_min=-1000, a_max=1000)
        # low_action = np.clip(env.action_space.low, a_min=-1000, a_max=1000)
        # dtype = torch.float32
        dtype = torch.float64
        action_scale = torch.tensor((high_action - low_action) / 2.0, dtype=dtype)
        action_bias = torch.tensor((high_action + low_action) / 2.0, dtype=dtype)
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    args = parse_args()
    curr_time = int(time.time())

    # Generate a random seed
    sampled_seed = np.random.randint(1000)

    run_name = f"{args.env_id}__{args.exp_name}__{sampled_seed}__{curr_time}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Create folder for model checkpoints
    model_chkpt_path = f"./saved_models/{args.env_id}/value_based_sa{'k' if args.koopman else ''}c_chkpts_{curr_time}"
    create_folder(model_chkpt_path)

    # TRY NOT TO MODIFY: seeding
    random.seed(sampled_seed)
    np.random.seed(sampled_seed)
    torch.manual_seed(sampled_seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Running everything on CPU
    device = torch.device("cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, sampled_seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)

    # Koopman Logic
    # TODO: Generate the Koopman Tensor in-place to do away with having to store it beforehand
    koopman_tensor = generate_koopman_tensor(
        args.env_id, args.seed, args.num_paths, args.num_steps_per_path, args.state_order, args.action_order, args.regressor)
    #koopman_tensor = load_tensor(args.env_id, "path_based_tensor")
    vf = SoftKoopmanVNetwork(koopman_tensor).to(device)
    vf_target = SoftKoopmanVNetwork(koopman_tensor).to(device)
    
    # Usual Value function logic
    vf_target.load_state_dict(vf.state_dict())
    v_optimizer = optim.Adam(list(vf.parameters()), lr=args.v_lr)
    # v_optimizer = optim.Adam(list(vf.parameters()), lr=args.v_lr, weight_decay=1e-5)
    # v_optimizer = optim.Adam(list(vf.parameters()), lr=args.v_lr, weight_decay=1e3)

    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(
            torch.Tensor(envs.single_action_space.shape).to(device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)
    else:
        alpha = args.alpha

    # envs.single_observation_space.dtype = np.float32
    envs.single_observation_space.dtype = np.float64
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions, log_probs, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}"
                )
                writer.add_scalar(
                    "charts/episodic_return", info["episode"]["r"], global_step
                )
                writer.add_scalar(
                    "charts/episodic_length", info["episode"]["l"], global_step
                )
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            # Sample from replay buffer
            data = rb.sample(args.batch_size)

            # E_s_t~D [ 1/2 ( V_psi( s_t ) - E_a_t~pi_phi [ Q_theta( s_t, a_t ) - log pi_phi( a_t | s_t ) ] )^2 ]
            vf_values = vf(data.observations).view(-1)
            with torch.no_grad():
                state_actions, state_log_pis, _ = actor.get_action(data.observations)
                q_values = torch.min(
                    qf1(data.observations, state_actions),
                    qf2(data.observations, state_actions),
                ).view(-1)
            vf_loss = F.mse_loss(vf_values, q_values - alpha * state_log_pis.view(-1))
            # vf_loss = F.l1_loss(vf_values, q_values - alpha * state_log_pis.view(-1))
            # Calculate L1 regularization term
            # with torch.no_grad():
            #     l1_regularization = torch.tensor(0., requires_grad=True)
            #     for param in vf.parameters():
            #         l1_regularization += torch.norm(param, p=1)
            # total_vf_loss = vf_loss + l1_regularization

            v_optimizer.zero_grad()
            vf_loss.backward()
            v_optimizer.step()

            # E_( s_t, a_t )~D [ 1/2 ( Q_theta( s_t, a_t ) - Q_target( s_t, a_t ) )^2 ]
            with torch.no_grad():
                if args.koopman:
                    expected_phi_x_primes = koopman_tensor.phi_f(
                        data.observations.T, data.actions.T
                    ).T
                    vf_next_target = (
                        (1 - data.dones.flatten())
                        * args.gamma
                        * vf_target.linear(expected_phi_x_primes).view(-1)
                    )
                else:
                    vf_next_target = (
                        (1 - data.dones.flatten())
                        * args.gamma
                        * vf_target(data.next_observations).view(-1)
                    )
                q_target_values = data.rewards.flatten() + vf_next_target

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, q_target_values)
            qf2_loss = F.mse_loss(qf2_a_values, q_target_values)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # E_s_t~D,e_t~N [ log pi_phi( f_phi( e_t; s_t ) | s_t ) - Q_theta( s_t, f_phi( e_t; s_t ) ) ]
            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(vf.parameters(), vf_target.parameters()):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            if global_step % 100 == 0:
                writer.add_scalar(
                    "losses/vf_values", vf_values.mean().item(), global_step
                )
                writer.add_scalar("losses/vf_loss", vf_loss.item(), global_step)
                writer.add_scalar(
                    "losses/qf1_values", qf1_a_values.mean().item(), global_step
                )
                writer.add_scalar(
                    "losses/qf2_values", qf2_a_values.mean().item(), global_step
                )
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                if args.autotune:
                    writer.add_scalar(
                        "losses/alpha_loss", alpha_loss.item(), global_step
                    )
                sps = int(global_step / (time.time() - start_time))
                print("Steps per second (SPS):", sps)
                writer.add_scalar("charts/SPS", sps, global_step)

            # Checkpoint policy network every so often
            if global_step == 0 or (global_step + 1) % 1000 == 0:
                torch.save(
                    actor.state_dict(), f"{model_chkpt_path}/step_{global_step + 1}.pt"
                )

    envs.close()
    writer.close()

    # Get optimal value function weights from Koopman model
    if args.koopman:
        value_function_weights = list(vf.parameters())
        target_value_function_weights = list(vf_target.parameters())
        print(f"Value function weights: {value_function_weights}")
        print(f"Target value function weights: {target_value_function_weights}")
