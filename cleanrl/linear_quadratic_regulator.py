import argparse
import gym
import numpy as np
import os
import time
import torch
torch.set_default_dtype(torch.float64)

from custom_envs import *
from control import dlqr, lqr
from distutils.util import strtobool
from scipy.stats import norm
from torch.utils.tensorboard import SummaryWriter

class LQRPolicy:
    def __init__(
        self,
        A,
        B,
        Q,
        R,
        reference_point,
        gamma=0.99,
        alpha=1.0,
        dt=None,
        is_continuous=False,
        seed=1
    ):
        """
        Initialize an LQR (Linear Quadratic Regulator) policy for an arbitrary system.

        Parameters
        ----------
        A : array_like, shape (n, n)
            Dynamics matrix describing the state evolution of the system.
        B : array_like, shape (n, m)
            Control matrix describing the action influence on the system.
        Q : array_like, shape (n, n)
            Cost coefficients for the state.
        R : array_like, shape (m, m)
            Cost coefficients for the action.
        reference_point : array_like, shape (n,)
            Point to which the system should tend.
        gamma : float, optional (default=0.99)
            The discount factor of the system, assuming the time step (dt) is 1.0.
        alpha : float, optional (default=1.0)
            The alpha (temperature) of the policy.
        dt : float, optional (default=None)
            The time step of the system.
        is_continuous : bool, optional (default=False)
            Boolean indicating whether A and B describe x or dx (discrete or continuous time dynamics).
        seed : int, optional (default=123)
            Seed for reproducibility.

        Notes
        -----
        The LQR policy is a control strategy that minimizes a quadratic cost function
        with linear dynamics, suitable for systems where the state evolution and action
        influence are represented by matrices A and B, and the cost coefficients for
        the state and action are represented by Q and R, respectively.
        """

        self.seed = seed

        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.reference_point = np.vstack(reference_point)
        self.gamma = gamma
        self.alpha = alpha
        self.dt = dt
        if self.dt is None:
            self.dt = 1.0
        self.discount_factor = self.gamma**self.dt
        self.is_continuous = is_continuous

        self.discounted_A = self.A * np.sqrt(self.discount_factor)
        self.discounted_R = self.R / self.discount_factor

        if is_continuous:
            self.lqr_soln = lqr(
                self.discounted_A,
                self.B,
                self.Q,
                self.discounted_R
            )
        else:
            self.lqr_soln = dlqr(
                self.discounted_A,
                self.B,
                self.Q,
                self.discounted_R
            )

        self.C = self.lqr_soln[0]
        self.P = self.lqr_soln[1]
        self.sigma_t = np.linalg.inv(self.discounted_R + self.B.T @ self.P @ self.B) * self.alpha

    def get_action_density(self, u, x, is_entropy_regularized=True):
        """
        Compute the normal density of an action given the current state.

        Parameters
        ----------
        u : array_like
            Action as a column vector.
        x : array_like
            State of the system as a column vector.
        is_entropy_regularized : bool, optional
            Whether or not to sample from a normal distribution.
            Default is True.

        Returns
        -------
        ndarray
            Density value of the (optimal) action conditional on the state `x`
            from the maximum entropy Linear Quadratic Regulator (LQR) policy.

        Raises
        ------
        Exception
            If `is_entropy_regularized` is False, indicating that the density method
            is only applicable in the entropy regularized case.

        Notes
        -----
        If `is_entropy_regularized` is True, the density is computed using the normal
        distribution with mean -C @ (x - reference_point) and standard deviation `sigma_t`.
        """

        if is_entropy_regularized:
            return norm.pdf(u, loc=-self.C @ (x - self.reference_point), scale=self.sigma_t)
        else:
            raise Exception("Density method is only applicable in the entropy regularized case")

    def get_action(self, x, is_entropy_regularized=True):
        """
        Compute the action given the current state.

        Parameters
        ----------
        x : array_like
            State of the system as a column vector.
        is_entropy_regularized : bool, optional
            Whether or not to sample from a normal distribution.
            Default is True.

        Returns
        -------
        np.ndarray
            Action from the Linear Quadratic Regulator (LQR) policy.

        Notes
        -----
        If `is_entropy_regularized` is True, the action is sampled from a normal
        distribution with mean -C @ (x - reference_point) and standard deviation `sigma_t`.
        If `is_entropy_regularized` is False, the action is deterministic and computed
        as -C @ (x - reference_point).
        """

        if is_entropy_regularized:
            return np.random.normal(loc=-self.C @ (x - self.reference_point), scale=self.sigma_t)
        else:
            return -self.C @ (x - self.reference_point)

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
        help="if toggled, cuda will be enabled (default: True)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="LinearSystem-v0",
        help="the id of the environment (default: LinearSystem-v0)")
    parser.add_argument("--total-timesteps", type=int, default=100_000,
        help="total timesteps of the experiments (default: 100_000)")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma (default: 0.99)")
    parser.add_argument("--alpha", type=float, default=1.0,
        help="entropy regularization coefficient (default: 1.0)")
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

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, False, run_name)])

    try:
        dt = envs.envs[0].dt
    except:
        dt = None

    # Construct LQR policy
    discrete_systems = ('LinearSystem-v0')
    is_continuous = False if args.env_id in discrete_systems else True
    try:
        lqr_policy = LQRPolicy(
            A=envs.envs[0].continuous_A,
            B=envs.envs[0].continuous_B,
            Q=envs.envs[0].Q,
            R=envs.envs[0].R,
            reference_point=envs.envs[0].reference_point,
            gamma=args.gamma,
            alpha=args.alpha,
            dt=dt,
            is_continuous=is_continuous,
            seed=args.seed
        )
    except:
        lqr_policy = LQRPolicy(
            A=envs.envs[0].A,
            B=envs.envs[0].B,
            Q=envs.envs[0].Q,
            R=envs.envs[0].R,
            reference_point=envs.envs[0].reference_point,
            gamma=args.gamma,
            alpha=args.alpha,
            dt=dt,
            is_continuous=is_continuous,
            seed=args.seed
        )

    envs.single_observation_space.dtype = np.float64
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        actions = lqr_policy.get_action(obs.T)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # Write data
        if global_step % 100 == 0:
            try:
                sps = int(global_step / (time.time() - start_time))
                print("Steps per second (SPS):", sps)
                writer.add_scalar("charts/SPS", sps, global_step)
            except:
                pass

    envs.close()
    writer.close()