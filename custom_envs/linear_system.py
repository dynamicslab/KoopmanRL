import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from gym import spaces
from gym.envs.registration import register

max_episode_steps = 200

register(
    id='LinearSystem-v0',
    entry_point='custom_envs.linear_system:LinearSystem',
    max_episode_steps=max_episode_steps
)

class LinearSystem(gym.Env):
    def __init__(self):
        # Configuration with hardcoded values
        self.state_dim = 3
        self.action_dim = 1

        self.state_range = [-25.0, 25.0]

        self.action_range = [-10.0, 10.0]

        self.max_episode_steps = max_episode_steps

        # Dynamics
        max_eigen_factor = np.random.uniform(0.7, 1)
        print(f"max eigen factor: {max_eigen_factor}")
        Z = np.random.rand(self.state_dim, self.state_dim)
        _, sigma, _ = np.linalg.svd(Z)
        Z = Z * np.sqrt(max_eigen_factor) / np.max(sigma)
        self.A = Z.T @ Z
        W, _ = np.linalg.eig(self.A)
        max_abs_real_eigen_val = np.max(np.abs(np.real(W)))

        print(f"A:\n{self.A}")
        print(f"A's max absolute real eigenvalue: {max_abs_real_eigen_val}")
        self.B = np.ones([self.state_dim, self.action_dim])

        # Define cost/reward values
        self.Q = np.eye(self.state_dim)
        self.R = np.eye(self.action_dim)

        self.reference_point = np.zeros(self.state_dim)

        # Observations are 3-dimensional vectors indicating spatial location.
        self.state_minimums = np.ones(self.state_dim) * self.state_range[0]
        self.state_maximums = np.ones(self.state_dim) * self.state_range[1]
        self.observation_space = spaces.Box(
            low=self.state_minimums,
            high=self.state_maximums,
            shape=(self.state_dim,),
            dtype=np.float64
        )

        # We have a continuous action space. In this case, there is only 1 dimension per action
        self.action_minimums = np.ones(self.action_dim) * self.action_range[0]
        self.action_maximums = np.ones(self.action_dim) * self.action_range[1]
        self.action_space = spaces.Box(
            low=self.action_minimums,
            high=self.action_maximums,
            shape=(self.action_dim,),
            dtype=np.float64
        )

        # History of states traversed during the current episode
        self.states = []

    def reset(self, seed=None, options={}):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the initial state uniformly at random
        # self.state = self.observation_space.sample()
        self.state = np.random.uniform(
            low=self.state_minimums,
            high=self.state_maximums,
            size=(self.state_dim,)
        )
        self.states = [self.state]

        # Track number of steps taken
        self.step_count = 0

        # return self.state, {}
        return self.state

    def cost_fn(self, state, action):
        _state = state - self.reference_point

        cost = _state @ self.Q @ _state.T + action @ self.R @ action.T

        return cost

    def reward_fn(self, state, action):
        return -self.cost_fn(state, action)
    
    def vectorized_cost_fn(self, states, actions):
        _states = (states - self.reference_point).T
        mat = torch.diag(_states.T @ self.Q @ _states).unsqueeze(-1) + torch.pow(actions.T, 2) * self.R

        return mat.T
    
    def vectorized_reward_fn(self, states, actions):
        return -self.vectorized_cost_fn(states, actions)

    def f(self, state, action):
        """
            True dynamics of linear system.

            INPUTS:
                state - State as an array.
                action - Action as an array.

            OUTPUTS:
                state' - Next state as an array.
        """

        return self.A @ state + self.B @ action

    def step(self, action):
        # Compute reward of system
        reward = self.reward_fn(self.state, action)

        # Update state
        self.state = self.f(self.state, action)
        self.states.append(self.state)

        # Update global step count
        self.step_count += 1

        # An episode is done if the system has run for max_episode_steps
        terminated = self.step_count >= max_episode_steps

        # return self.state, reward, terminated, False, {}
        return self.state, reward, terminated, {}