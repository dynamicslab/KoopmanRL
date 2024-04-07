import gym
# import matplotlib.pyplot as plt
import numpy as np
import torch

from gym import spaces
from gym.envs.registration import register
from scipy.integrate import solve_ivp

dt = 0.01
max_episode_steps = int(20 / dt)
# max_episode_steps = int(2 / dt)

register(
    id='Lorenz-v0',
    entry_point='custom_envs.lorenz:Lorenz',
    max_episode_steps=max_episode_steps
)

class Lorenz(gym.Env):
    def __init__(self):
        # Configuration with hardcoded values
        self.state_dim = 3
        self.action_dim = 1

        self.state_range = [-25.0, 25.0]
        # self.state_range = [-5.0, 5.0]

        # self.action_range = [-10.0, 10.0]
        self.action_range = [-75.0, 75.0]
        # self.action_range = [-500.0, 500.0]

        self.dt = dt

        # Dynamics
        self.sigma = 10
        self.rho = 28
        self.beta = 8/3

        self.x_e = np.sqrt( self.beta * ( self.rho - 1 ) )
        self.y_e = np.sqrt( self.beta * ( self.rho - 1 ) )
        self.z_e = self.rho - 1

        # For LQR
        x_bar = self.x_e
        y_bar = self.y_e
        z_bar = self.z_e
        self.continuous_A = np.array([
            [-self.sigma, self.sigma, 0],
            [self.rho - z_bar, -1, 0],
            [y_bar, x_bar, -self.beta]
        ])
        self.continuous_B = np.array([
            [1],
            [0],
            [0]
        ])

        # Define cost/reward values
        self.Q = np.eye(self.state_dim)
        self.R = np.eye(self.action_dim) * 0.001

        self.reference_point = np.array([self.x_e, self.y_e, self.z_e])

        # Observations are 3-dimensional vectors indicating spatial location.
        self.state_minimums = np.array([-20.0, -50.0, 0.0])
        self.state_maximums = np.array([20.0, 50.0, 50.0])
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

    def reset(self, seed=None, options={"state": None}):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the initial state uniformly at random
        # self.state = self.observation_space.sample() if options['state'] is None else options['state']
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

    def continuous_f(self, action=None):
        """
            True, continuous dynamics of the system.

            INPUTS:
                action - Action vector. If left as None, then random policy is used.
        """

        def f_u(t, input):
            """
                INPUTS:
                    input - State vector.
                    t - Timestep.
            """

            x, y, z = input

            x_dot = self.sigma * ( y - x )   # sigma*y - sigma*x
            y_dot = ( self.rho - z ) * x - y # rho*x - x*z - y
            z_dot = x * y - self.beta * z    # x*y - beta*z

            u = action
            if u is None:
                u = np.zeros(self.action_dim)

            return [ x_dot + u[0], y_dot, z_dot ]

        return f_u

    def f(self, state, action):
        """
            True, discretized dynamics of the system. Pushes forward from (t) to (t + dt) using a constant action.

            INPUTS:
                state - State array.
                action - Action array.

            OUTPUTS:
                State array pushed forward in time.
        """

        soln = solve_ivp(fun=self.continuous_f(action), t_span=[0, dt], y0=state, method='RK45')

        return soln.y[:, -1]

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