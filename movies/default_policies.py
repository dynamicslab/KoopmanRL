import numpy as np

from movies import Policy

class ZeroPolicy(Policy):
    def __init__(self, is_2d=False):
        self.is_2d = is_2d

    def get_action(self, state, *args, **kwargs):
        return np.array([[0]]) if self.is_2d else np.array([0])

class RandomPolicy(Policy):
    def __init__(self, gym_env):
        self.env = gym_env

    def get_action(self, state, *args, **kwargs):
        return self.env.action_space.sample()