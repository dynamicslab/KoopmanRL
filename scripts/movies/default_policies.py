import numpy as np

from movies import Policy

class ZeroPolicy(Policy):
    def __init__(self, is_2d=False, name=None):
        self.is_2d = is_2d
        self._name = name

    @property
    def name(self):
        if self._name is None:
            return self.__class__.__name__
        return self._name

    def get_action(self, state, *args, **kwargs):
        return np.array([[0]]) if self.is_2d else np.array([0])

class RandomPolicy(Policy):
    def __init__(self, gym_env, name=None):
        self.env = gym_env
        self._name = name

    @property
    def name(self):
        if self._name is None:
            return self.__class__.__name__
        return self._name

    def get_action(self, state, *args, **kwargs):
        return self.env.action_space.sample()