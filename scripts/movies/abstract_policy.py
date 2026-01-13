from abc import ABC, abstractmethod

class Policy(ABC):
    @abstractmethod
    def get_action(self, state, *args, **kwargs):
        raise NotImplementedError