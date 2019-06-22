from abc import ABC, abstractmethod


class BaseEnvironment(ABC):

    @abstractmethod
    def reset(self):
        # TODO
        pass

    @abstractmethod
    def step(self, action):
        # TODO
        pass

    @property
    def action_space(self):
        return self._get_action_space()

    @abstractmethod
    def _get_action_space(self):
        pass

    @abstractmethod
    def sample_random_action(self):
        pass
