from abc import ABC, abstractmethod


class BaseAgent(ABC):

    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def learn(self, state, action, next_state, reward):
        pass
