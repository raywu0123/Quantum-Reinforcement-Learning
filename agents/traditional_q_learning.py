import operator
import random

from .base import BaseAgent


class TraditionalQLearningAgent(BaseAgent):

    def __init__(self, action_space, discount_factor=0.99, alpha=0.8, epsilon=0.5, **kwargs):
        self.action_space = action_space
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.epsilon = epsilon
        self.explore_rate = epsilon
        self.Q_table = {}

    def get_action(self, state, env):
        if state not in self.Q_table or random.random() < self.explore_rate:
            return env.sample_random_action()

        action_reward_map = self.Q_table[state]
        self.update_explore_rate()
        return self.get_max(action_reward_map)

    def learn(self, state, action, next_state, reward):
        if state not in self.Q_table:
            action_reward_map = {}
        else:
            action_reward_map = self.Q_table[state]

        if action in action_reward_map:
            Q = action_reward_map[action]
        else:
            Q = 0.

        if next_state in self.Q_table:
            max_next_Q = self.get_max(self.Q_table[next_state], idx=1)
        else:
            max_next_Q = 0.

        action_reward_map[action] = Q + \
            self.alpha * (reward + self.discount_factor * max_next_Q - Q)
        self.Q_table[state] = action_reward_map

    @staticmethod
    def get_max(action_reward_map, idx=0):
        return max(action_reward_map.items(), key=operator.itemgetter(1))[idx]

    def update_explore_rate(self):
        self.explore_rate = max(self.explore_rate - self.epsilon / 1e5, 0)
