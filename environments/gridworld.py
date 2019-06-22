import random

from .base import BaseEnvironment


class GridWorldEnv(BaseEnvironment):

    def __init__(self, **kwargs):
        self.grid = [[1, 2, 3, 4, 5],
                     [6, 7, 8, 9, 10],
                     [11, 12, -1, 13, 14],
                     [15, 16, -1, 17, 18],
                     [19, 20, 21, 22, 23]]
        self.state = [0, 0]
        self.position = 1
        self.actions = [0, 1, 2, 3]  # left, up, right, down
        self.states = 23
        self.final_state = 23
        self.reward = [[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, -10, 0, 10]]
        self.done = False

    def reset(self):
        self.state = [0, 0]
        self.position = 1
        self.done = False

        return self.position

    def step(self, action):
        if action not in self.action_space:
            raise ValueError(f'Invalid action: {action}')

        if action == 0:
            self.state[1] -= 1
        elif action == 1:
            self.state[0] -= 1
        elif action == 2:
            self.state[1] += 1
        elif action == 3:
            self.state[0] += 1

        if self.state[0] < 0:
            self.state[0] = 0
        elif self.state[0] > 4:
            self.state[0] = 4
        elif self.state[1] < 0:
            self.state[1] = 0
        elif self.state[1] > 4:
            self.state[1] = 4
        elif self.state[1] == 2 and (self.state[0] == 2 or self.state[0] == 3):
            if action == 0:
                self.state[1] = 3
            elif action == 1:
                self.state[0] = 4
            elif action == 2:
                self.state[1] = 1
            elif action == 3:
                self.state[0] = 1

        self.position = self.grid[self.state[0]][self.state[1]]
        reward = self.reward[self.state[0]][self.state[1]]

        if self.position == self.final_state:
            self.done = True
        return self.position, reward, self.done

    def _get_action_space(self):
        return [0, 1, 2, 3]

    def sample_random_action(self):
        return random.choice(self.action_space)
