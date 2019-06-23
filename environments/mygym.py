import random
import gym
import numpy as np
from .base import BaseEnvironment


class GymEnv(BaseEnvironment):

    def __init__(self, env_id, save, **kwargs):
        print(env_id)
        self.env = gym.make(env_id, **kwargs)
        if save:
            self.env = gym.wrappers.Monitor(
                self.env,
                './video/',
                video_callable=lambda episode_id: True,
                force=True
            )
        # int64 float32
        self.s_type = self.env.observation_space.dtype.name
        self.a_type = self.env.action_space.dtype.name

        # determine the number of total states & action
        # notice that the range may be so large such that we cannot use discrete state
        # i.e. CartPole-v1
        if 'float' in self.s_type:
            self.s_size = 100
            high = np.clip(self.env.observation_space.high, a_min=-1e8, a_max=1e8)
            low = np.clip(self.env.observation_space.low, a_min=-1e8, a_max=1e8)
            self.s_unit = (high - low) / self.s_size

        elif 'int' in self.s_type:
            self.s_size = self.env.observation_space.n

        else:
            raise ValueError(f'unexpected s_type {self.s_type}')

        if 'float' in self.a_type:
            self.a_size = 100
            high = np.clip(self.env.action_space.high, a_min=-1e8, a_max=1e8)
            low = np.clip(self.env.action_space.low, a_min=-1e8, a_max=1e8)
            self.a_unit = (high - low) / self.a_size
            self.a_space = []
            for i in range(self.a_size):
                self.a_space.append(self.env.action_space.low + i * self.a_unit)

        elif 'int' in self.a_type:
            self.a_size = self.env.action_space.n
            self.a_space = []
            for i in range(self.a_size):
                self.a_space.append(i)
        else:
            raise ValueError(f'unexpected a_type {self.a_type}')

    def reset(self):
        s = self.env.reset()
        s = self.to_discrete_state(s)
        s_serialized = self.serialize_state(s)
        return s_serialized

    def step(self, action):
        if action not in self.a_space:
            raise ValueError(f'Invalid action: {action}')

        s, r, done, info = self.env.step(action)
        s = self.to_discrete_state(s)
        s_serialized = self.serialize_state(s)
        return s_serialized, r, done

    def _get_action_space(self):
        return self.a_space

    def sample_random_action(self):
        return random.choice(self.action_space)

    @staticmethod
    def serialize_state(state):
        if isinstance(state, np.ndarray):
            return state.tostring()
        elif isinstance(state, int) or isinstance(state, np.int64):
            return int(state)
        else:
            raise ValueError(f'unexpected state: {state}, type: {type(state)}')

    @staticmethod
    def quantize_state(state, unit):
        return state // unit

    def to_discrete_state(self, s):
        if 'float' in self.s_type:
            s = self.quantize_state(s - self.env.observation_space.low, self.s_unit)
        return s
