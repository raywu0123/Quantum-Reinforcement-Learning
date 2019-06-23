import random
import numpy as np
import gym
import pickle
from .base import BaseEnvironment


class GymEnv(BaseEnvironment):

    def __init__(self, env_id, save):
        print(env_id)
        self.env = gym.make(env_id)
        if save:
            self.env = gym.wrappers.Monitor(self.env, './video/',video_callable=lambda episode_id: True, force = True)
        # int64 float32
        self.s_type = self.env.observation_space.dtype.name
        self.a_type = self.env.action_space.dtype.name

        # determine the number of total states & action
        # notice that the range may be so large such that we cannot use discrete state
        # i.e. CartPole-v1
        if 'float' in self.s_type:
            self.s_size = 100.
            self.s_unit = ((self.env.observation_space.high - self.env.observation_space.low) / self.s_size)
        
        elif 'int' in self.s_type:
            self.s_size = self.env.observation_space.n
        
        else:
            print('unexpected error')

        if 'float' in self.a_type:
            self.a_size = 100
            self.a_unit = ((self.env.action_space.high - self.env.action_space.low) / self.a_size)
            self.a_space = []
            for i in range(self.a_size):
                self.a_space.append(self.env.action_space.low + i * self.a_unit)

        elif 'int' in self.a_type:
            self.a_size = self.env.action_space.n
            self.a_space = []
            for i in range(self.a_size):
                self.a_space.append(i)
        
        else:
            print('unexpected error')


    def reset(self):
        s = self.env.reset()

        if 'float' in self.s_type:
            s = self.env.observation_space.low + (s - self.env.observation_space.low).astype(int) / self.s_unit

        s_serialized = pickle.dumps(s)

        return s_serialized

    def step(self, action):
        print(self.a_space, action)
        if action not in self.a_space:
            raise ValueError(f'Invalid action: {action}')

        s, r, done, info = self.env.step(action)

        if 'float' in self.s_type:
            s = self.env.observation_space.low + (s - self.env.observation_space.low).astype(int) / self.s_unit

        s_serialized = pickle.dumps(s)

        return s_serialized, r, done

    def _get_action_space(self):
        return self.a_space

    def sample_random_action(self):
        return random.choice(self.action_space)

    def render(self):
        self.render()