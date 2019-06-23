from functools import partial

from .gridworld import GridWorldEnv
from .mygym import GymEnv


class EnvHub:

    custom_env_hub = {
        'gridworld': GridWorldEnv,
        'FrozenLake-notslip': partial(
            GymEnv,
            env_id='FrozenLake-v0',
            is_slippery=False,
        ),
        'FrozenLake-slip': partial(
            GymEnv,
            env_id='FrozenLake-v0',
            is_slippery=True,
        ),
        'FrozenLake8x8-notslip': partial(
            GymEnv,
            env_id='FrozenLake8x8-v0',
            is_slippery=False,
        ),
        'FrozenLake8x8-slip': partial(
            GymEnv,
            env_id='FrozenLake8x8-v0',
            is_slippery=True,
        )
    }

    @classmethod
    def get_env(cls, env_id, **kwargs):
        if env_id in cls.custom_env_hub:
            return cls.custom_env_hub[env_id](**kwargs)
        else:
            return GymEnv(env_id, **kwargs)



