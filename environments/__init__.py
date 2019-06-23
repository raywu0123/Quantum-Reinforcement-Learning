from .gridworld import GridWorldEnv
from .mygym import GymEnv


class EnvHub:

    custom_env_hub = {
        'gridworld': GridWorldEnv,
    }

    @classmethod
    def get_env(cls, env_id, **kwargs):
        if env_id in cls.custom_env_hub:
            return cls.custom_env_hub[env_id](**kwargs)
        else:
            return GymEnv(env_id, **kwargs)



