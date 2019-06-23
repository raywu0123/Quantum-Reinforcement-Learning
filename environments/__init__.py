from .gridworld import GridWorldEnv
from .mygym import GymEnv

class Hub:
    def __init__(self, hub):
        self.hub = hub

    def __getitem__(self, env_id):
        if env_id in self.hub:
            return hub[env_id]
        else:
            return lambda save: GymEnv(env_id, save)

# EnvHub = {
#     'gridworld': GridWorldEnv,
# }

EnvHub = Hub({
    'gridworld': GridWorldEnv,
})