import numpy as np
from tqdm import tqdm

from parser import get_parser
from environments import EnvHub
from agents import AgentHub
from lib import plotting


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    get_env = EnvHub[args.env_id]
    get_agent = AgentHub[args.agent_id]

    env = get_env(save=args.save)
    agent = get_agent(
        action_space=env.action_space,
        discount_factor=args.discount_factor,
    )

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(args.num_episodes),
        episode_rewards=np.zeros(args.num_episodes),
    )

    for i_episode in tqdm(range(args.num_episodes)):
        state = env.reset()
        done = False
        t = 0
        while not done:
            if args.random:
                action = env.sample_random_action()
            else:
                action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, next_state, reward)
            state = next_state

            stats.episode_rewards[i_episode] += (args.discount_factor ** t) * reward
            stats.episode_lengths[i_episode] = t + 1
            t += 1

    plotting.plot_episode_stats(stats)
