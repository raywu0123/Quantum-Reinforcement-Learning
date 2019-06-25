import numpy as np
from tqdm import tqdm

from parser import get_parser
from environments import EnvHub
from agents import AgentHub
from lib import plotting


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    env = EnvHub.get_env(env_id=args.env_id, save=args.save)
    get_agent = AgentHub[args.agent_id]

    agent = get_agent(
        action_space=env.action_space,
        discount_factor=args.discount_factor,
    )

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(args.num_episodes),
        episode_rewards=np.zeros(args.num_episodes),
    )

    p_bar = tqdm(range(args.num_episodes))
    for i_episode in p_bar:
        state = env.reset()
        done = False
        t = 0
        while not done:
            if args.random:
                action = env.sample_random_action()
            else:
                action = agent.get_action(state, env)

            next_state, reward, done = env.step(action)
            agent.learn(state, action, next_state, reward)
            state = next_state

            stats.episode_rewards[i_episode] += (args.discount_factor ** t) * reward
            stats.episode_lengths[i_episode] = t + 1
            t += 1

        p_bar.set_description(f'episode_reward: {stats.episode_rewards[i_episode]:.3f}')
    plotting.plot_episode_stats(stats)
