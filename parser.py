from argparse import ArgumentParser


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        '-e',
        '--env_id',
        type=str,
        default='gridworld',
        help='name of the environment '
             '['
             'gridworld / '
             'FrozenLake-notslip / FrozenLake-slip / '
             'FrozenLake8x8-notslip / FrozenLake8x8-slip / '
             'Taxi-v2'
             ']'
    )
    parser.add_argument(
        '-a',
        '--agent_id',
        type=str,
        default='quantum',
        help='name of the agent [quantum / tradition-q]'
    )
    parser.add_argument(
        '-ep',
        '--num_episodes',
        type=int,
        default=200,
        help='num of episodes to run'
    )
    parser.add_argument(
        '-d',
        '--discount_factor',
        type=float,
        default=0.9,
    )
    parser.add_argument(
        '-r',
        '--render',
        dest='render',
        action='store_true'
    )
    parser.add_argument(
        '-s',
        '--save',
        dest='save',
        action='store_true'
    )
    parser.add_argument(
        '--random',
        dest='random',
        action='store_true'
    )
    parser.set_defaults(render=False, save=False, random=False)
    return parser
