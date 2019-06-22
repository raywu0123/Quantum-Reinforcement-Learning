from argparse import ArgumentParser


def get_parser():
    parser = ArgumentParser()
    # TODO
    parser.add_argument(
        '-e',
        '--env_id',
        type=str,
        default='gridworld',
        help='name of the environment [gridworld]'
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
        type=bool,
        default=False,
    )
    parser.add_argument(
        '-s',
        '--save',
        type=bool,
        default=False,
    )
    parser.add_argument(
        '--random',
        dest='random',
        action='store_true'
    )
    parser.set_defaults(random=False)
    return parser
