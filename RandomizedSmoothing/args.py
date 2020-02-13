import argparse
from .datasets import DATASETS


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Args:
    def __init__(self):
        self.args = self._get_parser().parse_args()

    @staticmethod
    def _get_parser():
        parser = argparse.ArgumentParser(description='Breaking Certifiable Defenses')
        parser.add_argument('-d', '--dataset', choices=DATASETS, help='Dataset: Cifar or ImageNet')
        parser.add_argument('-c', '--checkpoint', type=str, help='Path to saved pytorch model of base classifier')
        parser.add_argument('-g', '--sigma', type=float, default=0.5, help='Noise sigma hyper-parameter')
        parser.add_argument('-b', '--batch', type=int, default=400, help='Batch size')  # was 1000
        parser.add_argument('-k', '--skip', type=int, default=20, help='How many examples to skip for validation')
        parser.add_argument('-a', '--alpha', type=float, default=0.001, help='Failure probability')
        parser.add_argument('--N0', type=int, default=100)
        parser.add_argument('--N', type=int, default=100000, help='number of samples to use')

        parser.add_argument('-n', '--steps', default=300, type=int, help='#PGD steps')
        parser.add_argument('-p', '--print_stats', type=str2bool, default=True,
                            help='Do you want to see stats during the attack')

        parser.add_argument('-l', '--lr', type=float, default=0.1, help='Learning rate for attack')
        parser.add_argument('-r', '--duplicate_rgb', type=str2bool, default=True, help='Duplicate one Channel to RGB?')
        parser.add_argument('-t', '--tv_lam', type=float, default=0.3, help='TV-Lambda')
        parser.add_argument('-e', '--ch_lam', type=float, default=1.0, help='Channels Lambda')
        parser.add_argument('-s', '--dissim_lam', type=float, default=0.5, help='Sim Lambda')
        return parser

    def get_args(self):
        return self.args

    def get_name(self):
        return '_'.join(['{}{}'.format(k[0], v) for k, v in vars(self.args).items()])
