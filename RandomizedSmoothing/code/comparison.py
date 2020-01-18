import setGPU
import pdb
import argparse
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
from architectures import get_architecture
from attack import SmoothAttack
import numpy as np
from advertorch.attacks import JSMA, LinfPGDAttack, L2PGDAttack

from utils import visualize, j_print, j_header
from colorshift import SVHAttack


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, default=0.5, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=300, help="batch size")  # was 1000
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=200, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")

parser.add_argument('-s', '--steps', default=512, type=int, help='#PGD steps')
parser.add_argument('-v', '--visualize', type=str2bool, default=True, help='Do you want to see the resulting images')
parser.add_argument('-d', '--duplicate_rgb', type=str2bool, default=True, help='Duplicate one Channel to RGB?')
parser.add_argument('--tv_lam', type=float, default=0.1, help='Max TV-Lambda')
parser.add_argument('--c_lam', type=float, default=20.0, help='Channels Lambda')
parser.add_argument('--s_lam', type=float, default=5.0, help='Sim Lambda')
args = parser.parse_args()

def get_conf(inx):
    model = base_classifier
    out = model.forward(inx.unsqueeze(0))
    return torch.nn.functional.softmax(out).max(), out.argmax()

def save(ad, name):
    visualize(ad, 'comparison', name)
    noise=ad-first_x
    torch.save(ad, name)
    print(name)
    print('linf=', noise.abs().max())
    print('l2=', noise.norm(p=2))
    print('l0=', noise.norm(p=0))
    print('l1=', noise.norm(p=1))
    print(get_conf(ad))
    noise=(noise+1.0) / 2.0 
    visualize(noise, 'comparison', 'noise_{}'.format(name))

if __name__ == "__main__":
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])
    smooth_attacker = SmoothAttack(base_classifier)

    dataset = get_dataset(args.dataset, args.split)
    (x, label) = dataset[4400]
    x = x.cuda()
    first_x = x.data
    visualize(first_x, 'comparison', 'nat')
    target = 463
    adv_x = smooth_attacker.perturb(x=first_x, y=target, eps=1.0, total_steps=args.steps,
                                     duplicate_rgb=args.duplicate_rgb, init_tv_lam=args.tv_lam, reg_lam=args.c_lam,
                                     sim_lam=args.s_lam, batch_size=1)
    save(adv_x, 'tvnew')
    exit(0)
    tt = torch.LongTensor([target]).cuda()
    linf = LinfPGDAttack(base_classifier, targeted=True, eps=0.03731).perturb(first_x.unsqueeze(0), tt)
    save(linf[0], 'linf')
    for i in range(100):
        svh = SVHAttack().perturb(x, None, 0.0)
        save(svh, 'svh{}'.format(i))
    # pdb.set_trace()
