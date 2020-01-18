#import setGPU
import argparse
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
from architectures import get_architecture
from attack import SmoothAttack
import numpy as np
from advertorch.attacks import JSMA

from utils import visualize, j_print, j_header

start_index = {
	0.12: 5820, 
	0.25: 6260, 
	0.50: 7060, 
	1.00: 8300,
}

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
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")

parser.add_argument('-s', '--steps', default=512, type=int, help='#PGD steps')
parser.add_argument('-v', '--visualize', type=str2bool, default=True, help='Do you want to see the resulting images')
parser.add_argument('-d', '--duplicate_rgb', type=str2bool, default=True, help='Duplicate one Channel to RGB?')
parser.add_argument('--tv_lam', type=float, default=0.1, help='Max TV-Lambda')
parser.add_argument('--c_lam', type=float, default=20.0, help='Channels Lambda')
parser.add_argument('--s_lam', type=float, default=5.0, help='Sim Lambda')
args = parser.parse_args()

if __name__ == "__main__":
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])

    attacker = SmoothAttack(base_classifier)
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    f = open(args.outfile, 'w')
    j_header('idx', 'label', 'target', 'nat_lbl', 'adv_lbl', 'nat_rad', 'adv_rad', 'adv_suc', 'success', 'time', file=f)

    dataset = get_dataset(args.dataset, args.split)
    average_nat = []
    average_adv = []
    # for i in range(start_index[args.sigma], len(dataset), args.skip):
    for i in range(0, len(dataset), args.skip):
        print('attack', i)
        (x, label) = dataset[i]
        x = x.cuda()
        first_x = x.data
        before_time = time()

        nat_pred, nat_rad = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        if nat_pred is -1:
            continue
        if args.dataset == DATASETS[0]:  # ImageNet
            targets = [j for j in range(0, 1000, 100) if j is not label]
            targets = [446]
        else:
            targets = [j for j in range(10) if j is not label]
        best_rad = -10.0
        best_image = None
        best_target = -1

        c_name = 'shadow_{}_{}_{}_{}_{}_{}_{}'.format(args.sigma, args.tv_lam, args.c_lam, args.steps, args.batch, i,
                                               args.s_lam)
        for target in targets:
            adv_x = attacker.perturb(x=first_x, y=target, eps=args.sigma, total_steps=args.steps,
                                     duplicate_rgb=args.duplicate_rgb, init_tv_lam=args.tv_lam, reg_lam=args.c_lam,
                                     sim_lam=args.s_lam, batch_size=400)
            adv_pred, adv_rad = smoothed_classifier.certify(adv_x, args.N0, 2 * args.N0, args.alpha, args.batch)
            adv_suc = (adv_pred != label) and (adv_pred != -1) and (nat_pred != -1)
            success = adv_suc and (adv_rad >= nat_rad)
            if not adv_suc:
                adv_rad = -adv_rad

            if adv_rad > best_rad:
                best_rad = adv_rad
                best_image = adv_x.data
                best_target = target

        visualize(best_image, 'figures', c_name, i, 'adv')
        visualize(first_x, 'figures', c_name, i, 'natural')
        best_pred, best_rad = smoothed_classifier.certify(best_image, args.N0, args.N, args.alpha, args.batch)
        j_print(best_rad, nat_rad, best_target, i, )
        j_print(best_rad, nat_rad, best_target, i, file=f)
    f.close()


