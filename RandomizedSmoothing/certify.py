import torch
import numpy as np

from architectures import get_architecture
from core import Smooth
from utils.log import j_print, j_header
from utils.visualize import FigureSaver
from .attack import SmoothAttack
from .datasets import get_dataset, DATASETS, get_num_classes
from args import Args


def main():
    args = Args().get_args()
    checkpoint = torch.load(args.checkpoint)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])

    attacker = SmoothAttack(base_classifier)
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    dataset = get_dataset(args.dataset, 'test')
    average_nat = []
    average_adv = []

    j_header('index', 'nat_y', 'adv_y', 'nat_rad', 'adv_rad', 'success')
    figure = FigureSaver()
    for i in range(0, len(dataset), args.skip):
        (x, label) = dataset[i]
        x = x.cuda()
        first_x = x.data

        nat_pred, nat_rad = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        if nat_pred is -1:
            continue
        if args.dataset == DATASETS[0]:  # ImageNet
            targets = [j for j in range(0, 1000, 100) if j is not label]
        else:
            targets = [j for j in range(10) if j is not label]
        best_rad = -10.0
        best_image = None
        best_target = -1

        for target in targets:
            adv_x = attacker.perturb(x=first_x, y=target, eps=args.sigma, steps=args.steps,
                                     duplicate_rgb=args.duplicate_rgb, tv_lam=args.tv_lam, ch_lam=args.c_lam,
                                     dissim_lam=args.s_lam, batch=400)
            # If you want to do wasserstein attack, uncomment the following and change the attacker to wasserstein
            # adv_x = attacker.perturb(x=first_x, y=target, eps=args.sigma, steps=args.steps, batch=400)
            adv_pred, adv_rad = smoothed_classifier.certify(adv_x, args.N0, 2 * args.N0, args.alpha, args.batch)
            adv_suc = (adv_pred != label) and (adv_pred != -1) and (nat_pred != -1)
            adv_rad = adv_x if adv_suc else -adv_rad

            if adv_rad > best_rad:
                best_rad = adv_rad
                best_image = adv_x.data
                best_target = target

        figure.save(best_image, i, 'best={}'.format(best_target))
        figure.save(first_x, i, 'natural')
        best_pred, best_rad = smoothed_classifier.certify(best_image, args.N0, args.N, args.alpha, args.batch)
        j_print(i, label, best_target, nat_rad, best_rad)
        average_adv.append(best_rad)
        average_nat.append(nat_rad)
    average_nat = np.array(average_nat)
    average_adv = np.array(average_adv)
    print('Average nat radii {}, Average adv radii {}'.format(average_nat.mean(), average_adv.mean()))


if __name__ == "__main__":
    main()
