import time
import pdb

import numpy as np
import torch
import torch.nn

from bound_layers import BoundLinear, BoundConv2d
from convex_adversarial import DualNetwork
from shadow_attack import get_args, get_exp_name, get_normal, get_unit, Shadow, save_images, get_unit01
from train import AverageMeter


def attack(model, model_name, loader, start_eps, end_eps, max_eps, norm, logger, verbose, method, **kwargs):
    torch.manual_seed(6247423)
    num_class = 10
    losses = AverageMeter()
    l1_losses = AverageMeter()
    errors = AverageMeter()
    robust_errors = AverageMeter()
    regular_ce_losses = AverageMeter()
    robust_ce_losses = AverageMeter()
    relu_activities = AverageMeter()
    bound_bias = AverageMeter()
    bound_diff = AverageMeter()
    unstable_neurons = AverageMeter()
    dead_neurons = AverageMeter()
    alive_neurons = AverageMeter()
    batch_time = AverageMeter()
    # initial
    model.eval()
    duplicate_rgb = True
    # pregenerate the array for specifications, will be used for scatter
    sa = np.zeros((num_class, num_class - 1), dtype=np.int32)
    for i in range(sa.shape[0]):
        for j in range(sa.shape[1]):
            if j < i:
                sa[i][j] = j
            else:
                sa[i][j] = j + 1
    sa = torch.LongTensor(sa)
    total = len(loader.dataset)
    batch_size = loader.batch_size
    print(batch_size)
    std = torch.tensor(loader.std).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    total_steps = 300

    batch_eps = np.linspace(start_eps, end_eps, (total // batch_size) + 1)
    if end_eps < 1e-6:
        logger.log('eps {} close to 0, using natural training'.format(end_eps))
        method = "natural"

    exp_name = 'outputs/[{}:{}]'.format(get_exp_name(), model_name)
    # real_i = 0
    for i, (init_data, init_labels) in enumerate(loader):
        # labels = torch.zeros_like(init_labels)
        init_data = init_data.cuda()
        tv_eps, tv_lam, reg_lam = get_args(duplicate_rgb=duplicate_rgb)
        attacker = Shadow(init_data, init_labels, tv_lam, reg_lam, tv_eps)
        success = np.zeros(len(init_data))
        # saved_advs = torch.zeros_like(init_data).cuda()
        for t_i in range(9):

            attacker.iterate_labels_not_equal_to(init_labels)
            attacker.renew_t()
            labels = attacker.labels

            for rep in range(total_steps):
                ct = attacker.get_ct()
                data = init_data + ct
                data.data = get_normal(get_unit01(data))

                # ========================== The rest of code is taken from CROWN-IBP REPO
                start = time.time()
                eps = batch_eps[i]
                c = torch.eye(num_class).type_as(data)[labels].unsqueeze(1) - torch.eye(num_class).type_as(
                    data).unsqueeze(0)
                # remove specifications to self
                eye = (~(labels.data.unsqueeze(1) == torch.arange(num_class).type_as(labels.data).unsqueeze(0)))
                c = (c[eye].view(data.size(0), num_class - 1, num_class))
                # scatter matrix to avoid compute margin to self
                sa_labels = sa[labels]
                # storing computed lower bounds after scatter
                lb_s = torch.zeros(data.size(0), num_class)

                # FIXME: Assume data is from range 0 - 1
                if kwargs["bounded_input"]:
                    assert loader.std == [1, 1, 1] or loader.std == [1]
                    # bounded input only makes sense for Linf perturbation
                    assert norm == np.inf
                    data_ub = (data + eps).clamp(max=1.0)
                    data_lb = (data - eps).clamp(min=0.0)
                else:
                    if norm == np.inf:
                        data_ub = data.cpu() + (eps / std)
                        data_lb = data.cpu() - (eps / std)
                    else:
                        data_ub = data_lb = data

                if list(model.parameters())[0].is_cuda:
                    data = data.cuda()
                    data_ub = data_ub.cuda()
                    data_lb = data_lb.cuda()
                    labels = labels.cuda()
                    c = c.cuda()
                    sa_labels = sa_labels.cuda()
                    lb_s = lb_s.cuda()
                # convert epsilon to a tensor
                eps_tensor = data.new(1)
                eps_tensor[0] = eps

                # omit the regular cross entropy, since we use robust error
                output = model(data)
                regular_ce = torch.nn.CrossEntropyLoss()(output, labels)
                regular_ce_losses.update(regular_ce.cpu().detach().numpy(), data.size(0))
                errors.update(torch.sum(torch.argmax(output, dim=1) != labels).cpu().detach().numpy() / data.size(0),
                              data.size(0))
                # get range statistic

                if verbose or method != "natural":
                    if kwargs["bound_type"] == "convex-adv":
                        # Wong and Kolter's bound, or equivalently Fast-Lin
                        if kwargs["convex-proj"] is not None:
                            proj = kwargs["convex-proj"]
                            if norm == np.inf:
                                norm_type = "l1_median"
                            elif norm == 2:
                                norm_type = "l2_normal"
                            else:
                                raise (ValueError("Unsupported norm {} for convex-adv".format(norm)))
                        else:
                            proj = None
                            if norm == np.inf:
                                norm_type = "l1"
                            elif norm == 2:
                                norm_type = "l2"
                            else:
                                raise (ValueError("Unsupported norm {} for convex-adv".format(norm)))
                        if loader.std == [1] or loader.std == [1, 1, 1]:
                            convex_eps = eps
                        else:
                            convex_eps = eps / np.mean(loader.std)
                            # for CIFAR we are roughly / 0.2
                            # FIXME this is due to a bug in convex_adversarial, we cannot use per-channel eps
                        if norm == np.inf:
                            # bounded input is only for Linf
                            if kwargs["bounded_input"]:
                                # FIXME the bounded projection in convex_adversarial has a bug, data range must be positive
                                data_l = 0.0
                                data_u = 1.0
                            else:
                                data_l = -np.inf
                                data_u = np.inf
                        else:
                            data_l = data_u = None
                        f = DualNetwork(model, data, convex_eps, proj=proj, norm_type=norm_type,
                                        bounded_input=kwargs["bounded_input"], data_l=data_l, data_u=data_u)
                        lb = f(c)
                    elif kwargs["bound_type"] == "interval":
                        ub, lb, relu_activity, unstable, dead, alive = model.interval_range(norm=norm, x_U=data_ub,
                                                                                            x_L=data_lb,
                                                                                            eps=eps, C=c)
                    elif kwargs["bound_type"] == "crown-interval":
                        ub, ilb, relu_activity, unstable, dead, alive = model.interval_range(norm=norm, x_U=data_ub,
                                                                                             x_L=data_lb, eps=eps, C=c)
                        crown_final_factor = kwargs['final-beta']
                        factor = (max_eps - eps * (1.0 - crown_final_factor)) / max_eps
                        if factor < 1e-5:
                            lb = ilb
                        else:
                            if kwargs["runnerup_only"]:
                                masked_output = output.detach().scatter(1, labels.unsqueeze(-1), -100)
                                runner_up = masked_output.max(1)[1]
                                runnerup_c = torch.eye(num_class).type_as(data)[labels]
                                runnerup_c.scatter_(1, runner_up.unsqueeze(-1), -1)
                                runnerup_c = runnerup_c.unsqueeze(1).detach()
                                clb, bias = model.backward_range(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c)
                                clb = clb.expand(clb.size(0), num_class - 1)
                            else:
                                clb, bias = model.backward_range(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c)
                                bound_bias.update(bias.sum() / data.size(0))
                            diff = (clb - ilb).sum().item()
                            bound_diff.update(diff / data.size(0), data.size(0))
                            lb = clb * factor + ilb * (1 - factor)
                    else:
                        raise RuntimeError("Unknown bound_type " + kwargs["bound_type"])

                    lb = lb_s.scatter(1, sa_labels, lb)
                    robust_ce = torch.nn.CrossEntropyLoss()(-lb, labels)
                    if kwargs["bound_type"] != "convex-adv":
                        relu_activities.update(relu_activity.detach().cpu().item() / data.size(0), data.size(0))
                        unstable_neurons.update(unstable / data.size(0), data.size(0))
                        dead_neurons.update(dead / data.size(0), data.size(0))
                        alive_neurons.update(alive / data.size(0), data.size(0))

                if method == "robust":
                    loss = robust_ce
                elif method == "robust_activity":
                    loss = robust_ce + kwargs["activity_reg"] * relu_activity
                elif method == "natural":
                    loss = regular_ce
                elif method == "robust_natural":
                    natural_final_factor = kwargs["final-kappa"]
                    kappa = (max_eps - eps * (1.0 - natural_final_factor)) / max_eps
                    loss = (1 - kappa) * robust_ce + kappa * regular_ce
                else:
                    raise ValueError("Unknown method " + method)

                if "l1_reg" in kwargs:
                    reg = kwargs["l1_reg"]
                    l1_loss = 0.0
                    for name, param in model.named_parameters():
                        if 'bias' not in name:
                            l1_loss = l1_loss + (reg * torch.sum(torch.abs(param)))
                    loss = loss + l1_loss
                    l1_losses.update(l1_loss.cpu().detach().numpy(), data.size(0))

                # =========================================== The rest is from breaking paper not from CROWN-IBP Repo
                c_loss = -loss
                attacker.back_prop(c_loss, rep)

                batch_time.update(time.time() - start)
                losses.update(loss.cpu().detach().numpy(), data.size(0))

                if (verbose or method != "natural") and rep == total_steps - 1:
                    robust_ce_losses.update(robust_ce.cpu().detach().numpy(), data.size(0))
                    certified = (lb < 0).any(dim=1).cpu().numpy()
                    success = success + np.ones(len(success)) - certified
                    # saved_advs[certified == False] = data[certified == False].data
            torch.cuda.empty_cache()
            to_print='{}\t{}\t{}'.format((success > 0).sum(), t_i, attacker.log)
            print(to_print, flush=True)
            attacker.labels = attacker.labels + 1
        # save_images(get_unit01(torch.cat((saved_advs, init_data), dim=-1)), success.astype(np.bool), real_i, exp_name)
        # real_i += len(saved_advs)
        robust_errors.update((success > 0).sum() / len(success), len(success))
        print('====', robust_errors.avg, '===', flush=True)
    for i, l in enumerate(model):
        if isinstance(l, BoundLinear) or isinstance(l, BoundConv2d):
            norm = l.weight.data.detach().view(l.weight.size(0), -1).abs().sum(1).max().cpu()
            logger.log('layer {} norm {}'.format(i, norm))
    if method == "natural":
        return errors.avg, errors.avg
    else:
        return robust_errors.avg, errors.avg
