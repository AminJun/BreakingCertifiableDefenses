import datetime
import pdb
import os

import torch
from torchvision.transforms import ToPILImage


class Shadow:
    def __init__(self, x, y, tv_lam, reg_lam, eps):
        self.t_shape = x[:, :1, :, :].shape
        self.t = None
        self.labels = torch.zeros_like(y)
        self.tv_lam = tv_lam
        self.reg_lam = reg_lam
        self.eps = eps
        self.log = ''

    def renew_t(self):
        t = torch.rand(size=self.t_shape).cuda() - 0.5
        t.requires_grad_()
        self.t = t

    def iterate_labels_not_equal_to(self, y):
        labels = self.labels
        self.labels = labels.cpu() + (labels.cpu() == y.cpu()).type(torch.int64)

    def get_ct(self):
        return self.t.repeat((1, 3, 1, 1))

    def back_prop(self, c_loss, rep):
        tv_lam = self.tv_lam
        reg_lam = self.reg_lam
        ct = self.get_ct()
        tv_loss = tv_lam * get_tv(ct)
        #t_reg_loss = reg_lam * (ct.mean([2, 3]).norm(2, 1) ** 2).sum()
        t_reg_loss = reg_lam * (ct.norm(p=2))
        loss = c_loss - tv_loss - t_reg_loss
        loss.backward()

        cur_tv_eps = get_tv_eps(rep, 300, self.eps)
        t = self.t
        t.data = t.data + cur_tv_eps * t.grad.data
        t.grad.data.zero_()

        if rep % 300 == 299:
             self.log = j_log(c_loss, tv_loss / (tv_lam*len(t)), t_reg_loss / (reg_lam * len(t)), loss)
        #    j_print(rep, c_loss, tv_loss / (tv_lam*len(t)), tv_loss, t_reg_loss / (reg_lam*len(t)), t_reg_loss, loss, )


def get_tv_eps(i: int, pgd_steps: int = 200, init_tv_eps: float = 0.01):
    if i < pgd_steps / 2.0:
        return init_tv_eps
    if i < pgd_steps * (5.0 / 6.0):
        return init_tv_eps / 3.0
    return init_tv_eps / 9.0


def get_tv(t: torch.Tensor) -> torch.Tensor:
    return smooth_tv(t[:, 0]) + smooth_tv(t[:, 1]) + smooth_tv(t[:, 2])


def get_args(read_input: bool = False, duplicate_rgb: bool = False):
    if read_input:
        return [float(input(var + '\n')) for var in ['tv_eps (0.2)', 'tv_lam (0.3)', 'reg_lam(1.0)']]
    if duplicate_rgb:
        return [200.5, 0.000009, 0.02]
    return [0.2, 0.0, 0.0, 0.0001]


def get_exp_name():
    tt = datetime.datetime.now()
    return '{}:{}'.format(tt.hour, tt.minute)


def _convert_form(f):
    if isinstance(f, torch.Tensor):
        f = f.item()
    if hasattr(f, '__round__') and 'bool' not in f.__class__.__name__:
        return round(f, 2)
    return f

def j_log(*args):
    array = [_convert_form(i) for i in args]
    plain = '{}\t' * len(args)
    return plain.format(*array)

def j_print(*args, file=None):
    array = [_convert_form(i) for i in args]
    j_header(*array, file=file)


def j_header(*args, file=None):
    plain = '{}\t' * len(args)
    print(plain.format(*args), flush=True, file=file)


def _tv_diff(c):
    x_wise = c[:, :, 1:] - c[:, :, :-1]
    y_wise = c[:, 1:, :] - c[:, :-1, :]
    return x_wise, y_wise


def smooth_tv(c) -> torch.Tensor:
    x_wise, y_wise = _tv_diff(c)
    return (x_wise * x_wise).sum() + (y_wise * y_wise).sum()


def visualize(img: torch.Tensor, *path):
    to_pil = ToPILImage()
    pil = to_pil(img.cpu())
    path = [str(folder) for folder in path]
    par = os.path.join(*path[:-1])
    os.makedirs(par, exist_ok=True)
    file_name = "{}/{}.png".format(par, path[-1])
    file_name_tf = "{}/{}.advc10".format(par, path[-1])
    pil.save(file_name)
    torch.save(img[:,:32,:32], file_name_tf)


def save_images(images: torch.Tensor, success, first_index, *path):
    for img, suc in zip(images, success):
        if suc:
            visualize(img, *path, first_index)
        first_index += 1


std_norm = torch.FloatTensor([0.2023, 0.1994, 0.2010]).reshape([1, 3, 1, 1]).cuda()
mean_norm = torch.FloatTensor([0.4914, 0.4822, 0.4465]).reshape([1, 3, 1, 1]).cuda()


def get_normal(ten: torch.FloatTensor) -> torch.FloatTensor:
    return (ten - mean_norm) / std_norm


def get_unit(ten: torch.FloatTensor) -> torch.FloatTensor:
    return (ten * std_norm) + mean_norm


def get_unit01(ten: torch.Tensor) -> torch.Tensor:
    return torch.clamp(get_unit(ten), 0., 1.)
