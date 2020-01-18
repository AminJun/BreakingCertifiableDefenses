import torch
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from torch.nn import CrossEntropyLoss

from utils import j_print, j_header


def _tv_diff(c):
    x_wise = c[:, 1:] - c[:, :-1]
    y_wise = c[1:, :] - c[:-1, :]
    return x_wise, y_wise


def anisotropic_tv(c):
    x_wise, y_wise = _tv_diff(c)
    return torch.norm(x_wise.view(-1), 1) + torch.norm(y_wise.view(-1), 1)


def isotropic_tv(c) -> torch.Tensor:
    epsilon = .05
    eps2 = epsilon ** 2
    x_wise, y_wise = _tv_diff(c)
    return (x_wise * x_wise + eps2).sqrt().sum() + (y_wise * y_wise + eps2).sqrt().sum()


class SVHAttack:
    def __init__(self, ):
        pass

    def perturb(self, x: torch.Tensor = None, x_hvc: np.array = None, saturation_bound: float = 0.0) -> torch.Tensor:
        if x_hvc is None:
            x_numpy = x.cpu().numpy()
            x_chanel_last = np.rollaxis(x_numpy, 0, 3)
            x_hvc = rgb_to_hsv(x_chanel_last)
            x_adv_hsv = x_hvc
        else:
            x_adv_hsv = np.copy(x_hvc)

        d_h = np.random.uniform(0, 1, size=(1,))
        d_s = np.random.uniform(-1, 1, size=(1,)) * saturation_bound

        x_adv_hsv[:, :, 0] = (x_hvc[:, :, 0] + d_h) % 1.0
        x_adv_hsv[:, :, 1] = np.clip(x_hvc[:, :, 1] + d_s, 0., 1.)
        tx = hsv_to_rgb(x_adv_hsv)
        channel_first = np.rollaxis(np.clip(tx, 0., 1.), 2, 0)
        tx = torch.from_numpy(channel_first)
        return tx.cuda()


class CompoundAttack:
    def __init__(self, classifier: torch.nn.Module):
        self.classifier = classifier
        self.target = 0
        self._loss = CrossEntropyLoss(reduction='mean').cuda()

    @staticmethod
    def get_tv(t, isotropic: bool = False) -> torch.Tensor:
        tv_func = anisotropic_tv
        if isotropic:
            tv_func = isotropic_tv
        return tv_func(t[0]) + tv_func(t[1]) + tv_func(t[2])

    @staticmethod
    def _repeat_size(batch_size: int, current_size: torch.Size):
        return (batch_size,) + tuple([1] * len(current_size))

    @staticmethod
    def _clamp_by_norm(p: torch.Tensor, eps: float, order: int = 2):
        return eps / torch.norm(p, order) * p

    def loss(self, output: torch.FloatTensor, y: torch.Tensor, targeted: bool = False) -> torch.FloatTensor:
        if not targeted:
            return self._loss(output, y)
        pred = output.argmax(-1)
        others = pred[pred != y]
        if others.shape[0] > 0:
            target = others.mode()[0]
        else:
            target = self.target
        self.target = target
        return -self._loss(output, target.repeat(y.shape[0]))

    @staticmethod
    def get_tv_eps(t: torch.Tensor, tv_eps: float, adjust: bool = False, method: str = 'max'):
        if not adjust:
            return tv_eps
        xv = (t[:, :, 1:] - t[:, :, :-1]).abs()
        yv = (t[:, 1:, :] - t[:, :-1, :]).abs()
        red_x = getattr(xv, method)
        red_y = getattr(yv, method)
        ret = (max(red_x().item(), red_y().item()) / 100.0 + 0.0002) / 2.0
        return ret

    def perturb(self, x: torch.Tensor, y: int, batch_size: int = 512, eps: float = 0.5, tv_eps: float = 0.0001,
                pgd_steps: int = 200, isotropic: bool = False, targeted: bool = False, adjust_tv_eps: bool = True) -> \
            (torch.Tensor, torch.Tensor):
        p = torch.zeros_like(x).cuda()
        t = torch.rand_like(x).cuda() - 0.5
        p.requires_grad_()
        t.requires_grad_()

        copy_size = self._repeat_size(batch_size, p.size())
        x_batch = x.repeat(copy_size).cuda()
        noise = torch.randn_like(x_batch).cuda() * eps
        x_batch = x_batch + noise
        y = torch.LongTensor([y]).cuda().repeat((batch_size,))

        for i in range(pgd_steps):
            cur_in = torch.clamp(x_batch + (p + t).repeat(copy_size), min=0.0, max=1.0)
            outputs = self.classifier(cur_in)
            c_loss = self.loss(outputs, y, targeted)
            loss = c_loss - self.get_tv(t, isotropic)
            loss.backward()

            p.data = self._clamp_by_norm(p.data + p.grad.data, eps)
            p.grad.data.zero_()

            cur_tv_eps = self.get_tv_eps(t, tv_eps, adjust_tv_eps)
            if i % 4 == 0:
                cur_tv_eps /= 2.0
            t.data = torch.clamp(t.data + torch.clamp(t.grad.data, -cur_tv_eps, cur_tv_eps), -1.0, 1.0)
            t.grad.data.zero_()
            print('{}\t{}\t{}\t{}\t{}\t{}'.format(i, round(loss.item(), 6),
                                                  round(self.get_tv(t, isotropic).item(), 2),
                                                  torch.norm(p, 2).item(), round(cur_tv_eps, 6),
                                                  round(c_loss.item(), 3)), flush=True)

        torch.cuda.empty_cache()
        return p.data, t.data


class RGBAttack:
    @staticmethod
    def get_acc(output, y):
        pred = output.argmax(1)
        return 100. * float(len(pred[pred == y])) / float(len(pred))

    def __init__(self, classifier):
        self.classifier = classifier
        self._loss = CrossEntropyLoss(reduction='none').cuda()

    @staticmethod
    def _repeat_size(batch_size: int, current_size: torch.Size):
        return (batch_size,) + tuple([1] * len(current_size))

    def classification_loss(self, pred: torch.Tensor, target: torch.Tensor = None, maximize: bool = True):
        sgn = 1 if maximize else -1
        instance_based = self._loss(pred, target)
        if maximize:
            instance_based = torch.clamp(instance_based, max=6.0)
        return sgn * torch.mean(instance_based)

    @staticmethod
    def _clamp_by_norm(p: torch.Tensor, eps: float, order: int = 2):
        return eps / torch.norm(p, order) * p

    def perturb(self, x: torch.Tensor, y: int, eps: float = 0.5, batch_size: int = 1024, total_steps: int = 300,
                regularize: bool = True, rgb_lambda: float = 10.0, rgb_eps: float = (1.0 / 256.0),
                maximize: bool = True, p_eps: float = 0.0) -> torch.Tensor:
        copy_size = self._repeat_size(batch_size, x.size())
        x_batch = x.repeat(copy_size).cuda()
        noise = torch.randn_like(x_batch).cuda() * eps
        x_batch = x_batch + noise
        y = torch.LongTensor([y]).cuda().repeat((batch_size,))

        rgb_arr = torch.FloatTensor([[0], [0], [0]]).cuda().requires_grad_()
        rgb = torch.zeros_like(x).cuda().requires_grad_()
        if regularize:
            small_bound = 0.1
            for tensor in rgb_arr:
                tensor += np.random.uniform(-small_bound, small_bound)

        p = torch.zeros_like(x).cuda()
        p.requires_grad_()

        j_header('id', 'acc%', 'xent', 'reg', 'r', 'g', 'b')
        for i in range(total_steps):
            rgb.data = rgb_arr.repeat((1, rgb[0].numel())).view(rgb.shape)
            cur_in = x_batch + rgb.repeat(copy_size)
            if p_eps:
                cur_in = cur_in + p.repeat(copy_size)

            outputs = self.classifier(cur_in)
            acc = self.get_acc(outputs, y)
            cl_loss = self.classification_loss(outputs, target=y, maximize=maximize)
            reg = rgb_lambda * torch.norm(torch.mean(rgb, (1, 2)), p=2)
            loss = cl_loss - reg if regularize else cl_loss
            loss.backward()

            j_print(i, acc, cl_loss, reg, rgb_arr[0], rgb_arr[1], rgb_arr[2], )
            if p_eps:
                p_grad = self._clamp_by_norm(p.grad.data, 0.05, 2)
                p.data = self._clamp_by_norm(p.data + p_grad, p_eps)
                p.grad.data.zero_()

            rgb_grad = torch.clamp(torch.mean(rgb.grad.data, (1, 2)), -rgb_eps, rgb_eps)
            rgb_arr.data = torch.clamp(rgb_arr.data + rgb_grad.view(rgb_arr.shape), -1.0, 1.0)
            rgb.grad.data.zero_()
        torch.cuda.empty_cache()
        return torch.clamp(x + rgb + p, 0.0, 1.0)
