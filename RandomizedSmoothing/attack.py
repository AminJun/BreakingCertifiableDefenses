import torch.nn
from utils.log import j_print, j_header
from utils.regularizers import get_tv, get_sim, get_color
from utils.classification import get_acc
from projected_sinkhorn import projected_sinkhorn, wasserstein_cost


class SmoothAttack:
    def __init__(self, classifier: torch.nn.Module):
        self.classifier = classifier
        self._loss = torch.nn.CrossEntropyLoss(reduction='mean').cuda()

    def perturb(self, x: torch.Tensor, y: int, eps: float = 0.5, batch: int = 400, steps: int = 300,
                duplicate_rgb: bool = False, lr: float = 0.1, tv_lam: float = 0.1, ch_lam: float = 20.0,
                dissim_lam: float = 10.0, print_stats: bool = False, **_) -> torch.Tensor:
        print('Ignored args are: ', _)
        torch.manual_seed(6247423)

        t = torch.rand_like(x[0] if duplicate_rgb else x).cuda() - 0.5
        t.requires_grad_()

        copy_size = (batch, 1, 1, 1)
        x_batch = x.repeat(copy_size).cuda()
        x_batch = x_batch + torch.randn_like(x_batch).cuda() * eps
        y = torch.LongTensor([y]).cuda().repeat((batch,))

        if print_stats:
            j_header('step', 'acc', 'loss', 'cls', 'tv', 'col', 'dissim')
        for i in range(steps):
            ct = t.repeat((3, 1, 1)) if duplicate_rgb else t
            cur_in = x_batch + ct.repeat(copy_size)
            outputs = self.classifier(cur_in)
            acc, correct = get_acc(outputs, y)

            cl_loss = -torch.mean(self._loss(outputs, y))
            tv_loss = get_tv(ct)
            col_loss = get_color(ct)
            dissim_loss = 0 if duplicate_rgb else get_sim(ct)
            loss = cl_loss - tv_lam * tv_loss - ch_lam * col_loss - dissim_lam * dissim_loss
            loss.backward()

            t.data = t.data + lr * t.grad.data
            t.grad.data.zero_()

            if print_stats:
                j_print(i, acc, loss, cl_loss, tv_loss, col_loss, dissim_loss)

        ct = t.repeat((3, 1, 1)) if duplicate_rgb else t
        return torch.clamp((x + ct).data, 0.0, 1.0)


class Wasserstein:
    def __init__(self, classifier: torch.nn.Module):
        self.classifier = classifier
        self._loss = torch.nn.CrossEntropyLoss(reduction='mean').cuda()

    def perturb(self, x: torch.Tensor, y: int, eps: float = 0.5, batch_size: int = 400, total_steps: int = 300,
                alpha: float = 0.1, p: int = 2, kernel_size: int = 5, epsilon_factor: float = 1.17,
                epsilon: float = 0.01, regularization: int = 3000, sinkhorn_max_iter: int = 400, ) -> torch.Tensor:
        torch.manual_seed(6247423)

        t = torch.zeros_like(x).cuda()
        t.requires_grad_()
        normalization = x.sum()
        wass = wasserstein_cost(x.unsqueeze(0), p=p, kernel_size=kernel_size)

        copy_size = (batch_size, 1, 1, 1)
        x_batch = x.repeat(copy_size).cuda()
        noise = torch.randn_like(x_batch).cuda() * eps
        x_batch = x_batch + noise
        y = torch.Tensor([y]).cuda().repeat((batch_size,))
        best_noise = torch.zeros_like(x)
        best_acc = 0

        for i in range(total_steps):
            ct = t
            cur_in = x_batch + ct.repeat(copy_size)
            outputs = self.classifier(cur_in)
            loss = -torch.mean(self._loss(outputs, y))
            loss.backward()
            acc, correct = get_acc(outputs, y)
            t.data = t.data + alpha * t.grad.data / t.grad.data.norm()
            # t.data = t.data + alpha * t.grad.data.sign() # / t.grad.data.norm()
            t.grad.data.zero_()
            pt = t + x
            pt = projected_sinkhorn(x.unsqueeze(0).clone() / normalization,
                                    pt.unsqueeze(0).detach() / normalization,
                                    wass, epsilon, regularization, verbose=0,
                                    maxiters=sinkhorn_max_iter)[0] * normalization
            t.data = pt.data - x.data
            j_print(i, acc, loss, t.abs().max(), epsilon)
            if i % 10 == 9:
                epsilon *= epsilon_factor
            if acc > best_acc:
                best_acc = acc
                best_noise.data = t.data
            if acc == 100.0:
                break

        ct = t
        return torch.clamp((x + best_noise).data, 0.0, 1.0)
