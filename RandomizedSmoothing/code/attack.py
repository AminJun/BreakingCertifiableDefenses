import os
import torch.nn
from torchvision.transforms import ToPILImage
from utils import j_print, j_header
from projected_sinkhorn import projected_sinkhorn, wasserstein_cost


def visualize(img: torch.Tensor, *path):
    to_pil = ToPILImage()
    pil = to_pil(img.cpu())
    path = [str(folder) for folder in path]
    child = os.path.join(*path[1:])
    os.makedirs(path[0], exist_ok=True)
    file_name = "{}/{}.png".format(path[0], child)
    pil.save(file_name)


class SmoothAttack:
    def __init__(self, classifier: torch.nn.Module):
        self.classifier = classifier
        self.target = 0
        self._loss = torch.nn.CrossEntropyLoss(reduction='mean').cuda()

    @staticmethod
    def get_acc(output, y):
        pred = output.argmax(1)
        return 100. * float(len(pred[pred == y])) / float(len(pred))

    @staticmethod
    def get_tv(t) -> torch.Tensor:
        x_wise = t[:, :, 1:] - t[:, :, :-1]
        y_wise = t[:, 1:, :] - t[:, :-1, :]
        return (x_wise * x_wise).sum() + (y_wise * y_wise).sum()

    @staticmethod
    def _repeat_size(batch_size: int, current_size: torch.Size):
        return (batch_size,) + tuple([1] * len(current_size))

    @staticmethod
    def _clamp_by_norm(p: torch.Tensor, eps: float, order: int = 2):
        return eps / torch.norm(p, order) * p

    def perturb(self, x: torch.Tensor, y: int, eps: float = 0.5, batch_size: int = 400, total_steps: int = 300,
                duplicate_rgb: bool = False, tv_eps: float = 0.1, init_tv_lam: float = 0.1, reg_lam: float = 20.0,
                sim_lam: float = 10.0) -> torch.Tensor:
        torch.manual_seed(6247423)

        t = torch.rand_like(x[0] if duplicate_rgb else x).cuda() - 0.5
        t.requires_grad_()

        copy_size = self._repeat_size(batch_size, x.size())
        x_batch = x.repeat(copy_size).cuda()
        noise = torch.randn_like(x_batch).cuda() * eps
        x_batch = x_batch + noise
        y = torch.LongTensor([y]).cuda().repeat((batch_size,))
        tv_lam = init_tv_lam

        for i in range(total_steps):
            ct = t.repeat((3, 1, 1)) if duplicate_rgb else t
            cur_in = x_batch + ct.repeat(copy_size)
            outputs = self.classifier(cur_in)
            c_loss = -torch.mean(self._loss(outputs, y))
            tv_loss = tv_lam * self.get_tv(ct)
            t_reg_loss = reg_lam * (ct.abs().mean([1, 2]).norm() ** 2)
            t_similarity = 0
            if not duplicate_rgb:
                t_similarity = sim_lam * ((t[0] - t[1]) ** 2 + (t[1] - t[2]) ** 2 + (t[0] - t[2]) ** 2).norm(p=2)

            loss = c_loss - tv_loss - t_reg_loss - t_similarity
            loss.backward()
            acc = self.get_acc(outputs, y)

            t.data = t.data + tv_eps * t.grad.data
            t.grad.data.zero_()

            j_print(i, acc, loss, c_loss, tv_loss / tv_lam, t_reg_loss / reg_lam, tv_lam)

        ct = t.repeat((3, 1, 1)) if duplicate_rgb else t
        return torch.clamp((x + ct).data, 0.0, 1.0)


class Wasserstein:
    def __init__(self, classifier: torch.nn.Module):
        self.classifier = classifier
        self.target = 0
        self._loss = torch.nn.CrossEntropyLoss(reduction='mean').cuda()

    @staticmethod
    def get_acc(output, y):
        pred = output.argmax(1)
        return 100. * float(len(pred[pred == y])) / float(len(pred))

    @staticmethod
    def _repeat_size(batch_size: int, current_size: torch.Size):
        return (batch_size,) + tuple([1] * len(current_size))

    @staticmethod
    def _clamp_by_norm(p: torch.Tensor, eps: float, order: int = 2):
        return eps / torch.norm(p, order) * p

    def perturb(self, x: torch.Tensor, y: int, eps: float = 0.5, batch_size: int = 400, total_steps: int = 300,
                alpha: float = 0.1, p: int = 2, kernel_size: int = 5, epsilon_factor: float = 1.17,
                epsilon: float = 0.01, regularization: int = 3000, sinkhorn_max_iter: int = 400, ) -> torch.Tensor:
        torch.manual_seed(6247423)

        t = torch.zeros_like(x).cuda()
        t.requires_grad_()
        normalization = x.sum()
        C = wasserstein_cost(x.unsqueeze(0), p=p, kernel_size=kernel_size)

        copy_size = self._repeat_size(batch_size, x.size())
        x_batch = x.repeat(copy_size).cuda()
        noise = torch.randn_like(x_batch).cuda() * eps
        x_batch = x_batch + noise
        y = torch.LongTensor([y]).cuda().repeat((batch_size,))
        best_noise = torch.zeros_like(x)
        best_acc = 0

        for i in range(total_steps):
            ct = t
            cur_in = x_batch + ct.repeat(copy_size)
            outputs = self.classifier(cur_in)
            loss = -torch.mean(self._loss(outputs, y))
            loss.backward()
            acc = self.get_acc(outputs, y)
            t.data = t.data + alpha * t.grad.data / t.grad.data.norm()
            # t.data = t.data + alpha * t.grad.data.sign() # / t.grad.data.norm()
            t.grad.data.zero_()
            pt = t + x
            pt = projected_sinkhorn(x.unsqueeze(0).clone() / normalization,
                                    pt.unsqueeze(0).detach() / normalization,
                                    C, epsilon, regularization, verbose=0,
                                    maxiters=sinkhorn_max_iter)[0] * normalization
            # import pdb; pdb.set_trace()
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
