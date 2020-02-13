import torch


def get_tv(t: torch.Tensor) -> torch.Tensor:
    x_wise = t[:, :, 1:] - t[:, :, :-1]
    y_wise = t[:, 1:, :] - t[:, :-1, :]
    return (x_wise * x_wise).sum() + (y_wise * y_wise).sum()


def get_sim(t: torch.Tensor) -> torch.Tensor:
    return ((t[0] - t[1]) ** 2 + (t[1] - t[2]) ** 2 + (t[0] - t[2]) ** 2).norm(p=2)


def get_color(t: torch.Tensor) -> torch.Tensor:
    return t.abs().mean([1, 2]).norm() ** 2
