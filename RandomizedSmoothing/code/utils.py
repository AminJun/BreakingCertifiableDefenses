import os

import torch
from torchvision.transforms import ToPILImage


def _convert_form(f):
    if isinstance(f, torch.Tensor):
        f = f.item()
    if hasattr(f, '__round__') and 'bool' not in f.__class__.__name__:
        return round(f, 2)
    return f


def j_print(*args, file=None):
    array = [_convert_form(i) for i in args]
    j_header(*array, file=file)


def j_header(*args, file=None):
    plain = '{}\t' * len(args)
    print(plain.format(*args), flush=True, file=file)


def visualize(img: torch.Tensor, *path):
    path = [str(folder) for folder in path]
    par = os.path.join(*path[:-1])
    os.makedirs(par, exist_ok=True)
    torch.save(img.cpu(), '{}/{}.trc'.format(par, path[-1]))
    to_pil = ToPILImage()
    pil = to_pil(img.cpu())
    file_name = "{}/{}.png".format(par, path[-1])
    pil.save(file_name)
