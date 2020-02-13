import torch
import os
from utils.experiment import get_exp_name
from torchvision.transforms import ToPILImage


class FigureSaver:
    def __init__(self, folder: str = ''):
        self.folder = get_exp_name() + folder
        self._id = 0

    def save(self, t: torch.Tensor, index: int = 0, *path):
        to_pil = ToPILImage()
        pil = to_pil(t.cpu())
        path = [str(folder) for folder in path]
        child = '{}_{}'.format(self._id, '_'.join(path))
        par = os.path.join('desktop', self.folder, str(index))
        os.makedirs(par, exist_ok=True)
        file_name = "{}/{}.png".format(par, child)
        pil.save(file_name)
        self._id += 1
