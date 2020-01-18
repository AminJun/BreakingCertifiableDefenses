from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import torch.nn
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from torch.nn import CrossEntropyLoss
import numpy as np

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
