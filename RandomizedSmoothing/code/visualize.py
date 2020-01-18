from PIL import Image
from torchvision.transforms import ToPILImage
import torchvision.transforms.functional as TF
import os
import torch


def get_image(file: str):
    if os.path.isfile(file):
        image = Image.open(file)
        x = TF.to_tensor(image)
        return x
    return torch.zeros([3, 32, 32])


def read_images(path: list):
    back = torch.zeros((3, 32 * 10, 32))
    for i in range(10):
        file = os.path.join(*path, '{}.png'.format(i))
        if os.path.isfile(file):
            image = Image.open(file)
            x = TF.to_tensor(image)
            x.unsqueeze_(0)
            # x = x[0, :, 32:]
            # pdb.set_trace()
            slice_i = i
            back[:, slice_i * 32:(slice_i + 1) * 32] = x
    save_fig(back, path[1])


def save_fig(im: torch.Tensor, path: str = '', par: str = 'output_images'):
    to_pil = ToPILImage()
    pil = to_pil(im)
    os.makedirs(par, exist_ok=True)
    file_name = "{}/{}.png".format(par, path)
    pil.save(file_name)


def ablation_step_folder(path: list):
    folder = os.path.join(*path)
    black = torch.zeros([3, 32, 32 * 11])
    orig = get_image(os.path.join(folder, 'original.png'))
    black[:, :, 0:32] = orig
    folders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
    for i, f in enumerate(folders):
        img = get_image(os.path.join(folder, f, 'best.png'))
        black[:, :, 32 * (i + 1): 32 * (i + 2)] = img
    return black


def ablation_steps(path: list):
    par_dir = os.path.join(*path)
    black = torch.zeros([3, 32 * len(os.listdir(par_dir)), 11 * 32])
    for i, p in enumerate(os.listdir(par_dir)):
        img = ablation_step_folder([par_dir, p])
        save_fig(img, 'ablation_{}.png'.format(p))
        black[:, i * 32:(i + 1) * 32] = img
    save_fig(black, 'ablation_folder_{}.png'.format(path[0]))


def concat_images():
    par_dir = 'new_images'
    for p in os.listdir(par_dir):
        read_images([par_dir, p])


if __name__ == '__main__':
    # concat_images()
    ablation_steps(['figures', 'DiffSteps'])
