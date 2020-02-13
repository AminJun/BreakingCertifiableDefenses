import torch


def get_acc(output: torch.Tensor, y: torch.Tensor) -> (float, int):
    pred = output.argmax(1)
    correct = len(pred[pred == y])
    return 100. * float(correct) / float(len(pred)), correct
