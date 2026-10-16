import torch


def dice_coefficient(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    return (2.0 * intersection) / (pred.sum() + target.sum() + 1e-8)
