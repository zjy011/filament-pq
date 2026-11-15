"""E7a ResNet U-Net and Dice + BCE loss."""

import torch
from torch.nn import functional as F


def build_model(encoder="resnet34", weights="imagenet"):
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=1, classes=1)


def dice_bce(logits, target):
    probability = logits.sigmoid()
    axes = (1, 2, 3)
    intersection = (probability * target).sum(axes)
    dice = (2 * intersection + 1) / (probability.sum(axes) + target.sum(axes) + 1)
    return F.binary_cross_entropy_with_logits(logits, target) + (1 - dice).mean()


def select_device(name="auto"):
    return (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if name == "auto"
        else torch.device(name)
    )
