"""E7a ResNet U-Net and Dice + BCE loss."""

import torch
from torch.nn import functional as F


def build_model(encoder="resnet34", weights="imagenet"):
    import segmentation_models_pytorch as smp

    model = smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=1, classes=1)
    if weights is not None:
        from torchvision.models import ResNet18_Weights, ResNet34_Weights

        if weights != "imagenet" or encoder not in ("resnet18", "resnet34"):
            raise ValueError("Only ImageNet ResNet18/34 initialization is supported")
        source = ResNet18_Weights if encoder == "resnet18" else ResNet34_Weights
        state = torch.hub.load_state_dict_from_url(
            source.IMAGENET1K_V1.url, model_dir="cache/weights", check_hash=True, weights_only=True
        )
        state["conv1.weight"] = state["conv1.weight"].sum(1, keepdim=True)
        state.pop("fc.weight")
        state.pop("fc.bias")
        model.encoder.load_state_dict(state)
    return model


def dice_bce(logits, target):
    logits, target = logits.float(), target.float()
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
