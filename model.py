import torch.nn
import segmentation_models_pytorch as smp

import config

def build_model():
    model=smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )
    return model.to(config.DEVICE)