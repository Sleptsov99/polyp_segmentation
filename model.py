import segmentation_models_pytorch as smp

import config


def build_model(encoder_weights=config.ENCODER_WEIGHTS):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,
    )
    return model.to(config.DEVICE)
