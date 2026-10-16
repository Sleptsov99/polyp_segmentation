import argparse
import os

import matplotlib
import numpy as np
import torch
import torch.nn as nn

import config
from dataset import get_dataloaders
from model import build_model
from train import validate


def load_model():
    if not os.path.exists(config.CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Чекпоинт не найден: {config.CHECKPOINT_PATH}\nСначала обучите модель: python train.py"
        )
    # Веса ImageNet не нужны — их сразу перезапишет чекпоинт
    model = build_model(encoder_weights=None)
    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def predict(model, image):
    with torch.no_grad():
        preds = model(image)
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()
    return preds


def denormalize(image_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image_tensor.permute(1, 2, 0).cpu().numpy()

    image = std * image + mean
    image = np.clip(image, 0, 1)
    return image


def visualize(model, loader, num_samples=4, show=False):
    import matplotlib.pyplot as plt

    images, masks = next(iter(loader))
    num_samples = min(num_samples, len(images))

    images = images.to(config.DEVICE)
    preds = predict(model, images)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples), squeeze=False)
    for i in range(num_samples):
        original = denormalize(images[i])
        real_mask = masks[i].squeeze().cpu().numpy()

        pred_mask = preds[i].squeeze().cpu().numpy()

        axes[i, 0].imshow(original)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(real_mask, cmap="gray")
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_mask, cmap="gray")
        axes[i, 2].set_title("Predicted Mask")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(config.RESULTS_PATH, dpi=150)
    print(f"Результаты сохранены в {config.RESULTS_PATH}")

    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Оценка модели на тестовой выборке и визуализация")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--show", action="store_true", help="открыть окно с картинкой")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")  # работает и без дисплея (сервер, Docker, Colab)

    _, _, test_loader = get_dataloaders()
    model = load_model()

    test_loss, test_dice = validate(model, test_loader, nn.BCEWithLogitsLoss(), desc="Test")
    print(f"Test Loss: {test_loss:.4f}  Test Dice: {test_dice:.4f}")

    visualize(model, test_loader, num_samples=args.num_samples, show=args.show)


if __name__ == "__main__":
    main()
