import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

import config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_transforms(is_train: bool):
    if is_train:
        return A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])


class PolypDataset(Dataset):

    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            raise RuntimeError(f"Не удалось прочитать изображение: {self.image_paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Не удалось прочитать маску: {self.mask_paths[idx]}")
        mask = (mask > 127).astype(np.float32)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)
        return image, mask


def _list_images(directory):
    """Словарь {имя файла без расширения: полный путь}, только файлы изображений."""
    files = {}
    for name in os.listdir(directory):
        stem, ext = os.path.splitext(name)
        if ext.lower() in IMAGE_EXTENSIONS:
            files[stem] = os.path.join(directory, name)
    return files


def get_image_mask_pairs():
    """Сопоставляет снимки и маски по имени файла (а не по порядку в папке)."""
    for directory in (config.IMAGE_DIR, config.MASK_DIR):
        if not os.path.isdir(directory):
            raise FileNotFoundError(
                f"Папка не найдена: {directory}\n"
                f"Скачайте датасет командой `python download_data.py` "
                f"или укажите путь к Kvasir-SEG через переменную окружения DATA_DIR."
            )

    images = _list_images(config.IMAGE_DIR)
    masks = _list_images(config.MASK_DIR)

    common = sorted(images.keys() & masks.keys())
    if not common:
        raise RuntimeError(f"Не найдено ни одной пары снимок/маска в {config.DATA_DIR}")

    unmatched = (images.keys() | masks.keys()) - set(common)
    if unmatched:
        print(f"Предупреждение: {len(unmatched)} файлов без пары пропущены, "
              f"например: {sorted(unmatched)[:5]}")

    return [images[k] for k in common], [masks[k] for k in common]


def get_splits():
    """Детерминированное разбиение на train / val / test."""
    image_paths, mask_paths = get_image_mask_pairs()

    train_imgs, rest_imgs, train_masks, rest_masks = train_test_split(
        image_paths, mask_paths,
        test_size=config.VAL_SPLIT + config.TEST_SPLIT,
        random_state=config.SEED,
    )
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(
        rest_imgs, rest_masks,
        test_size=config.TEST_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT),
        random_state=config.SEED,
    )
    return {
        "train": (train_imgs, train_masks),
        "val":   (val_imgs, val_masks),
        "test":  (test_imgs, test_masks),
    }


def _make_loader(paths, is_train):
    imgs, masks = paths
    dataset = PolypDataset(imgs, masks, get_transforms(is_train=is_train))
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=is_train,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.DEVICE == "cuda",
    )


def get_dataloaders():
    """Возвращает (train_loader, val_loader, test_loader)."""
    splits = get_splits()
    print(f"Снимков: train={len(splits['train'][0])}, "
          f"val={len(splits['val'][0])}, test={len(splits['test'][0])}")
    return (
        _make_loader(splits["train"], is_train=True),
        _make_loader(splits["val"], is_train=False),
        _make_loader(splits["test"], is_train=False),
    )
