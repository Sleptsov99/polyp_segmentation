import inspect
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import mlflow
import mlflow.pytorch

import config
from dataset import get_dataloaders
from metrics import dice_coefficient
from model import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    total_dice = 0
    for images, masks in tqdm(loader, desc="Train"):
        images = images.to(config.DEVICE)
        masks = masks.to(config.DEVICE)

        optimizer.zero_grad()
        pred = model(images)
        loss = criterion(pred, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_dice += dice_coefficient(pred, masks).item()
    n = len(loader)
    return total_loss / n, total_dice / n


def validate(model, loader, criterion, desc="Val"):
    model.eval()
    total_loss = 0
    total_dice = 0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc=desc):
            images = images.to(config.DEVICE)
            masks = masks.to(config.DEVICE)

            preds = model(images)
            loss = criterion(preds, masks)

            total_loss += loss.item()
            total_dice += dice_coefficient(preds, masks).item()
    n = len(loader)
    return total_loss / n, total_dice / n


def train():
    set_seed(config.SEED)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(config.MLFLOW_EXPERIMENT, artifact_location=config.MLFLOW_ARTIFACT_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    # Загружаем данные до старта run, чтобы при отсутствии датасета не оставался пустой запуск
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Устройство: {config.DEVICE}")

    with mlflow.start_run(run_name=config.MLFLOW_RUN_NAME):
        mlflow.log_params({
            "image_size":    config.IMAGE_SIZE,
            "batch_size":    config.BATCH_SIZE,
            "num_epochs":    config.NUM_EPOCHS,
            "learning_rate": config.LEARNING_RATE,
            "val_split":     config.VAL_SPLIT,
            "test_split":    config.TEST_SPLIT,
            "seed":          config.SEED,
            "encoder":       "resnet34",
            "encoder_weights": str(config.ENCODER_WEIGHTS),
            "architecture":  "U-Net",
        })

        model     = build_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss()
        best_dice = -1.0

        for epoch in range(1, config.NUM_EPOCHS + 1):
            train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss,   val_dice   = validate(model, val_loader, criterion)

            print(f"Epoch {epoch}/{config.NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f}  Train Dice: {train_dice:.4f} | "
                  f"Val Loss: {val_loss:.4f}  Val Dice: {val_dice:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_dice": train_dice,
                "val_loss":   val_loss,
                "val_dice":   val_dice,
            }, step=epoch)

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(model.state_dict(), config.CHECKPOINT_PATH)
                print(f"  --> Сохранена лучшая модель (Dice: {best_dice:.4f})")

        mlflow.log_metric("best_val_dice", best_dice)

        # Финальная оценка лучшего чекпоинта на отложенной тестовой выборке
        model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE))
        test_loss, test_dice = validate(model, test_loader, criterion, desc="Test")
        print(f"Test Loss: {test_loss:.4f}  Test Dice: {test_dice:.4f}")
        mlflow.log_metrics({"test_loss": test_loss, "test_dice": test_dice})

        mlflow.log_artifact(config.CHECKPOINT_PATH)
        # input_example обязателен в MLflow 3.x; аргумент name заменил artifact_path
        input_example = np.zeros((1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.float32)
        model_dir_arg = "name" if "name" in inspect.signature(mlflow.pytorch.log_model).parameters else "artifact_path"
        mlflow.pytorch.log_model(model.cpu().eval(), input_example=input_example, **{model_dir_arg: "model"})


if __name__ == "__main__":
    train()
