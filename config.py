import os
from pathlib import Path

import torch

# Пути считаются от папки репозитория, поэтому скрипты работают из любой директории
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# По умолчанию датасет лежит в ./data/Kvasir-SEG (его создаёт download_data.py).
# Можно указать другой путь через переменную окружения DATA_DIR.
DATA_DIR  = os.environ.get("DATA_DIR", os.path.join(PROJECT_DIR, "data", "Kvasir-SEG"))
IMAGE_DIR = os.path.join(DATA_DIR, "images")
MASK_DIR  = os.path.join(DATA_DIR, "masks")

# Официальный архив Kvasir-SEG (Simula Research Laboratory)
DATASET_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"

# Веса энкодера: "imagenet" (по умолчанию, скачиваются при первом запуске) или "none" — без предобучения
ENCODER_WEIGHTS = os.environ.get("ENCODER_WEIGHTS", "imagenet")
if ENCODER_WEIGHTS.lower() == "none":
    ENCODER_WEIGHTS = None

IMAGE_SIZE    = 256
BATCH_SIZE    = 8
NUM_EPOCHS    = int(os.environ.get("NUM_EPOCHS", 30))
LEARNING_RATE = 1e-4

# Разбиение 80 / 10 / 10: val — для выбора лучшей эпохи, test — только для финальной оценки
VAL_SPLIT  = 0.1
TEST_SPLIT = 0.1
SEED       = 42

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 2))

if torch.cuda.is_available():
    DEVICE = "cuda"
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", os.path.join(PROJECT_DIR, "best_model.pth"))
RESULTS_PATH    = os.path.join(PROJECT_DIR, "results.png")

# MLflow: метрики в ./mlflow.db, артефакты в ./mlartifacts внутри репозитория.
# Файловое хранилище ./mlruns в MLflow 3.x отключено, поэтому используется SQLite.
MLFLOW_TRACKING_URI  = os.environ.get(
    "MLFLOW_TRACKING_URI", "sqlite:///" + Path(PROJECT_DIR, "mlflow.db").as_posix()
)
MLFLOW_ARTIFACT_URI  = Path(PROJECT_DIR, "mlartifacts").as_uri()
MLFLOW_EXPERIMENT   = "polyp-segmentation"
MLFLOW_RUN_NAME     = f"unet-resnet34-bs{BATCH_SIZE}-lr{LEARNING_RATE}"
