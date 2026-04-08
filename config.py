
import os
import torch

DATA_DIR = os.environ.get(
    "DATA_DIR",
    os.path.join(os.path.expanduser("~"), "Downloads", "archive", "kvasir-seg", "Kvasir-SEG"),
)
IMAGE_DIR = os.path.join(DATA_DIR, "images")
MASK_DIR  = os.path.join(DATA_DIR, "masks")

IMAGE_SIZE    = 256
BATCH_SIZE    = 8
NUM_EPOCHS    = 30
LEARNING_RATE = 1e-4
VAL_SPLIT     = 0.2

DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "best_model.pth"

# MLflow
MLFLOW_EXPERIMENT = "polyp-segmentation"
MLFLOW_RUN_NAME   = f"unet-resnet34-bs{BATCH_SIZE}-lr{LEARNING_RATE}"
