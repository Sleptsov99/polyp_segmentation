
import torch

DATA_DIR="C:\\Users\\CosmoPC\\Downloads\\archive\\kvasir-seg\\Kvasir-SEG"
IMAGE_DIR=DATA_DIR + "\\images"
MASK_DIR=DATA_DIR + "\\masks"

IMAGE_SIZE=256
BATCH_SIZE=8
NUM_EPOCHS=30
LEARNING_RATE=1e-4

VAL_SPLIT=0.2

DEVICE= "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_PATH="best_model.pth"
