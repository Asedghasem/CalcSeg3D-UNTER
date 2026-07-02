import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_ROOT = os.path.join(BASE_DIR, "Dataset", "Data96")

MODEL_WEIGHTS = os.path.join(BASE_DIR, "Codes", "UNETR++", "pretrainUnetRFineTune.pth")

TRAIN_DIR = os.path.join(DATA_ROOT, "Train")
VAL_DIR = os.path.join(DATA_ROOT, "Validation")
TEST_DIR = os.path.join(DATA_ROOT, "Test")
IMG_SIZE = (96,96,96)

train_batch_size = 4
val_batch_size = 1
test_batch_size = 1

LR = 1e-4

NUM_CLASSES = 2

DEVICE = "cuda"