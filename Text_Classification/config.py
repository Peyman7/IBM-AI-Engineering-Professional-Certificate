# config.py
import torch

BATCH_SIZE = 64
EMBED_DIM = 64
EPOCHS = 10
LR = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4  # AG_NEWS has 4 classes
