import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 16
NUM_WORKERS = 4

LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN = "gen.pth.tar"
CHECKPOINT_DISC = "disc.pth.tar"