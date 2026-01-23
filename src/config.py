"""Configuration and hyperparameters for Fashion-MNIST training."""

import torch


# Data settings
IMAGE_SIZE = 16
DATA_ROOT = ".fashion/data"
BATCH_SIZE = 100

# Model settings
OUT_CHANNELS_1 = 16
OUT_CHANNELS_2 = 32
NUM_CLASSES = 10
USE_BATCH_NORM = True

# Training settings
LEARNING_RATE = 0.1
NUM_EPOCHS = 5
RANDOM_SEED = 0

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
