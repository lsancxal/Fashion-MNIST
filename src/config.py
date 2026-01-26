"""Configuration and hyperparameters for Fashion-MNIST training."""

import torch


# Data settings
IMAGE_SIZE = 28
DATA_ROOT = ".fashion/data"
BATCH_SIZE = 100

# Model architecture settings
CONV_CHANNELS = (16, 32)       # Output channels for each conv layer, e.g., (16, 32) or (16, 32, 64)
HIDDEN_SIZES = (16,)              # Hidden FC layer sizes, e.g., (128,) or (256, 128). Empty for none.
NUM_CLASSES = 10
IN_CHANNELS = 1                # 1 for grayscale, 3 for RGB
KERNEL_SIZE = 5
USE_BATCH_NORM = True

# Training settings
LEARNING_RATE = 0.1
NUM_EPOCHS = 10
RANDOM_SEED = 0

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
