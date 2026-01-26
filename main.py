"""
Fashion-MNIST Classification with CNN

Main entry point for training and evaluating CNN models on Fashion-MNIST dataset.
"""

import torch

from src.config import (
    RANDOM_SEED,
    CONV_CHANNELS,
    HIDDEN_SIZES,
    NUM_CLASSES,
    IN_CHANNELS,
    KERNEL_SIZE,
    USE_BATCH_NORM,
    LEARNING_RATE,
    NUM_EPOCHS,
    DEVICE,
)
from src.data import get_dataloaders
from src.models import CNN, CNNBatchNorm
from src.training import train
from src.visualization import plot_sample, plot_training_results


def main():
    # Set random seed for reproducibility
    torch.manual_seed(RANDOM_SEED)

    print(f"Using device: {DEVICE}")

    # Load data
    print("Loading Fashion-MNIST dataset...")
    train_loader, test_loader, train_dataset, test_dataset = get_dataloaders()

    # Show sample images
    print("\nShowing sample images...")
    for i, sample in enumerate(test_dataset):
        plot_sample(sample)
        if i >= 2:
            break

    # Create model
    if USE_BATCH_NORM:
        print("\nUsing CNN with Batch Normalization")
        model = CNNBatchNorm(
            conv_channels=CONV_CHANNELS,
            hidden_sizes=HIDDEN_SIZES,
            num_classes=NUM_CLASSES,
            in_channels=IN_CHANNELS,
            kernel_size=KERNEL_SIZE,
        )
    else:
        print("\nUsing CNN without Batch Normalization")
        model = CNN(
            conv_channels=CONV_CHANNELS,
            hidden_sizes=HIDDEN_SIZES,
            num_classes=NUM_CLASSES,
            in_channels=IN_CHANNELS,
            kernel_size=KERNEL_SIZE,
        )

    print(f"Model architecture:\n{model}\n")

    # Train model
    print("Starting training...")
    cost_list, accuracy_list, training_time = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE
    )

    # Plot results
    plot_training_results(cost_list, accuracy_list)

    print(f"\nFinal accuracy: {accuracy_list[-1]:.4f}")


if __name__ == "__main__":
    main()
