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
from src.visualization import (
    plot_sample,
    plot_training_results,
    get_predictions,
    plot_confusion_matrix_with_stats,
    plot_precision,
    plot_recall,
    print_classification_report,
)


def create_model():
    """Create and return the CNN model based on config settings."""
    model_class = CNNBatchNorm if USE_BATCH_NORM else CNN
    model_name = "CNN with Batch Normalization" if USE_BATCH_NORM else "CNN"
    
    print(f"\nUsing {model_name}")
    
    return model_class(
        conv_channels=CONV_CHANNELS,
        hidden_sizes=HIDDEN_SIZES,
        num_classes=NUM_CLASSES,
        in_channels=IN_CHANNELS,
        kernel_size=KERNEL_SIZE,
    )


def show_samples(dataset, num_samples=3):
    """Display sample images from the dataset."""
    print("\nShowing sample images...")
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        plot_sample(sample)


def evaluate_model(model, test_loader):
    """Generate all evaluation visualizations and metrics."""
    print("\nEvaluating model...")
    
    # Get predictions once - reuse for all visualizations
    predictions, labels = get_predictions(model, test_loader)
    
    # Confusion matrix
    print("Generating confusion matrix...")
    plot_confusion_matrix_with_stats(predictions, labels)
    
    # Precision and recall plots
    print("Generating precision plot...")
    plot_precision(predictions, labels)
    
    print("Generating recall plot...")
    plot_recall(predictions, labels)
    
    # Classification report
    print_classification_report(predictions, labels)


def main():
    # Set random seed for reproducibility
    torch.manual_seed(RANDOM_SEED)
    print(f"Using device: {DEVICE}")

    # Load data
    print("Loading Fashion-MNIST dataset...")
    train_loader, test_loader, train_dataset, test_dataset = get_dataloaders()

    # Show sample images
    show_samples(test_dataset)

    # Create and display model
    model = create_model()
    print(f"Model architecture:\n{model}\n")

    # Train model
    print("Starting training...")
    cost_list, accuracy_list, _ = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE
    )

    # Plot training progress
    plot_training_results(cost_list, accuracy_list)

    # Evaluate and visualize results
    evaluate_model(model, test_loader)
    
    print(f"\nFinal accuracy: {accuracy_list[-1]:.4f}")


if __name__ == "__main__":
    main()
