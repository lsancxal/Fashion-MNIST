"""Utility functions for Fashion-MNIST visualization."""

import matplotlib.pyplot as plt
import torch

from src.config import DEVICE, IMAGE_SIZE

# Fashion-MNIST class labels
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


def get_predictions(model, data_loader):
    """
    Get all predictions and true labels from the model.
    
    Args:
        model: The trained neural network model.
        data_loader: DataLoader for the dataset.
        
    Returns:
        tuple: (predictions, labels) as numpy arrays.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(DEVICE)
            output = model(x)
            _, predicted = torch.max(output, 1)
            all_predictions.append(predicted.cpu())
            all_labels.append(y)
    
    return (
        torch.cat(all_predictions).numpy(),
        torch.cat(all_labels).numpy()
    )


def plot_sample(data_sample, show=True):
    """
    Display a single data sample.
    
    Args:
        data_sample: Tuple of (image_tensor, label).
        show: Whether to call plt.show().
    """
    image, label = data_sample
    plt.imshow(image.numpy().reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
    plt.title(f'Label: {label}')
    if show:
        plt.show()
