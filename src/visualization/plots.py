"""Visualization utilities for Fashion-MNIST."""

import matplotlib.pyplot as plt

from src.config import IMAGE_SIZE


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


def plot_training_results(cost_list, accuracy_list, show=True):
    """
    Plot training cost and accuracy over epochs.
    
    Args:
        cost_list: List of cost values per epoch.
        accuracy_list: List of accuracy values per epoch.
        show: Whether to call plt.show().
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot cost
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cost', color=color)
    ax1.plot(cost_list, color=color, label='Cost')
    ax1.tick_params(axis='y', labelcolor=color)

    # Plot accuracy on secondary axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(accuracy_list, color=color, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Training Progress')
    
    if show:
        plt.show()
    
    return fig
