"""Confusion matrix visualization for Fashion-MNIST classification."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from src.config import DEVICE

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
        tuple: (all_predictions, all_labels) as numpy arrays.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(DEVICE)
            output = model(x)
            _, predicted = torch.max(output.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(y.numpy())
    
    return np.array(all_predictions), np.array(all_labels)


def plot_confusion_matrix(model, data_loader, normalize=True, figsize=(12, 10), show=True):
    """
    Plot confusion matrix for the model predictions.
    
    Args:
        model: The trained neural network model.
        data_loader: DataLoader for the dataset (typically test set).
        normalize: If True, normalize the confusion matrix by row (true labels).
        figsize: Figure size as (width, height).
        show: Whether to call plt.show().
        
    Returns:
        tuple: (fig, confusion_matrix_array)
    """
    # Get predictions
    predictions, labels = get_predictions(model, data_loader)
    
    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Normalize if requested
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        square=True,
        cbar_kws={'shrink': 0.8}
    )
    
    # Labels and title
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig, cm


def plot_confusion_matrix_with_stats(model, data_loader, figsize=(14, 10), show=True):
    """
    Plot confusion matrix with per-class accuracy statistics.
    
    Args:
        model: The trained neural network model.
        data_loader: DataLoader for the dataset (typically test set).
        figsize: Figure size as (width, height).
        show: Whether to call plt.show().
        
    Returns:
        tuple: (fig, confusion_matrix_array, per_class_accuracy)
    """
    # Get predictions
    predictions, labels = get_predictions(model, data_loader)
    
    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Calculate per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    # Normalize for display
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, 
                                    gridspec_kw={'width_ratios': [3, 1]})
    
    # Plot confusion matrix heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax1,
        square=True,
        cbar_kws={'shrink': 0.8}
    )
    
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Plot per-class accuracy bar chart
    colors = plt.cm.Blues(per_class_accuracy)
    bars = ax2.barh(range(len(CLASS_NAMES)), per_class_accuracy, color=colors)
    ax2.set_yticks(range(len(CLASS_NAMES)))
    ax2.set_yticklabels(CLASS_NAMES)
    ax2.set_xlabel('Accuracy', fontsize=12)
    ax2.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.invert_yaxis()  # Match confusion matrix order
    
    # Add accuracy values on bars
    for i, (bar, acc) in enumerate(zip(bars, per_class_accuracy)):
        ax2.text(acc + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{acc:.2%}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    # Print summary statistics
    overall_accuracy = (predictions == labels).sum() / len(labels)
    print(f"\nOverall Accuracy: {overall_accuracy:.2%}")
    print(f"\nPer-Class Accuracy:")
    for name, acc in zip(CLASS_NAMES, per_class_accuracy):
        print(f"  {name:12s}: {acc:.2%}")
    
    return fig, cm, per_class_accuracy
