"""Data loading and transformation utilities for Fashion-MNIST."""

import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.config import IMAGE_SIZE, DATA_ROOT, BATCH_SIZE


def get_transforms():
    """Get the data transformation pipeline."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])


def get_dataloaders(batch_size=BATCH_SIZE):
    """
    Load Fashion-MNIST dataset and create data loaders.
    
    Args:
        batch_size: Batch size for the data loaders.
        
    Returns:
        tuple: (train_loader, test_loader, train_dataset, test_dataset)
    """
    transform = get_transforms()

    train_dataset = dsets.FashionMNIST(
        root=DATA_ROOT,
        train=True,
        transform=transform,
        download=True
    )
    
    test_dataset = dsets.FashionMNIST(
        root=DATA_ROOT,
        train=False,
        transform=transform,
        download=True
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset
