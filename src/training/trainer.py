"""Training logic for Fashion-MNIST models."""

import time

import torch
import torch.nn as nn

from src.config import DEVICE


def train_epoch(model, train_loader, criterion, optimizer):
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        
    Returns:
        float: Total loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss


def evaluate(model, test_loader):
    """
    Evaluate the model on test data.
    
    Args:
        model: The neural network model.
        test_loader: DataLoader for test data.
        
    Returns:
        float: Accuracy on the test set.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = model(x)
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return correct / total


def train(model, train_loader, test_loader, num_epochs, learning_rate):
    """
    Train the model for multiple epochs.
    
    Args:
        model: The neural network model.
        train_loader: DataLoader for training data.
        test_loader: DataLoader for test data.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        
    Returns:
        tuple: (cost_list, accuracy_list, training_time)
    """
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    cost_list = []
    accuracy_list = []

    start_time = time.time()

    for epoch in range(num_epochs):
        # Train
        cost = train_epoch(model, train_loader, criterion, optimizer)
        cost_list.append(cost)

        # Evaluate
        accuracy = evaluate(model, test_loader)
        accuracy_list.append(accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}, Cost: {cost:.4f}, Accuracy: {accuracy:.4f}")

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    return cost_list, accuracy_list, training_time
