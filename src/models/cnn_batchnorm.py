"""CNN model with batch normalization for Fashion-MNIST classification."""

import torch.nn as nn
import torch.nn.functional as F


class CNNBatchNorm(nn.Module):
    """
    Dynamic CNN model with batch normalization.
    
    Args:
        conv_channels: Tuple of output channels for each conv layer, e.g., (16, 32) or (16, 32, 64).
        hidden_sizes: Tuple of hidden layer sizes before final output, e.g., (128,) or (256, 128).
                      Use empty tuple () for no hidden layers.
        num_classes: Number of output classes.
        in_channels: Number of input channels (1 for grayscale, 3 for RGB).
        kernel_size: Convolution kernel size.
    """

    def __init__(
        self,
        conv_channels=(16, 32),
        hidden_sizes=(),
        num_classes=10,
        in_channels=1,
        kernel_size=5,
    ):
        super().__init__()
        
        # Build convolutional layers with batch norm
        self.conv_layers = nn.ModuleList()
        self.bn_conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        padding = kernel_size // 2  # Same padding to preserve spatial size
        channels = [in_channels] + list(conv_channels)
        
        for i in range(len(conv_channels)):
            self.conv_layers.append(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernel_size, padding=padding)
            )
            self.bn_conv_layers.append(nn.BatchNorm2d(channels[i + 1]))
            self.pool_layers.append(nn.MaxPool2d(kernel_size=2))
        
        # Build fully connected layers with batch norm
        self.flatten = nn.Flatten()
        self.fc_layers = nn.ModuleList()
        self.bn_fc_layers = nn.ModuleList()
        
        # Hidden layers
        if hidden_sizes:
            # First hidden layer uses LazyLinear
            self.fc_layers.append(nn.LazyLinear(hidden_sizes[0]))
            self.bn_fc_layers.append(nn.BatchNorm1d(hidden_sizes[0]))
            
            # Remaining hidden layers
            for i in range(1, len(hidden_sizes)):
                self.fc_layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
                self.bn_fc_layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            
            # Output layer with batch norm
            self.output_layer = nn.Linear(hidden_sizes[-1], num_classes)
        else:
            # No hidden layers - direct connection to output
            self.output_layer = nn.LazyLinear(num_classes)
        
        self.bn_output = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        # Convolutional layers
        for conv, bn, pool in zip(self.conv_layers, self.bn_conv_layers, self.pool_layers):
            x = conv(x)
            x = bn(x)
            x = F.leaky_relu(x)
            x = pool(x)
        
        # Flatten
        x = self.flatten(x)
        
        # Hidden layers
        for fc, bn in zip(self.fc_layers, self.bn_fc_layers):
            x = fc(x)
            x = bn(x)
            x = F.leaky_relu(x)
        
        # Output layer
        x = self.output_layer(x)
        x = self.bn_output(x)
        return x
