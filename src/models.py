import torch
import torch.nn as nn
from typing import Tuple


class FeedForwardNN(nn.Module):
    """
    A flexible feedforward neural network module.
    """
    def __init__(
        self,
        input_features: int,
        hidden_units: Tuple[int, ...],
        output_features: int,
        dropout_rate: float = 0.5
    ):
        """
        Args:
            input_features (int): Number of input features.
            hidden_units (Tuple[int, ...]): A tuple where each element is the number
                                           of neurons in a hidden layer.
            output_features (int): Number of output features.
            dropout_rate (float): The dropout probability.
        """
        super().__init__()
        
        layers = []
        in_features = input_features
        
        # Dynamically create hidden layers
        for h_units in hidden_units:
            layers.append(nn.Linear(in_features, h_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = h_units
            
        # Add the output layer
        layers.append(nn.Linear(in_features, output_features))
        
        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            The output tensor.
        """
        return self.network(x)
