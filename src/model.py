"""
Physics-Informed Neural Network model for solving the wave equation.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class WavePINN(nn.Module):
    """
    Physics-Informed Neural Network for solving the 1D wave equation.
    
    The network takes (x, t) coordinates as input and outputs the wave
    amplitude u(x, t) at that point.
    
    Args:
        layers: List of layer sizes. Default: [2, 64, 64, 64, 64, 1]
        activation: Activation function. Default: nn.Tanh
    """
    
    def __init__(
        self, 
        layers: Optional[List[int]] = None,
        activation: Optional[nn.Module] = None
    ):
        super(WavePINN, self).__init__()
        
        if layers is None:
            layers = [2, 64, 64, 64, 64, 1]
        
        if activation is None:
            activation = nn.Tanh
        
        self.layers = nn.ModuleList()
        self.activation = activation
        
        # Build network layers
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        
        # Initialize weights using Xavier initialization
        self.init_weights()
    
    def init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # Use smaller weights for the last layer
        nn.init.xavier_normal_(self.layers[-1].weight, gain=0.1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Spatial coordinates, shape (N, 1)
            t: Temporal coordinates, shape (N, 1)
            
        Returns:
            u: Wave amplitude at (x, t), shape (N, 1)
        """
        # Concatenate inputs
        inputs = torch.cat([x, t], dim=1)
        
        # Pass through layers
        out = inputs
        for i, layer in enumerate(self.layers[:-1]):
            out = self.activation()(layer(out))
        
        # No activation on the output layer
        out = self.layers[-1](out)
        
        return out


class ImprovedWavePINN(WavePINN):
    """
    Enhanced PINN with additional features for better training stability.
    
    Includes:
    - Input scaling
    - Skip connections (optional)
    - Batch normalization (optional)
    """
    
    def __init__(
        self,
        layers: Optional[List[int]] = None,
        activation: Optional[nn.Module] = None,
        use_skip: bool = False,
        use_batch_norm: bool = False,
        input_scale: float = 1.0
    ):
        super().__init__(layers, activation)
        
        self.use_skip = use_skip
        self.use_batch_norm = use_batch_norm
        self.input_scale = input_scale
        
        if self.use_batch_norm:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(layers[i + 1]) 
                for i in range(len(layers) - 2)
            ])
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Enhanced forward pass with optional features.
        """
        # Scale inputs for better conditioning
        x = x * self.input_scale
        t = t * self.input_scale
        
        inputs = torch.cat([x, t], dim=1)
        out = inputs
        
        # Store for skip connections
        skip = None
        
        for i, layer in enumerate(self.layers[:-1]):
            out = layer(out)
            
            # Batch normalization
            if self.use_batch_norm and i < len(self.batch_norms):
                out = self.batch_norms[i](out)
            
            out = self.activation()(out)
            
            # Skip connection from input
            if self.use_skip and i == 0:
                skip = out
            elif self.use_skip and i == len(self.layers) - 2:
                out = out + skip
        
        # Output layer
        out = self.layers[-1](out)
        
        return out


def create_model(
    model_type: str = "standard",
    **kwargs
) -> nn.Module:
    """
    Factory function to create PINN models.
    
    Args:
        model_type: Type of model ("standard" or "improved")
        **kwargs: Additional arguments for the model
        
    Returns:
        model: PINN model instance
    """
    if model_type == "standard":
        return WavePINN(**kwargs)
    elif model_type == "improved":
        return ImprovedWavePINN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
