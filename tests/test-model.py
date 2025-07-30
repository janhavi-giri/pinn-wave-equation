"""
Unit tests for PINN model architectures.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import WavePINN, ImprovedWavePINN, create_model


class TestWavePINN:
    """Test cases for the WavePINN model."""
    
    @pytest.fixture
    def model(self):
        """Create a model instance for testing."""
        return WavePINN()
    
    @pytest.fixture
    def device(self):
        """Get the appropriate device."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_model_creation(self, model):
        """Test that model is created correctly."""
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'layers')
        assert hasattr(model, 'forward')
    
    def test_model_architecture(self):
        """Test model architecture with custom layers."""
        layers = [2, 32, 32, 1]
        model = WavePINN(layers=layers)
        
        # Check number of layers
        assert len(model.layers) == len(layers) - 1
        
        # Check layer sizes
        for i, layer in enumerate(model.layers):
            assert isinstance(layer, nn.Linear)
            assert layer.in_features == layers[i]
            assert layer.out_features == layers[i + 1]
    
    def test_forward_pass(self, model, device):
        """Test forward pass with various input sizes."""
        model = model.to(device)
        batch_sizes = [1, 10, 100, 1000]
        
        for batch_size in batch_sizes:
            x = torch.rand(batch_size, 1, device=device)
            t = torch.rand(batch_size, 1, device=device)
            
            output = model(x, t)
            
            # Check output shape
            assert output.shape == (batch_size, 1)
            
            # Check output is finite
            assert torch.isfinite(output).all()
    
    def test_gradient_computation(self, model, device):
        """Test that gradients can be computed."""
        model = model.to(device)
        
        x = torch.rand(10, 1, device=device, requires_grad=True)
        t = torch.rand(10, 1, device=device, requires_grad=True)
        
        u = model(x, t)
        
        # Compute gradients
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        
        assert u_x.shape == x.shape
        assert u_t.shape == t.shape
        assert torch.isfinite(u_x).all()
        assert torch.isfinite(u_t).all()
    
    def test_second_derivatives(self, model, device):
        """Test computation of second derivatives."""
        model = model.to(device)
        
        x = torch.rand(10, 1, device=device, requires_grad=True)
        t = torch.rand(10, 1, device=device, requires_grad=True)
        
        u = model(x, t)
        
        # First derivatives
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        
        # Second derivatives
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t.sum(), t, create_graph=True)[0]
        
        assert u_xx.shape == x.shape
        assert u_tt.shape == t.shape
        assert torch.isfinite(u_xx).all()
        assert torch.isfinite(u_tt).all()
    
    def test_parameter_count(self):
        """Test parameter counting."""
        model = WavePINN(layers=[2, 64, 64, 64, 64, 1])
        
        total_params = sum(p.numel() for p in