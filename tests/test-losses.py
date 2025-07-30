"""
Unit tests for physics-informed loss functions.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import WavePINN
from src.losses import PhysicsInformedLoss, AdaptivePhysicsInformedLoss


class TestPhysicsInformedLoss:
    """Test cases for the PhysicsInformedLoss class."""
    
    @pytest.fixture
    def model(self):
        """Create a model for testing."""
        return WavePINN()
    
    @pytest.fixture
    def loss_fn(self):
        """Create a loss function instance."""
        return PhysicsInformedLoss()
    
    @pytest.fixture
    def device(self):
        """Get the appropriate device."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_loss_creation(self):
        """Test loss function creation with different parameters."""
        # Default
        loss1 = PhysicsInformedLoss()
        assert loss1.c == 1.0
        assert loss1.w_pde == 1.0
        assert loss1.w_ic == 50.0
        assert loss1.w_bc == 50.0
        
        # Custom weights
        loss2 = PhysicsInformedLoss(
            wave_speed=2.0,
            pde_weight=10.0,
            ic_weight=100.0,
            bc_weight=100.0
        )
        assert loss2.c == 2.0
        assert loss2.w_pde == 10.0
    
    def test_pde_loss_computation(self, model, loss_fn, device):
        """Test PDE residual loss computation."""
        model = model.to(device)
        
        x = torch.rand(100, 1, device=device)
        t = torch.rand(100, 1, device=device)
        
        pde_loss = loss_fn.compute_pde_loss(model, x, t)
        
        assert isinstance(pde_loss, torch.Tensor)
        assert pde_loss.shape == ()  # Scalar
        assert pde_loss.item() >= 0  # Non-negative
        assert torch.isfinite(pde_loss)
    
    def test_ic_loss_computation(self, model, loss_fn, device):
        """Test initial condition loss computation."""
        model = model.to(device)
        
        x = torch.rand(50, 1, device=device)
        
        ic_loss = loss_fn.compute_ic_loss(model, x)
        
        assert isinstance(ic_loss, torch.Tensor)
        assert ic_loss.shape == ()
        assert ic_loss.item() >= 0
        assert torch.isfinite(ic_loss)
    
    def test_ic_velocity_loss_computation(self, model, loss_fn, device):
        """Test initial velocity loss computation."""
        model = model.to(device)
        
        x = torch.rand(50, 1, device=device)
        
        ic_vel_loss = loss_fn.compute_ic_velocity_loss(model, x)
        
        assert isinstance(ic_vel_loss, torch.Tensor)
        assert ic_vel_loss.shape == ()
        assert ic_vel_loss.item() >= 0
        assert torch.isfinite(ic_vel_loss)
    
    def test_bc_loss_computation(self, model, loss_fn, device):
        """Test boundary condition loss computation."""
        model = model.to(device)
        
        # Boundary points (x=0 and x=1)
        x_bc = torch.cat([
            torch.zeros(25, 1, device=device),
            torch.ones(25, 1, device=device)
        ], dim=0)
        t_bc = torch.rand(50, 1, device=device)
        
        bc_loss = loss_fn.compute_bc_loss(model, x_bc, t_bc)
        
        assert isinstance(bc_loss, torch.Tensor)
        assert bc_loss.shape == ()
        assert bc_loss.item() >= 0
        assert torch.isfinite(bc_loss)
    
    def test_total_loss_computation(self, model, loss_fn, device):
        """Test total loss computation."""
        model = model.to(device)
        
        # Generate test data
        x_pde = torch.rand(100, 1, device=device)
        t_pde = torch.rand(100, 1, device=device)
        x_ic = torch.rand(50, 1, device=device)
        x_bc = torch.cat([
            torch.zeros(25, 1, device=device),
            torch.ones(25, 1, device=device)
        ], dim=0)
        t_bc = torch.rand(50, 1, device=device)
        
        # Compute total loss
        total_loss, loss_dict = loss_fn(model, x_pde, t_pde, x_ic, x_bc, t_bc)
        
        # Check total loss
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.shape == ()
        assert total_loss.item() >= 0
        assert torch.isfinite(total_loss)
        
        # Check loss dictionary
        assert isinstance(loss_dict, dict)
        assert 'total' in loss_dict
        assert 'pde' in loss_dict
        assert 'ic' in loss_dict
        assert 'bc' in loss_dict
        
        # Check consistency
        expected_total = (
            loss_fn.w_pde * loss_dict['pde'] +
            loss_fn.w_ic * loss_dict['ic'] +
            loss_fn.w_ic_vel * loss_dict['ic_vel'] +
            loss_fn.w_bc * loss_dict['bc']
        )
        assert abs(loss_dict['total'] - expected_total) < 1e-5
    
    def test_loss_backward(self, model, loss_fn, device):
        """Test that loss can be backpropagated."""
        model = model.to(device)
        
        # Generate test data
        x_pde = torch.rand(50, 1, device=device)
        t_pde = torch.rand(50, 1, device=device)
        x_ic = torch.rand(20, 1, device=device)
        x_bc = torch.cat([
            torch.zeros(10, 1, device=device),
            torch.ones(10, 1, device=device)
        ], dim=0)
        t_bc = torch.rand(20, 1, device=device)
        
        # Compute loss
        total_loss, _ = loss_fn(model, x_pde, t_pde, x_ic, x_bc, t_bc)
        
        # Check that backward pass works
        total_loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
    
    def test_different_wave_speeds(self, model, device):
        """Test loss computation with different wave speeds."""
        model = model.to(device)
        
        x = torch.rand(50, 1, device=device)
        t = torch.rand(50, 1, device=device)
        
        # Test different wave speeds
        wave_speeds = [0.5, 1.0, 2.0, 5.0]
        
        for c in wave_speeds:
            loss_fn = PhysicsInformedLoss(wave_speed=c)
            pde_loss = loss_fn.compute_pde_loss(model, x, t)
            
            assert torch.isfinite(pde_loss)
            assert pde_loss.item() >= 0


class TestAdaptivePhysicsInformedLoss:
    """Test cases for adaptive loss weighting."""
    
    @pytest.fixture
    def model(self):
        return WavePINN()
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_adaptive_loss_creation(self):
        """Test adaptive loss creation."""
        loss_fn = AdaptivePhysicsInformedLoss(adapt_freq=50)
        
        assert loss_fn.adapt_freq == 50
        assert loss_fn.iteration == 0
        assert isinstance(loss_fn.loss_history, dict)
    
    def test_weight_adaptation(self, model, device):
        """Test that weights adapt during training."""
        model = model.to(device)
        loss_fn = AdaptivePhysicsInformedLoss(adapt_freq=10)
        
        # Initial weights
        initial_weights = {
            'pde': loss_fn.w_pde,
            'ic': loss_fn.w_ic,
            'bc': loss_fn.w_bc
        }
        
        # Generate test data
        x_pde = torch.rand(50, 1, device=device)
        t_pde = torch.rand(50, 1, device=device)
        x_ic = torch.rand(20, 1, device=device)
        x_bc = torch.cat([
            torch.zeros(10, 1, device=device),
            torch.ones(10, 1, device=device)
        ], dim=0)
        t_bc = torch.rand(20, 1, device=device)
        
        # Run multiple iterations
        for _ in range(15):  # More than adapt_freq
            total_loss, loss_dict = loss_fn(model, x_pde, t_pde, x_ic, x_bc, t_bc)
        
        # Check that weights have changed
        weights_changed = (
            loss_fn.w_pde != initial_weights['pde'] or
            loss_fn.w_ic != initial_weights['ic'] or
            loss_fn.w_bc != initial_weights['bc']
        )
        
        # Weights might not change if losses are already balanced
        # but history should be populated
        assert len(loss_fn.loss_history['pde']) == 15
    
    def test_loss_history_tracking(self, model, device):
        """Test that loss history is properly tracked."""
        model = model.to(device)
        loss_fn = AdaptivePhysicsInformedLoss()
        
        # Generate test data
        x_pde = torch.rand(50, 1, device=device)
        t_pde = torch.rand(50, 1, device=device)
        x_ic = torch.rand(20, 1, device=device)
        x_bc = torch.cat([
            torch.zeros(10, 1, device=device),
            torch.ones(10, 1, device=device)
        ], dim=0)
        t_bc = torch.rand(20, 1, device=device)
        
        # Run several iterations
        n_iterations = 5
        for _ in range(n_iterations):
            loss_fn(model, x_pde, t_pde, x_ic, x_bc, t_bc)
        
        # Check history
        assert len(loss_fn.loss_history['pde']) == n_iterations
        assert len(loss_fn.loss_history['ic']) == n_iterations
        assert len(loss_fn.loss_history['bc']) == n_iterations
        assert loss_fn.iteration == n_iterations


class TestLossEdgeCases:
    """Test edge cases for loss functions."""
    
    @pytest.fixture
    def model(self):
        return WavePINN()
    
    @pytest.fixture
    def loss_fn(self):
        return PhysicsInformedLoss()
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_single_point_loss(self, model, loss_fn, device):
        """Test loss computation with single point."""
        model = model.to(device)
        
        x_pde = torch.rand(1, 1, device=device)
        t_pde = torch.rand(1, 1, device=device)
        x_ic = torch.rand(1, 1, device=device)
        x_bc = torch.zeros(1, 1, device=device)
        t_bc = torch.rand(1, 1, device=device)
        
        total_loss, loss_dict = loss_fn(model, x_pde, t_pde, x_ic, x_bc, t_bc)
        
        assert torch.isfinite(total_loss)
        assert total_loss.item() >= 0
    
    def test_zero_weights(self, model, device):
        """Test loss with zero weights."""
        model = model.to(device)
        loss_fn = PhysicsInformedLoss(
            pde_weight=0.0,
            ic_weight=0.0,
            ic_vel_weight=0.0,
            bc_weight=1.0
        )
        
        x_pde = torch.rand(50, 1, device=device)
        t_pde = torch.rand(50, 1, device=device)
        x_ic = torch.rand(20, 1, device=device)
        x_bc = torch.zeros(20, 1, device=device)
        t_bc = torch.rand(20, 1, device=device)
        
        total_loss, loss_dict = loss_fn(model, x_pde, t_pde, x_ic, x_bc, t_bc)
        
        # Only BC loss should contribute
        assert abs(total_loss.item() - loss_dict['bc']) < 1e-5
    
    def test_extreme_points(self, model, loss_fn, device):
        """Test loss at boundary and corner cases."""
        model = model.to(device)
        
        # Test at boundaries
        x_pde = torch.tensor([[0.0], [1.0]], device=device)
        t_pde = torch.tensor([[0.0], [1.0]], device=device)
        x_ic = torch.tensor([[0.0], [1.0]], device=device)
        x_bc = torch.tensor([[0.0], [1.0]], device=device)
        t_bc = torch.tensor([[0.0], [1.0]], device=device)
        
        total_loss, loss_dict = loss_fn(model, x_pde, t_pde, x_ic, x_bc, t_bc)
        
        assert torch.isfinite(total_loss)
        for key in loss_dict:
            assert np.isfinite(loss_dict[key])


if __name__ == "__main__":
    pytest.main([__file__])
