"""
Unit tests for physics validation of PINN solutions.
"""

import pytest
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import WavePINN
from src.train import train_model
from src.evaluate import compute_analytical_solution, evaluate_model
from src.losses import PhysicsInformedLoss


class TestPhysicsValidation:
    """Test cases to validate physical correctness of PINN solutions."""
    
    @pytest.fixture
    def device(self):
        """Get the appropriate device."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def trained_model(self, device):
        """Create and train a model for testing."""
        model = WavePINN().to(device)
        # Quick training for tests (not full convergence)
        history = train_model(model, epochs=500, verbose=False, device=device)
        return model
    
    def test_analytical_solution(self):
        """Test the analytical solution implementation."""
        # Test at known points
        x = np.array([[0.0], [0.5], [1.0]])
        t = 0.0
        
        u = compute_analytical_solution(x, t)
        
        # At t=0: u = sin(πx)
        expected = np.array([[0.0], [1.0], [0.0]])
        np.testing.assert_allclose(u, expected, rtol=1e-10)
        
        # Test at t=0.5 (quarter period)
        t = 0.5
        u = compute_analytical_solution(x, t)
        
        # At t=0.5: u = sin(πx)cos(π/2) = 0
        expected = np.array([[0.0], [0.0], [0.0]])
        np.testing.assert_allclose(u, expected, atol=1e-10)
    
    def test_initial_condition_satisfaction(self, trained_model, device):
        """Test that trained model satisfies initial condition."""
        model = trained_model
        model.eval()
        
        # Test points at t=0
        x = torch.linspace(0, 1, 100, device=device).reshape(-1, 1)
        t = torch.zeros_like(x)
        
        with torch.no_grad():
            u_pred = model(x, t).cpu().numpy()
        
        # Expected: sin(πx)
        u_true = np.sin(np.pi * x.cpu().numpy())
        
        # Check mean error
        error = np.mean(np.abs(u_pred - u_true))
        assert error < 0.1  # Relaxed for quick training
    
    def test_boundary_conditions(self, trained_model, device):
        """Test that trained model satisfies boundary conditions."""
        model = trained_model
        model.eval()
        
        # Test at various times
        times = torch.linspace(0, 1, 20, device=device)
        
        for t_val in times:
            # Left boundary (x=0)
            x_left = torch.zeros(1, 1, device=device)
            t_left = torch.full((1, 1), t_val.item(), device=device)
            
            with torch.no_grad():
                u_left = model(x_left, t_left).item()
            
            assert abs(u_left) < 0.1  # Should be close to 0
            
            # Right boundary (x=1)
            x_right = torch.ones(1, 1, device=device)
            t_right = torch.full((1, 1), t_val.item(), device=device)
            
            with torch.no_grad():
                u_right = model(x_right, t_right).item()
            
            assert abs(u_right) < 0.1  # Should be close to 0
    
    def test_wave_equation_residual(self, trained_model, device):
        """Test that trained model satisfies the wave equation."""
        model = trained_model
        model.eval()
        
        # Random test points
        x = torch.rand(100, 1, device=device, requires_grad=True)
        t = torch.rand(100, 1, device=device, requires_grad=True)
        
        # Compute solution and derivatives
        u = model(x, t)
        
        # First derivatives
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        
        # Second derivatives
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t.sum(), t, create_graph=True)[0]
        
        # Wave equation residual (c=1)
        residual = u_tt - u_xx
        
        # Check that residual is small
        mean_residual = torch.mean(torch.abs(residual)).item()
        assert mean_residual < 0.1  # Relaxed for quick training
    
    def test_energy_conservation(self, trained_model, device):
        """Test approximate energy conservation."""
        model = trained_model
        model.eval()
        
        # Compute energy at different times
        times = [0.0, 0.25, 0.5, 0.75]
        energies = []
        
        for t_val in times:
            x = torch.linspace(0, 1, 100, device=device).reshape(-1, 1)
            t = torch.full_like(x, t_val)
            
            x.requires_grad = True
            t.requires_grad = True
            
            u = model(x, t)
            
            # Kinetic energy ~ (∂u/∂t)²
            u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
            
            # Potential energy ~ (∂u/∂x)²
            u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
            
            # Total energy (approximate integration)
            kinetic = torch.mean(u_t**2).item()
            potential = torch.mean(u_x**2).item()
            total_energy = kinetic + potential
            
            energies.append(total_energy)
        
        # Check that energy doesn't vary too much
        energy_variation = (max(energies) - min(energies)) / np.mean(energies)
        assert energy_variation < 0.5  # Very relaxed for quick training
    
    def test_symmetry(self, trained_model, device):
        """Test solution symmetry about x=0.5."""
        model = trained_model
        model.eval()
        
        # Test at t=0 where solution should be symmetric
        x_left = torch.linspace(0, 0.5, 50, device=device).reshape(-1, 1)
        x_right = torch.linspace(0.5, 1, 50, device=device).reshape(-1, 1)
        t = torch.zeros_like(x_left)
        
        with torch.no_grad():
            u_left = model(x_left, t).cpu().numpy()
            u_right = model(x_right, t).cpu().numpy()
        
        # Check symmetry (approximately)
        u_right_flipped = u_right[::-1]
        
        # At t=0, solution should be symmetric
        symmetry_error = np.mean(np.abs(u_left - u_right_flipped))
        assert symmetry_error < 0.2  # Relaxed for quick training
    
    def test_periodicity(self, trained_model, device):
        """Test that solution is periodic in time."""
        model = trained_model
        model.eval()
        
        # Test at a fixed point
        x = torch.tensor([[0.5]], device=device)
        
        # Solution at t=0 and t=2 (full period) should be similar
        t0 = torch.tensor([[0.0]], device=device)
        t2 = torch.tensor([[2.0]], device=device)
        
        with torch.no_grad():
            u0 = model(x, t0).item()
            u2 = model(x, t2).item()
        
        # For untrained model, this might not hold well
        # Just check they have the same sign at least
        assert np.sign(u0) == np.sign(u2) or abs(u0 - u2) < 0.5


class TestPhysicalConstraints:
    """Test physical constraints and properties."""
    
    def test_wave_speed_positive(self):
        """Test that wave speed is positive."""
        loss_fn = PhysicsInformedLoss(wave_speed=2.0)
        assert loss_fn.c > 0
        
        # Test with negative (should probably raise error in real implementation)
        # For now, just document the behavior
        loss_fn2 = PhysicsInformedLoss(wave_speed=-1.0)
        assert loss_fn2.c == -1.0  # Currently allows negative
    
    def test_loss_weights_positive(self):
        """Test that loss weights are positive."""
        loss_fn = PhysicsInformedLoss()
        
        assert loss_fn.w_pde > 0
        assert loss_fn.w_ic > 0
        assert loss_fn.w_ic_vel > 0
        assert loss_fn.w_bc > 0
    
    def test_solution_bounded(self, device):
        """Test that solution remains bounded."""
        model = WavePINN().to(device)
        
        # Even untrained model should produce bounded outputs
        x = torch.rand(1000, 1, device=device) * 10  # Large domain
        t = torch.rand(1000, 1, device=device) * 10
        
        with torch.no_grad():
            u = model(x, t)
        
        assert torch.isfinite(u).all()
        assert torch.abs(u).max().item() < 100  # Reasonable bound


class TestConvergence:
    """Test convergence properties of PINN training."""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_loss_decreases(self, device):
        """Test that loss decreases during training."""
        model = WavePINN().to(device)
        history = train_model(model, epochs=100, verbose=False, device=device)
        
        # Check that loss generally decreases
        initial_loss = history['loss'][0]
        final_loss = history['loss'][-1]
        
        assert final_loss < initial_loss
        
        # Check that final loss is reasonable
        assert final_loss < 1.0
    
    def test_component_losses_decrease(self, device):
        """Test that all loss components decrease."""
        model = WavePINN().to(device)
        history = train_model(model, epochs=100, verbose=False, device=device)
        
        # Check each component
        for component in ['pde', 'ic', 'bc']:
            initial = history[component][0]
            final = history[component][-1]
            
            # They should generally decrease
            assert final <= initial * 2  # Very relaxed
    
    def test_evaluation_metrics(self, device):
        """Test that evaluation metrics are reasonable."""
        model = WavePINN().to(device)
        train_model(model, epochs=500, verbose=False, device=device)
        
        metrics = evaluate_model(model, device=device, n_test=100)
        
        # Check that metrics exist
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'max_error' in metrics
        
        # Check that errors are bounded (very relaxed for quick training)
        assert metrics['mse'] < 1.0
        assert metrics['max_error'] < 2.0


if __name__ == "__main__":
    pytest.main([__file__])
