"""
Evaluation utilities for Physics-Informed Neural Networks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_analytical_solution(
    x: np.ndarray,
    t: float,
    c: float = 1.0
) -> np.ndarray:
    """
    Compute the analytical solution of the wave equation.
    
    For initial condition u(x,0) = sin(πx) and zero initial velocity,
    the solution is: u(x,t) = sin(πx)cos(πct)
    
    Args:
        x: Spatial coordinates
        t: Time value
        c: Wave speed
        
    Returns:
        u: Wave amplitude at (x, t)
    """
    return np.sin(np.pi * x) * np.cos(np.pi * c * t)


def evaluate_model(
    model: nn.Module,
    device: Optional[torch.device] = None,
    n_test: int = 1000
) -> Dict[str, float]:
    """
    Evaluate a trained PINN model.
    
    Args:
        model: Trained PINN model
        device: Device to evaluate on
        n_test: Number of test points
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    metrics = {}
    
    # Test on random points
    x_test = torch.rand(n_test, 1, device=device)
    t_test = torch.rand(n_test, 1, device=device)
    
    with torch.no_grad():
        u_pred = model(x_test, t_test).cpu().numpy()
    
    x_np = x_test.cpu().numpy()
    t_np = t_test.cpu().numpy()
    u_true = compute_analytical_solution(x_np, t_np)
    
    # Compute metrics
    mse = np.mean((u_pred - u_true)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(u_pred - u_true))
    max_error = np.max(np.abs(u_pred - u_true))
    
    metrics['mse'] = mse
    metrics['rmse'] = rmse
    metrics['mae'] = mae
    metrics['max_error'] = max_error
    
    # Test specific cases
    test_cases = {
        'initial_center': (0.5, 0.0, 1.0),  # sin(π/2) = 1
        'boundary_left': (0.0, 0.5, 0.0),   # boundary = 0
        'boundary_right': (1.0, 0.5, 0.0),  # boundary = 0
        'half_period': (0.5, 0.5, 0.0),     # cos(π/2) = 0
    }
    
    metrics['test_cases'] = {}
    
    for name, (x_val, t_val, expected) in test_cases.items():
        x = torch.tensor([[x_val]], device=device)
        t = torch.tensor([[t_val]], device=device)
        
        with torch.no_grad():
            pred = model(x, t).item()
        
        true_val = compute_analytical_solution(
            np.array([[x_val]]), t_val
        )[0, 0]
        
        error = abs(pred - true_val)
        metrics['test_cases'][name] = error
    
    return metrics


def evaluate_on_grid(
    model: nn.Module,
    nx: int = 100,
    nt: int = 100,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on a regular grid for visualization.
    
    Args:
        model: Trained PINN model
        nx: Number of spatial points
        nt: Number of temporal points
        device: Device to evaluate on
        
    Returns:
        X, T: Grid coordinates
        U_pred: Predicted values
        U_true: True values
        Error: Absolute error
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Create grid
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t)
    
    # Flatten for model input
    x_flat = torch.tensor(X.flatten(), dtype=torch.float32, device=device).reshape(-1, 1)
    t_flat = torch.tensor(T.flatten(), dtype=torch.float32, device=device).reshape(-1, 1)
    
    # Predict
    with torch.no_grad():
        u_pred_flat = model(x_flat, t_flat).cpu().numpy()
    
    # Reshape
    U_pred = u_pred_flat.reshape(nt, nx)
    
    # Compute true solution
    U_true = compute_analytical_solution(X, T)
    
    # Compute error
    Error = np.abs(U_pred - U_true)
    
    return X, T, U_pred, U_true, Error


def compute_conservation_metrics(
    model: nn.Module,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Compute physics conservation metrics.
    
    For the wave equation, energy should be conserved:
    E = ∫(u_t² + c²u_x²)dx
    
    Args:
        model: Trained PINN model
        device: Device to evaluate on
        
    Returns:
        Dictionary of conservation metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    metrics = {}
    
    # Sample points for integration
    n_points = 1000
    x = torch.linspace(0, 1, n_points, device=device).reshape(-1, 1)
    
    # Compute energy at different times
    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    energies = []
    
    for t_val in times:
        t = torch.full_like(x, t_val)
        x_grad = x.requires_grad_(True)
        t_grad = t.requires_grad_(True)
        
        u = model(x_grad, t_grad)
        
        # Compute derivatives
        u_x = torch.autograd.grad(
            u, x_grad, torch.ones_like(u),
            create_graph=False, retain_graph=True
        )[0]
        
        u_t = torch.autograd.grad(
            u, t_grad, torch.ones_like(u),
            create_graph=False, retain_graph=False
        )[0]
        
        # Energy density
        energy_density = u_t**2 + u_x**2
        
        # Integrate (simple trapezoidal rule)
        energy = torch.trapezoid(energy_density.squeeze(), x.squeeze())
        energies.append(energy.item())
    
    metrics['energies'] = energies
    metrics['energy_variation'] = np.std(energies) / np.mean(energies)
    
    return metrics
