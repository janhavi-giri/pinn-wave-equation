"""
Visualization utilities for Physics-Informed Neural Networks.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Tuple
import seaborn as sns

from .evaluate import compute_analytical_solution, evaluate_on_grid


def plot_wave_evolution(
    model: nn.Module,
    times: Optional[List[float]] = None,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot wave evolution at different time snapshots.
    
    Args:
        model: Trained PINN model
        times: List of time values to plot
        device: Device for computation
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    if times is None:
        times = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Create figure
    n_times = len(times)
    n_cols = min(3, n_times)
    n_rows = (n_times + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_times > 1 else [axes]
    
    # Spatial points
    x = torch.linspace(0, 1, 200, device=device).reshape(-1, 1)
    x_np = x.cpu().numpy()
    
    for idx, t_val in enumerate(times):
        ax = axes[idx]
        
        # Model prediction
        t = torch.full_like(x, t_val)
        with torch.no_grad():
            u_pred = model(x, t).cpu().numpy()
        
        # Analytical solution
        u_true = compute_analytical_solution(x_np, t_val)
        
        # Plot
        ax.plot(x_np, u_true, 'k--', linewidth=2, label='Analytical', alpha=0.8)
        ax.plot(x_np, u_pred, 'r-', linewidth=2.5, label='PINN')
        
        # Error
        error = np.abs(u_pred - u_true)
        ax_twin = ax.twinx()
        ax_twin.fill_between(x_np.squeeze(), 0, error.squeeze(), 
                           alpha=0.3, color='orange', label='Error')
        ax_twin.set_ylabel('Absolute Error', color='orange')
        ax_twin.tick_params(axis='y', labelcolor='orange')
        ax_twin.set_ylim(0, max(0.1, np.max(error) * 1.2))
        
        ax.set_xlabel('x')
        ax.set_ylabel('u(x,t)')
        ax.set_title(f't = {t_val:.2f}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.5, 1.5)
        
        if idx == 0:
            ax.legend(loc='upper right')
    
    # Remove extra subplots
    for idx in range(n_times, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Wave Evolution: PINN vs Analytical Solution', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot training history with loss components.
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Total loss
    ax = axes[0, 0]
    ax.semilogy(history['loss'], 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss Evolution')
    ax.grid(True, alpha=0.3)
    
    # Add annotation for final loss
    final_loss = history['loss'][-1]
    ax.annotate(f'Final: {final_loss:.6f}',
                xy=(len(history['loss'])-1, final_loss),
                xytext=(len(history['loss'])*0.7, history['loss'][0]/10),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # Individual losses
    ax = axes[0, 1]
    ax.semilogy(history['pde'], label='PDE', linewidth=2)
    ax.semilogy(history['ic'], label='IC', linewidth=2)
    ax.semilogy(history['bc'], label='BC', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Loss reduction
    ax = axes[1, 0]
    reduction = np.array(history['loss']) / history['loss'][0]
    ax.plot(reduction, 'g-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss / Initial Loss')
    ax.set_title(f'Loss Reduction: {1/reduction[-1]:.1f}x')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Learning rate
    if 'lr' in history:
        ax = axes[1, 1]
        ax.plot(history['lr'], 'r-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    else:
        axes[1, 1].axis('off')
    
    plt.suptitle('Training History', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_error_heatmap(
    model: nn.Module,
    nx: int = 100,
    nt: int = 100,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot error heatmap over space-time domain.
    
    Args:
        model: Trained PINN model
        nx: Number of spatial points
        nt: Number of temporal points
        device: Device for computation
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    # Evaluate on grid
    X, T, U_pred, U_true, Error = evaluate_on_grid(model, nx, nt, device)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Predicted solution
    im1 = axes[0, 0].imshow(U_pred, aspect='auto', origin='lower',
                           extent=[0, 1, 0, 1], cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('t')
    axes[0, 0].set_title('PINN Solution')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # True solution
    im2 = axes[0, 1].imshow(U_true, aspect='auto', origin='lower',
                           extent=[0, 1, 0, 1], cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('t')
    axes[0, 1].set_title('Analytical Solution')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Absolute error
    im3 = axes[1, 0].imshow(Error, aspect='auto', origin='lower',
                           extent=[0, 1, 0, 1], cmap='hot_r')
    axes[1, 0].set_xlabel('x')
    axes[1, 