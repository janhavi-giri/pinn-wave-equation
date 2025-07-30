"""
Physics-Informed Neural Networks for Wave Equation

A clean implementation of PINNs for solving the 1D wave equation.
"""

from .model import WavePINN, ImprovedWavePINN, create_model
from .losses import PhysicsInformedLoss, AdaptivePhysicsInformedLoss
from .train import Trainer, train_model
from .evaluate import evaluate_model, compute_analytical_solution
from .visualization import (
    plot_wave_evolution,
    plot_training_history,
    plot_error_heatmap,
    create_summary_figure
)

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    # Models
    "WavePINN",
    "ImprovedWavePINN",
    "create_model",
    
    # Losses
    "PhysicsInformedLoss",
    "AdaptivePhysicsInformedLoss",
    
    # Training
    "Trainer",
    "train_model",
    
    # Evaluation
    "evaluate_model",
    "compute_analytical_solution",
    
    # Visualization
    "plot_wave_evolution",
    "plot_training_history",
    "plot_error_heatmap",
    "create_summary_figure",
]
