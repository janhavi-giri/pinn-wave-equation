"""
Simple demonstration of Physics-Informed Neural Networks for the wave equation.

This script shows the basic workflow:
1. Create a PINN model
2. Train it to solve the wave equation
3. Evaluate and visualize results
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.model import WavePINN
from src.train import train_model
from src.evaluate import evaluate_model
from src.visualization import plot_wave_evolution, plot_training_history


def main():
    """Run a simple PINN demo."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("\n1. Creating PINN model...")
    model = WavePINN().to(device)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    print("\n2. Training model...")
    history = train_model(
        model,
        epochs=5000,
        lr=1e-3,
        device=device,
        log_every=1000
    )
    
    # Evaluate model
    print("\n3. Evaluating model...")
    metrics = evaluate_model(model, device=device)
    
    print("\nTest Results:")
    print("-" * 40)
    for test_name, error in metrics['test_cases'].items():
        print(f"{test_name}: Error = {error:.4f}")
    
    print(f"\nOverall Metrics:")
    print(f"  Mean Squared Error: {metrics['mse']:.6f}")
    print(f"  Maximum Error: {metrics['max_error']:.4f}")
    
    # Visualize results
    print("\n4. Creating visualizations...")
    
    # Training history
    plot_training_history(history, save_path='outputs/training_history.png')
    
    # Wave evolution
    plot_wave_evolution(model, device=device, save_path='outputs/wave_evolution.png')
    
    print("\nDemo completed! Check the 'outputs' directory for visualizations.")


if __name__ == "__main__":
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # Run demo
    main()
