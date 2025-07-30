"""
Generate all visualizations for the PINN wave equation project.

This script creates:
1. Before/After training comparison
2. Training history plots
3. Wave evolution visualization
4. Error heatmap
5. Summary dashboard
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
from src.visualization import (
    plot_wave_evolution,
    plot_training_history,
    plot_error_heatmap,
    create_summary_figure,
    plot_untrained_vs_trained
)


def main():
    """Generate all visualizations."""
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models
    print("\n1. Creating models...")
    model_untrained = WavePINN().to(device)
    model = WavePINN().to(device)
    
    # Train model
    print("\n2. Training model...")
    history = train_model(
        model,
        epochs=5000,
        lr=1e-3,
        device=device,
        log_every=1000,
        verbose=True
    )
    
    # Generate visualizations
    print("\n3. Generating visualizations...")
    
    # Before/After comparison
    print("   - Creating before/after comparison...")
    plot_untrained_vs_trained(
        model_trained=model,
        model_untrained=model_untrained,
        t_snapshot=0.5,
        device=device,
        save_path='outputs/before_after_training.png'
    )
    
    # Training history
    print("   - Creating training history plots...")
    plot_training_history(
        history,
        save_path='outputs/training_history.png'
    )
    
    # Wave evolution
    print("   - Creating wave evolution visualization...")
    plot_wave_evolution(
        model,
        times=[0.0, 0.25, 0.5, 0.75, 1.0],
        device=device,
        save_path='outputs/wave_evolution.png'
    )
    
    # Error heatmap
    print("   - Creating error heatmap...")
    plot_error_heatmap(
        model,
        nx=100,
        nt=100,
        device=device,
        save_path='outputs/error_heatmap.png'
    )
    
    # Summary dashboard
    print("   - Creating summary dashboard...")
    create_summary_figure(
        model,
        history,
        device=device,
        save_path='outputs/summary_dashboard.png'
    )
    
    # Evaluate and print metrics
    print("\n4. Final evaluation...")
    metrics = evaluate_model(model, device=device)
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"Max Error: {metrics['max_error']:.4f}")
    print(f"Mean Absolute Error: {metrics['mae']:.6f}")
    
    print("\nTest Cases:")
    for name, error in metrics['test_cases'].items():
        print(f"  {name}: {error:.6f}")
    
    print("\n✅ All visualizations saved to 'outputs/' directory!")
    
    # Create a simple README for outputs
    with open('outputs/README.md', 'w') as f:
        f.write("# Generated Visualizations\n\n")
        f.write("This directory contains all generated plots:\n\n")
        f.write("- `before_after_training.png`: Comparison of untrained vs trained model\n")
        f.write("- `training_history.png`: Loss evolution during training\n")
        f.write("- `wave_evolution.png`: Wave propagation at different times\n")
        f.write("- `error_heatmap.png`: Error distribution over space-time\n")
        f.write("- `summary_dashboard.png`: Comprehensive results summary\n")
        f.write(f"\nGenerated with device: {device}\n")
        f.write(f"Final MSE: {metrics['mse']:.6f}\n")
        f.write(f"Max Error: {metrics['max_error']:.4f}\n")


if __name__ == "__main__":
    main()
