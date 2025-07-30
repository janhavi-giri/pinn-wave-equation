"""
Training utilities for Physics-Informed Neural Networks.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import time
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from .losses import PhysicsInformedLoss


class DataGenerator:
    """
    Generates training data points for PINN training.
    
    Args:
        device: torch device to place tensors on
        domain: Spatial domain bounds (x_min, x_max)
        time_domain: Temporal domain bounds (t_min, t_max)
    """
    
    def __init__(
        self,
        device: torch.device = None,
        domain: Tuple[float, float] = (0.0, 1.0),
        time_domain: Tuple[float, float] = (0.0, 1.0)
    ):
        self.device = device or torch.device('cpu')
        self.x_min, self.x_max = domain
        self.t_min, self.t_max = time_domain
    
    def generate_pde_points(self, n_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random collocation points for PDE loss."""
        x = torch.rand(n_points, 1, device=self.device) * (self.x_max - self.x_min) + self.x_min
        t = torch.rand(n_points, 1, device=self.device) * (self.t_max - self.t_min) + self.t_min
        return x, t
    
    def generate_ic_points(self, n_points: int) -> torch.Tensor:
        """Generate points for initial condition."""
        x = torch.rand(n_points, 1, device=self.device) * (self.x_max - self.x_min) + self.x_min
        return x
    
    def generate_bc_points(self, n_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate points for boundary conditions."""
        # Half points at x=0, half at x=1
        n_half = n_points // 2
        x_left = torch.full((n_half, 1), self.x_min, device=self.device)
        x_right = torch.full((n_points - n_half, 1), self.x_max, device=self.device)
        x = torch.cat([x_left, x_right], dim=0)
        t = torch.rand(n_points, 1, device=self.device) * (self.t_max - self.t_min) + self.t_min
        return x, t


class Trainer:
    """
    Trainer class for Physics-Informed Neural Networks.
    
    Args:
        model: PINN model to train
        loss_fn: Physics-informed loss function
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to train on
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: PhysicsInformedLoss,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device('cpu')
        
        self.model.to(self.device)
        self.data_generator = DataGenerator(device=self.device)
        
        # Training history
        self.history = {
            'loss': [],
            'pde': [],
            'ic': [],
            'ic_vel': [],
            'bc': [],
            'lr': []
        }
    
    def train_step(
        self,
        n_pde: int = 1000,
        n_ic: int = 200,
        n_bc: int = 200
    ) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            n_pde: Number of PDE collocation points
            n_ic: Number of initial condition points
            n_bc: Number of boundary condition points
            
        Returns:
            Dictionary of loss values
        """
        # Generate training data
        x_pde, t_pde = self.data_generator.generate_pde_points(n_pde)
        x_ic = self.data_generator.generate_ic_points(n_ic)
        x_bc, t_bc = self.data_generator.generate_bc_points(n_bc)
        
        # Compute loss
        loss, loss_dict = self.loss_fn(
            self.model, x_pde, t_pde, x_ic, x_bc, t_bc
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update weights
        self.optimizer.step()
        
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Record current learning rate
        loss_dict['lr'] = self.optimizer.param_groups[0]['lr']
        
        return loss_dict
    
    def train(
        self,
        epochs: int,
        n_pde: int = 1000,
        n_ic: int = 200,
        n_bc: int = 200,
        log_every: int = 100,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the PINN model.
        
        Args:
            epochs: Number of training epochs
            n_pde: Number of PDE collocation points per epoch
            n_ic: Number of initial condition points per epoch
            n_bc: Number of boundary condition points per epoch
            log_every: Log progress every N epochs
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        self.model.train()
        
        if verbose:
            print(f"Training on {self.device}")
            print(f"Epochs: {epochs}")
            print(f"Points per epoch - PDE: {n_pde}, IC: {n_ic}, BC: {n_bc}")
            print("-" * 60)
        
        start_time = time.time()
        
        # Progress bar
        pbar = tqdm(range(epochs), desc="Training", disable=not verbose)
        
        for epoch in pbar:
            # Training step
            loss_dict = self.train_step(n_pde, n_ic, n_bc)
            
            # Record history
            for key in self.history:
                if key in loss_dict:
                    self.history[key].append(loss_dict[key])
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.6f}",
                'pde': f"{loss_dict['pde']:.6f}"
            })
            
            # Detailed logging
            if verbose and (epoch + 1) % log_every == 0:
                elapsed = time.time() - start_time
                print(f"\nEpoch {epoch + 1}/{epochs} "
                      f"[{elapsed:.1f}s] - "
                      f"Loss: {loss_dict['total']:.6f}, "
                      f"PDE: {loss_dict['pde']:.6f}, "
                      f"IC: {loss_dict['ic']:.6f}, "
                      f"BC: {loss_dict['bc']:.6f}")
        
        training_time = time.time() - start_time
        
        if verbose:
            print(f"\nTraining completed in {training_time:.2f} seconds")
            print(f"Final loss: {self.history['loss'][-1]:.6f}")
        
        return self.history


def train_model(
    model: nn.Module,
    epochs: int = 5000,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    **kwargs
) -> Dict[str, List[float]]:
    """
    Convenience function to train a PINN model with default settings.
    
    Args:
        model: PINN model to train
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        **kwargs: Additional arguments passed to Trainer.train()
        
    Returns:
        Training history
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create loss function
    loss_fn = PhysicsInformedLoss()
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1000, gamma=0.5
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )
    
    # Train
    history = trainer.train(epochs=epochs, **kwargs)
    
    return history
