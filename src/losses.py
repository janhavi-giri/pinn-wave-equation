"""
Physics-informed loss functions for training PINNs.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional


class PhysicsInformedLoss:
    """
    Computes the physics-informed loss for the wave equation.
    
    The total loss consists of:
    - PDE residual loss (wave equation satisfaction)
    - Initial condition loss
    - Initial velocity loss
    - Boundary condition loss
    
    Args:
        wave_speed: Speed of wave propagation (default: 1.0)
        pde_weight: Weight for PDE loss term
        ic_weight: Weight for initial condition loss
        ic_vel_weight: Weight for initial velocity loss
        bc_weight: Weight for boundary condition loss
    """
    
    def __init__(
        self,
        wave_speed: float = 1.0,
        pde_weight: float = 1.0,
        ic_weight: float = 50.0,
        ic_vel_weight: float = 50.0,
        bc_weight: float = 50.0
    ):
        self.c = wave_speed
        self.w_pde = pde_weight
        self.w_ic = ic_weight
        self.w_ic_vel = ic_vel_weight
        self.w_bc = bc_weight
    
    def compute_pde_loss(
        self, 
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute PDE residual loss for the wave equation.
        
        Wave equation: ∂²u/∂t² = c² ∂²u/∂x²
        """
        # Enable gradients
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)
        
        # Forward pass
        u = model(x, t)
        
        # First derivatives
        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        # Second derivatives
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0]
        
        u_tt = torch.autograd.grad(
            u_t, t, grad_outputs=torch.ones_like(u_t),
            create_graph=True, retain_graph=True
        )[0]
        
        # Wave equation residual
        pde_residual = u_tt - self.c**2 * u_xx
        
        return torch.mean(pde_residual**2)
    
    def compute_ic_loss(
        self,
        model: nn.Module,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute initial condition loss.
        
        Initial condition: u(x, 0) = sin(πx)
        """
        t = torch.zeros_like(x)
        u_pred = model(x, t)
        u_true = torch.sin(torch.pi * x)
        
        return torch.mean((u_pred - u_true)**2)
    
    def compute_ic_velocity_loss(
        self,
        model: nn.Module,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute initial velocity condition loss.
        
        Initial velocity: ∂u/∂t(x, 0) = 0
        """
        x = x.clone().detach().requires_grad_(True)
        t = torch.zeros_like(x).requires_grad_(True)
        
        u = model(x, t)
        
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        return torch.mean(u_t**2)
    
    def compute_bc_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute boundary condition loss.
        
        Boundary conditions: u(0, t) = u(1, t) = 0
        """
        u_pred = model(x, t)
        return torch.mean(u_pred**2)
    
    def __call__(
        self,
        model: nn.Module,
        x_pde: torch.Tensor,
        t_pde: torch.Tensor,
        x_ic: torch.Tensor,
        x_bc: torch.Tensor,
        t_bc: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total physics-informed loss.
        
        Returns:
            total_loss: Weighted sum of all loss components
            loss_dict: Dictionary containing individual loss values
        """
        # Compute individual losses
        pde_loss = self.compute_pde_loss(model, x_pde, t_pde)
        ic_loss = self.compute_ic_loss(model, x_ic)
        ic_vel_loss = self.compute_ic_velocity_loss(model, x_ic)
        bc_loss = self.compute_bc_loss(model, x_bc, t_bc)
        
        # Total weighted loss
        total_loss = (
            self.w_pde * pde_loss +
            self.w_ic * ic_loss +
            self.w_ic_vel * ic_vel_loss +
            self.w_bc * bc_loss
        )
        
        # Store individual losses for logging
        loss_dict = {
            'total': total_loss.item(),
            'pde': pde_loss.item(),
            'ic': ic_loss.item(),
            'ic_vel': ic_vel_loss.item(),
            'bc': bc_loss.item()
        }
        
        return total_loss, loss_dict


class AdaptivePhysicsInformedLoss(PhysicsInformedLoss):
    """
    Physics-informed loss with adaptive weighting.
    
    Automatically adjusts weights during training based on the
    relative magnitudes of different loss components.
    """
    
    def __init__(self, *args, adapt_freq: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapt_freq = adapt_freq
        self.iteration = 0
        self.loss_history = {
            'pde': [],
            'ic': [],
            'ic_vel': [],
            'bc': []
        }
    
    def adapt_weights(self):
        """Adapt weights based on loss history."""
        if len(self.loss_history['pde']) < 10:
            return
        
        # Compute recent average losses
        recent_losses = {}
        for key in self.loss_history:
            recent_losses[key] = sum(self.loss_history[key][-10:]) / 10
        
        # Compute scaling factors
        total_loss = sum(recent_losses.values())
        if total_loss > 0:
            target_contribution = 0.25  # Each component should contribute 25%
            
            self.w_pde *= target_contribution / (recent_losses['pde'] / total_loss + 1e-8)
            self.w_ic *= target_contribution / (recent_losses['ic'] / total_loss + 1e-8)
            self.w_ic_vel *= target_contribution / (recent_losses['ic_vel'] / total_loss + 1e-8)
            self.w_bc *= target_contribution / (recent_losses['bc'] / total_loss + 1e-8)
            
            # Normalize weights
            weight_sum = self.w_pde + self.w_ic + self.w_ic_vel + self.w_bc
            self.w_pde /= weight_sum
            self.w_ic /= weight_sum
            self.w_ic_vel /= weight_sum
            self.w_bc /= weight_sum
    
    def __call__(self, *args, **kwargs):
        """Compute loss with adaptive weighting."""
        total_loss, loss_dict = super().__call__(*args, **kwargs)
        
        # Update history
        self.loss_history['pde'].append(loss_dict['pde'])
        self.loss_history['ic'].append(loss_dict['ic'])
        self.loss_history['ic_vel'].append(loss_dict['ic_vel'])
        self.loss_history['bc'].append(loss_dict['bc'])
        
        # Adapt weights periodically
        self.iteration += 1
        if self.iteration % self.adapt_freq == 0:
            self.adapt_weights()
        
        return total_loss, loss_dict
