"""
Euclidean to UHG inverse mapping.

Implements inverse projection from Euclidean space back to UHG space.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class EuclideanToUHG(nn.Module):
    """
    Inverse mapping from Euclidean space to UHG space.
    
    Maps Euclidean vectors back to hyperbolic space using various projection methods.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        projection_type: str = "inverse_chordal",
        epsilon: float = 1e-6
    ):
        """
        Initialize Euclidean to UHG projector.
        
        Args:
            input_dim: Input dimension (Euclidean space)
            output_dim: Output dimension (UHG space)
            projection_type: Type of inverse projection ("inverse_chordal", "linear", "learned")
            epsilon: Small value for numerical stability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection_type = projection_type
        self.epsilon = epsilon
        
        if projection_type == "inverse_chordal":
            self.scale = nn.Parameter(torch.ones(1))
        elif projection_type == "linear":
            self.projection_matrix = nn.Parameter(torch.randn(input_dim, output_dim) * 0.1)
            self.bias = nn.Parameter(torch.zeros(output_dim))
        elif projection_type == "learned":
            # Learnable inverse projection
            self.scale = nn.Parameter(torch.ones(1))
            self.projection_matrix = nn.Parameter(torch.randn(input_dim, output_dim) * 0.1)
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            raise ValueError(f"Unknown inverse projection type: {projection_type}")
        
        logger.info(f"Initialized {projection_type} inverse projector: {input_dim} -> {output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass from Euclidean to UHG space.
        
        Args:
            x: Input tensor in Euclidean space [..., input_dim]
            
        Returns:
            Projected tensor in UHG space [..., output_dim]
        """
        if self.projection_type == "inverse_chordal":
            return self._inverse_chordal_projection(x)
        elif self.projection_type == "linear":
            return self._linear_projection(x)
        elif self.projection_type == "learned":
            return self._learned_projection(x)
        else:
            raise ValueError(f"Unknown inverse projection type: {self.projection_type}")
    
    def _inverse_chordal_projection(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse chordal projection.
        
        Args:
            x: Input tensor in Euclidean space
            
        Returns:
            Projected tensor in UHG space
        """
        # Inverse chordal projection: x -> x / (1 - ||x||^2)
        norm_squared = torch.sum(x**2, dim=-1, keepdim=True)
        
        # Ensure we don't divide by zero or negative values
        denominator = torch.clamp(1 - norm_squared, min=self.epsilon)
        
        # Apply scaling
        output = self.scale * x / denominator
        
        # Project to output dimension if needed
        if self.output_dim != self.input_dim:
            output = self._project_to_output_dim(output)
        
        return output
    
    def _linear_projection(self, x: torch.Tensor) -> torch.Tensor:
        """
        Linear projection.
        
        Args:
            x: Input tensor in Euclidean space
            
        Returns:
            Projected tensor in UHG space
        """
        output = torch.matmul(x, self.projection_matrix) + self.bias
        return output
    
    def _learned_projection(self, x: torch.Tensor) -> torch.Tensor:
        """
        Learned projection combining chordal and linear components.
        
        Args:
            x: Input tensor in Euclidean space
            
        Returns:
            Projected tensor in UHG space
        """
        # Linear component
        linear_output = torch.matmul(x, self.projection_matrix) + self.bias
        
        # Chordal component
        norm_squared = torch.sum(x**2, dim=-1, keepdim=True)
        denominator = torch.clamp(1 - norm_squared, min=self.epsilon)
        chordal_output = self.scale * x / denominator
        
        # Combine components
        if self.output_dim == self.input_dim:
            # Same dimension: weighted combination
            output = 0.5 * linear_output + 0.5 * chordal_output
        else:
            # Different dimension: use linear projection
            output = linear_output
        
        return output
    
    def _project_to_output_dim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project tensor to output dimension.
        
        Args:
            x: Input tensor
            
        Returns:
            Projected tensor
        """
        if self.output_dim > self.input_dim:
            # Pad with zeros
            padding_size = self.output_dim - self.input_dim
            padding = torch.zeros(*x.shape[:-1], padding_size, device=x.device)
            output = torch.cat([x, padding], dim=-1)
        else:
            # Truncate
            output = x[..., :self.output_dim]
        
        return output
    
    def get_reconstruction_loss(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss for inverse projection.
        
        Args:
            original: Original UHG tensor
            reconstructed: Reconstructed UHG tensor
            
        Returns:
            Reconstruction loss
        """
        # L2 reconstruction loss
        l2_loss = torch.mean(torch.sum((original - reconstructed)**2, dim=-1))
        
        # Add regularization for numerical stability
        reg_loss = torch.mean(torch.sum(reconstructed**2, dim=-1))
        
        return l2_loss + 0.01 * reg_loss


class BidirectionalProjector(nn.Module):
    """
    Bidirectional projector between UHG and Euclidean spaces.
    
    Combines forward and inverse projections with consistency loss.
    """
    
    def __init__(
        self,
        uhg_dim: int,
        euclidean_dim: int,
        forward_type: str = "monotone_radial",
        inverse_type: str = "inverse_chordal"
    ):
        """
        Initialize bidirectional projector.
        
        Args:
            uhg_dim: UHG space dimension
            euclidean_dim: Euclidean space dimension
            forward_type: Type of forward projection
            inverse_type: Type of inverse projection
        """
        super().__init__()
        
        self.uhg_dim = uhg_dim
        self.euclidean_dim = euclidean_dim
        
        # Forward projector (UHG -> Euclidean)
        from .radial_projector import RadialProjector
        self.forward_projector = RadialProjector(
            input_dim=uhg_dim,
            output_dim=euclidean_dim,
            projection_type=forward_type
        )
        
        # Inverse projector (Euclidean -> UHG)
        self.inverse_projector = EuclideanToUHG(
            input_dim=euclidean_dim,
            output_dim=uhg_dim,
            projection_type=inverse_type
        )
        
        logger.info(f"Initialized bidirectional projector: {uhg_dim} <-> {euclidean_dim}")
    
    def forward(self, x: torch.Tensor, direction: str = "forward") -> torch.Tensor:
        """
        Forward pass through bidirectional projector.
        
        Args:
            x: Input tensor
            direction: Direction of projection ("forward" or "inverse")
            
        Returns:
            Projected tensor
        """
        if direction == "forward":
            return self.forward_projector(x)
        elif direction == "inverse":
            return self.inverse_projector(x)
        else:
            raise ValueError(f"Unknown direction: {direction}")
    
    def get_consistency_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute consistency loss for bidirectional projection.
        
        Args:
            x: Input tensor in UHG space
            
        Returns:
            Consistency loss
        """
        # Forward projection
        euclidean = self.forward_projector(x)
        
        # Inverse projection
        reconstructed = self.inverse_projector(euclidean)
        
        # Consistency loss: original should be close to reconstructed
        consistency_loss = torch.mean(torch.sum((x - reconstructed)**2, dim=-1))
        
        return consistency_loss
    
    def get_cycle_consistency_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute cycle consistency loss.
        
        Args:
            x: Input tensor in UHG space
            
        Returns:
            Cycle consistency loss
        """
        # UHG -> Euclidean -> UHG
        euclidean = self.forward_projector(x)
        reconstructed = self.inverse_projector(euclidean)
        cycle_loss_1 = torch.mean(torch.sum((x - reconstructed)**2, dim=-1))
        
        # Euclidean -> UHG -> Euclidean
        uhg = self.inverse_projector(euclidean)
        reconstructed_euclidean = self.forward_projector(uhg)
        cycle_loss_2 = torch.mean(torch.sum((euclidean - reconstructed_euclidean)**2, dim=-1))
        
        return cycle_loss_1 + cycle_loss_2


def create_inverse_projector(
    input_dim: int,
    output_dim: int,
    projection_type: str = "inverse_chordal"
) -> EuclideanToUHG:
    """
    Create an inverse projector instance.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        projection_type: Type of projection
        
    Returns:
        Inverse projector instance
    """
    return EuclideanToUHG(
        input_dim=input_dim,
        output_dim=output_dim,
        projection_type=projection_type
    )


def create_bidirectional_projector(
    uhg_dim: int,
    euclidean_dim: int,
    forward_type: str = "monotone_radial",
    inverse_type: str = "inverse_chordal"
) -> BidirectionalProjector:
    """
    Create a bidirectional projector instance.
    
    Args:
        uhg_dim: UHG space dimension
        euclidean_dim: Euclidean space dimension
        forward_type: Type of forward projection
        inverse_type: Type of inverse projection
        
    Returns:
        Bidirectional projector instance
    """
    return BidirectionalProjector(
        uhg_dim=uhg_dim,
        euclidean_dim=euclidean_dim,
        forward_type=forward_type,
        inverse_type=inverse_type
    )
