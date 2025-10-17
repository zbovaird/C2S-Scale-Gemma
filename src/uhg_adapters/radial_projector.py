"""
Radial projector for UHG to Euclidean space.

Implements monotone radial projection preserving radial order and angular information.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class RadialProjector(nn.Module):
    """
    Monotone radial projector from UHG (ℍ^d) to Euclidean (ℝ^d).
    
    Preserves radial order (monotonic in UHG radius) and maintains 
    angular information via normalized chordal map.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        projection_type: str = "monotone_radial",
        preserve_angular: bool = True,
        epsilon: float = 1e-6
    ):
        """
        Initialize radial projector.
        
        Args:
            input_dim: Input dimension (UHG space)
            output_dim: Output dimension (Euclidean space)
            projection_type: Type of projection ("monotone_radial", "chordal", "linear")
            preserve_angular: Whether to preserve angular information
            epsilon: Small value for numerical stability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection_type = projection_type
        self.preserve_angular = preserve_angular
        self.epsilon = epsilon
        
        # Initialize projection parameters
        if projection_type == "monotone_radial":
            # Monotone radial projection with learnable parameters
            self.radial_scale = nn.Parameter(torch.ones(1))
            self.radial_bias = nn.Parameter(torch.zeros(1))
            self.angular_scale = nn.Parameter(torch.ones(1))
        elif projection_type == "chordal":
            # Chordal projection (preserves angles exactly)
            self.scale = nn.Parameter(torch.ones(1))
        elif projection_type == "linear":
            # Linear projection
            self.projection_matrix = nn.Parameter(torch.randn(input_dim, output_dim) * 0.1)
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")
        
        logger.info(f"Initialized {projection_type} projector: {input_dim} -> {output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through radial projector.
        
        Args:
            x: Input tensor in UHG space [..., input_dim]
            
        Returns:
            Projected tensor in Euclidean space [..., output_dim]
        """
        if self.projection_type == "monotone_radial":
            return self._monotone_radial_projection(x)
        elif self.projection_type == "chordal":
            return self._chordal_projection(x)
        elif self.projection_type == "linear":
            return self._linear_projection(x)
        else:
            raise ValueError(f"Unknown projection type: {self.projection_type}")
    
    def _monotone_radial_projection(self, x: torch.Tensor) -> torch.Tensor:
        """
        Monotone radial projection preserving radial order.
        
        Args:
            x: Input tensor in UHG space
            
        Returns:
            Projected tensor in Euclidean space
        """
        # Compute UHG radius (distance from origin)
        uhg_radius = self._compute_uhg_radius(x)
        
        # Apply monotone transformation to radius
        # f(r) = scale * r + bias, ensuring monotonicity
        projected_radius = self.radial_scale * uhg_radius + self.radial_bias
        
        if self.preserve_angular:
            # Preserve angular information via normalized chordal map
            angular_component = self._preserve_angular_information(x, uhg_radius)
            
            # Combine radial and angular components
            if self.output_dim == self.input_dim:
                # Same dimension: scale by projected radius
                output = angular_component * projected_radius.unsqueeze(-1)
            else:
                # Different dimension: project angular component
                output = self._project_angular_component(angular_component, projected_radius)
        else:
            # Simple radial projection without angular preservation
            if self.output_dim == self.input_dim:
                output = x * (projected_radius / (uhg_radius + self.epsilon)).unsqueeze(-1)
            else:
                # Project to different dimension
                output = self._project_to_output_dim(x, projected_radius)
        
        return output
    
    def _chordal_projection(self, x: torch.Tensor) -> torch.Tensor:
        """
        Chordal projection preserving angles exactly.
        
        Args:
            x: Input tensor in UHG space
            
        Returns:
            Projected tensor in Euclidean space
        """
        # Chordal projection: x -> x / (1 + sqrt(1 + ||x||^2))
        norm_squared = torch.sum(x**2, dim=-1, keepdim=True)
        denominator = 1 + torch.sqrt(1 + norm_squared)
        
        # Apply scaling
        output = self.scale * x / denominator
        
        # Project to output dimension if needed
        if self.output_dim != self.input_dim:
            output = self._project_to_output_dim(output, torch.ones_like(norm_squared))
        
        return output
    
    def _linear_projection(self, x: torch.Tensor) -> torch.Tensor:
        """
        Linear projection.
        
        Args:
            x: Input tensor in UHG space
            
        Returns:
            Projected tensor in Euclidean space
        """
        # Linear transformation
        output = torch.matmul(x, self.projection_matrix) + self.bias
        return output
    
    def _compute_uhg_radius(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute UHG radius (distance from origin).
        
        Args:
            x: Input tensor in UHG space
            
        Returns:
            UHG radius
        """
        # For UHG, radius is computed using the Minkowski inner product
        # r = arccosh(sqrt(1 + ||x||^2))
        norm_squared = torch.sum(x**2, dim=-1)
        radius = torch.acosh(torch.sqrt(1 + norm_squared + self.epsilon))
        return radius
    
    def _preserve_angular_information(self, x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        """
        Preserve angular information via normalized chordal map.
        
        Args:
            x: Input tensor in UHG space
            radius: UHG radius
            
        Returns:
            Angular component preserving angles
        """
        # Normalize by radius to get angular component
        angular_component = x / (radius.unsqueeze(-1) + self.epsilon)
        
        # Apply angular scaling
        angular_component = angular_component * self.angular_scale
        
        return angular_component
    
    def _project_angular_component(self, angular_component: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        """
        Project angular component to output dimension.
        
        Args:
            angular_component: Angular component
            radius: Projected radius
            
        Returns:
            Projected tensor
        """
        # Simple projection: pad or truncate to output dimension
        if self.output_dim > self.input_dim:
            # Pad with zeros
            padding_size = self.output_dim - self.input_dim
            padding = torch.zeros(*angular_component.shape[:-1], padding_size, device=angular_component.device)
            projected_angular = torch.cat([angular_component, padding], dim=-1)
        else:
            # Truncate
            projected_angular = angular_component[..., :self.output_dim]
        
        # Scale by radius
        output = projected_angular * radius.unsqueeze(-1)
        
        return output
    
    def _project_to_output_dim(self, x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        """
        Project tensor to output dimension.
        
        Args:
            x: Input tensor
            radius: Radius for scaling
            
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
        
        # Scale by radius
        output = output * radius.unsqueeze(-1)
        
        return output
    
    def get_monotonicity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute monotonicity loss to ensure radial order is preserved.
        
        Args:
            x: Input tensor in UHG space
            
        Returns:
            Monotonicity loss
        """
        if self.projection_type != "monotone_radial":
            return torch.tensor(0.0, device=x.device)
        
        # Compute UHG radii
        uhg_radius = self._compute_uhg_radius(x)
        
        # Compute projected radii
        projected_radius = self.radial_scale * uhg_radius + self.radial_bias
        
        # Monotonicity loss: penalize negative gradients
        # This ensures that larger UHG radii map to larger projected radii
        radius_diff = projected_radius[1:] - projected_radius[:-1]
        uhg_diff = uhg_radius[1:] - uhg_radius[:-1]
        
        # Penalize cases where projected radius decreases when UHG radius increases
        monotonicity_violations = torch.relu(-radius_diff * uhg_diff)
        monotonicity_loss = torch.mean(monotonicity_violations)
        
        return monotonicity_loss
    
    def get_angular_preservation_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute angular preservation loss.
        
        Args:
            x: Input tensor in UHG space
            
        Returns:
            Angular preservation loss
        """
        if not self.preserve_angular or self.projection_type != "monotone_radial":
            return torch.tensor(0.0, device=x.device)
        
        # Compute angular components before and after projection
        uhg_radius = self._compute_uhg_radius(x)
        angular_before = x / (uhg_radius.unsqueeze(-1) + self.epsilon)
        
        # Project and compute angular component after projection
        projected = self.forward(x)
        projected_radius = torch.norm(projected, dim=-1)
        angular_after = projected / (projected_radius.unsqueeze(-1) + self.epsilon)
        
        # Angular preservation loss: minimize difference in angular components
        angular_loss = torch.mean(torch.sum((angular_before - angular_after)**2, dim=-1))
        
        return angular_loss


class EuclideanToUHG(nn.Module):
    """
    Inverse mapping from Euclidean space to UHG space.
    
    Maps Euclidean vectors back to hyperbolic space.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        projection_type: str = "inverse_chordal"
    ):
        """
        Initialize Euclidean to UHG projector.
        
        Args:
            input_dim: Input dimension (Euclidean space)
            output_dim: Output dimension (UHG space)
            projection_type: Type of inverse projection
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection_type = projection_type
        
        if projection_type == "inverse_chordal":
            self.scale = nn.Parameter(torch.ones(1))
        elif projection_type == "linear":
            self.projection_matrix = nn.Parameter(torch.randn(input_dim, output_dim) * 0.1)
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            raise ValueError(f"Unknown inverse projection type: {projection_type}")
        
        logger.info(f"Initialized {projection_type} inverse projector: {input_dim} -> {output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass from Euclidean to UHG space.
        
        Args:
            x: Input tensor in Euclidean space
            
        Returns:
            Projected tensor in UHG space
        """
        if self.projection_type == "inverse_chordal":
            return self._inverse_chordal_projection(x)
        elif self.projection_type == "linear":
            return self._linear_projection(x)
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
        denominator = torch.clamp(1 - norm_squared, min=1e-6)
        
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
