"""
Radial-band sampling curriculum for UHG neighborhoods.

Implements progressive sampling from inner to outer bands to preserve hierarchical structure.
"""

import logging
from typing import List, Optional, Tuple, Union

import torch
import numpy as np

logger = logging.getLogger(__name__)


class RadialBandSampler:
    """
    Radial-band sampling curriculum for neighborhood selection.
    
    Samples neighbors in radial bands to preserve hierarchical structure
    in hyperbolic space.
    """
    
    def __init__(
        self,
        num_bands: int = 5,
        band_size: int = 20,
        min_band_size: int = 5,
        max_band_size: int = 50,
        sampling_strategy: str = "uniform"
    ):
        """
        Initialize radial band sampler.
        
        Args:
            num_bands: Number of radial bands
            band_size: Size of each band
            min_band_size: Minimum band size
            max_band_size: Maximum band size
            sampling_strategy: Strategy for sampling within bands ("uniform", "weighted", "random")
        """
        self.num_bands = num_bands
        self.band_size = band_size
        self.min_band_size = min_band_size
        self.max_band_size = max_band_size
        self.sampling_strategy = sampling_strategy
        
        logger.info(f"Initialized radial band sampler: {num_bands} bands, {band_size} per band")
    
    def sample_neighbors(
        self,
        neighbors: torch.Tensor,
        distances: torch.Tensor,
        max_neighbors: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample neighbors using radial band strategy.
        
        Args:
            neighbors: Neighbor indices [N]
            distances: Distances to neighbors [N]
            max_neighbors: Maximum number of neighbors to return
            
        Returns:
            Tuple of (sampled_neighbors, sampled_distances)
        """
        if len(neighbors) <= self.num_bands * self.band_size:
            # Not enough neighbors for band sampling, return all
            return neighbors, distances
        
        # Sort by distance
        sorted_indices = distances.argsort()
        sorted_neighbors = neighbors[sorted_indices]
        sorted_distances = distances[sorted_indices]
        
        # Sample from each band
        sampled_neighbors = []
        sampled_distances = []
        
        for i in range(self.num_bands):
            start_idx = i * self.band_size
            end_idx = min((i + 1) * self.band_size, len(sorted_neighbors))
            
            if start_idx < len(sorted_neighbors):
                band_neighbors = sorted_neighbors[start_idx:end_idx]
                band_distances = sorted_distances[start_idx:end_idx]
                
                # Sample within band
                if self.sampling_strategy == "uniform":
                    # Take all neighbors in band
                    sampled_neighbors.append(band_neighbors)
                    sampled_distances.append(band_distances)
                elif self.sampling_strategy == "weighted":
                    # Weighted sampling based on distance
                    weights = self._compute_band_weights(band_distances, i)
                    sampled_indices = self._weighted_sample(weights, len(band_neighbors))
                    sampled_neighbors.append(band_neighbors[sampled_indices])
                    sampled_distances.append(band_distances[sampled_indices])
                elif self.sampling_strategy == "random":
                    # Random sampling within band
                    n_sample = min(len(band_neighbors), self.band_size)
                    sampled_indices = torch.randperm(len(band_neighbors))[:n_sample]
                    sampled_neighbors.append(band_neighbors[sampled_indices])
                    sampled_distances.append(band_distances[sampled_indices])
        
        # Concatenate all bands
        if sampled_neighbors:
            final_neighbors = torch.cat(sampled_neighbors)
            final_distances = torch.cat(sampled_distances)
        else:
            final_neighbors = neighbors[:self.band_size]
            final_distances = distances[:self.band_size]
        
        # Limit to max_neighbors if specified
        if max_neighbors is not None and len(final_neighbors) > max_neighbors:
            final_neighbors = final_neighbors[:max_neighbors]
            final_distances = final_distances[:max_neighbors]
        
        return final_neighbors, final_distances
    
    def _compute_band_weights(self, distances: torch.Tensor, band_idx: int) -> torch.Tensor:
        """
        Compute weights for sampling within a band.
        
        Args:
            distances: Distances within the band
            band_idx: Band index
            
        Returns:
            Sampling weights
        """
        if self.sampling_strategy == "weighted":
            # Weight by inverse distance (closer neighbors get higher weight)
            weights = 1.0 / (distances + 1e-6)
            
            # Adjust weights based on band (inner bands get higher weight)
            band_weight = 1.0 / (band_idx + 1)
            weights = weights * band_weight
            
            return weights
        else:
            # Uniform weights
            return torch.ones_like(distances)
    
    def _weighted_sample(self, weights: torch.Tensor, n_sample: int) -> torch.Tensor:
        """
        Perform weighted sampling.
        
        Args:
            weights: Sampling weights
            n_sample: Number of samples to draw
            
        Returns:
            Indices of sampled elements
        """
        # Normalize weights
        weights = weights / weights.sum()
        
        # Sample indices
        sampled_indices = torch.multinomial(weights, n_sample, replacement=False)
        
        return sampled_indices


class ProgressiveSampler:
    """
    Progressive sampling curriculum that increases complexity over time.
    
    Starts with simple sampling and gradually increases complexity.
    """
    
    def __init__(
        self,
        initial_bands: int = 2,
        final_bands: int = 5,
        initial_band_size: int = 10,
        final_band_size: int = 20,
        warmup_steps: int = 1000
    ):
        """
        Initialize progressive sampler.
        
        Args:
            initial_bands: Initial number of bands
            final_bands: Final number of bands
            initial_band_size: Initial band size
            final_band_size: Final band size
            warmup_steps: Number of warmup steps
        """
        self.initial_bands = initial_bands
        self.final_bands = final_bands
        self.initial_band_size = initial_band_size
        self.final_band_size = final_band_size
        self.warmup_steps = warmup_steps
        
        self.current_step = 0
        
        logger.info(f"Initialized progressive sampler: {initial_bands}->{final_bands} bands, {initial_band_size}->{final_band_size} per band")
    
    def update_step(self, step: int) -> None:
        """
        Update current training step.
        
        Args:
            step: Current training step
        """
        self.current_step = step
    
    def get_current_params(self) -> Tuple[int, int]:
        """
        Get current sampling parameters based on training step.
        
        Returns:
            Tuple of (num_bands, band_size)
        """
        if self.current_step >= self.warmup_steps:
            # Use final parameters
            return self.final_bands, self.final_band_size
        
        # Interpolate between initial and final parameters
        progress = self.current_step / self.warmup_steps
        
        num_bands = int(self.initial_bands + (self.final_bands - self.initial_bands) * progress)
        band_size = int(self.initial_band_size + (self.final_band_size - self.initial_band_size) * progress)
        
        return num_bands, band_size
    
    def sample_neighbors(
        self,
        neighbors: torch.Tensor,
        distances: torch.Tensor,
        max_neighbors: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample neighbors using current parameters.
        
        Args:
            neighbors: Neighbor indices
            distances: Distances to neighbors
            max_neighbors: Maximum number of neighbors
            
        Returns:
            Tuple of (sampled_neighbors, sampled_distances)
        """
        # Get current parameters
        num_bands, band_size = self.get_current_params()
        
        # Create sampler with current parameters
        sampler = RadialBandSampler(
            num_bands=num_bands,
            band_size=band_size,
            sampling_strategy="uniform"
        )
        
        # Sample neighbors
        return sampler.sample_neighbors(neighbors, distances, max_neighbors)


class AdaptiveSampler:
    """
    Adaptive sampler that adjusts based on data characteristics.
    
    Automatically adjusts sampling parameters based on neighborhood structure.
    """
    
    def __init__(
        self,
        min_bands: int = 2,
        max_bands: int = 8,
        min_band_size: int = 5,
        max_band_size: int = 30,
        adaptation_rate: float = 0.1
    ):
        """
        Initialize adaptive sampler.
        
        Args:
            min_bands: Minimum number of bands
            max_bands: Maximum number of bands
            min_band_size: Minimum band size
            max_band_size: Maximum band size
            adaptation_rate: Rate of adaptation
        """
        self.min_bands = min_bands
        self.max_bands = max_bands
        self.min_band_size = min_band_size
        self.max_band_size = max_band_size
        self.adaptation_rate = adaptation_rate
        
        # Current parameters
        self.current_bands = (min_bands + max_bands) // 2
        self.current_band_size = (min_band_size + max_band_size) // 2
        
        # Statistics for adaptation
        self.distance_stats = []
        self.sampling_efficiency = []
        
        logger.info(f"Initialized adaptive sampler: {self.current_bands} bands, {self.current_band_size} per band")
    
    def update_statistics(
        self,
        distances: torch.Tensor,
        sampling_efficiency: float
    ) -> None:
        """
        Update statistics for adaptation.
        
        Args:
            distances: Distances from recent sampling
            sampling_efficiency: Efficiency of recent sampling
        """
        self.distance_stats.append(distances.mean().item())
        self.sampling_efficiency.append(sampling_efficiency)
        
        # Keep only recent statistics
        if len(self.distance_stats) > 100:
            self.distance_stats = self.distance_stats[-100:]
            self.sampling_efficiency = self.sampling_efficiency[-100:]
    
    def adapt_parameters(self) -> None:
        """
        Adapt sampling parameters based on recent statistics.
        """
        if len(self.sampling_efficiency) < 10:
            return
        
        # Compute recent efficiency
        recent_efficiency = np.mean(self.sampling_efficiency[-10:])
        
        # Adapt based on efficiency
        if recent_efficiency < 0.5:
            # Low efficiency, reduce complexity
            self.current_bands = max(self.min_bands, self.current_bands - 1)
            self.current_band_size = max(self.min_band_size, self.current_band_size - 2)
        elif recent_efficiency > 0.8:
            # High efficiency, increase complexity
            self.current_bands = min(self.max_bands, self.current_bands + 1)
            self.current_band_size = min(self.max_band_size, self.current_band_size + 2)
        
        logger.info(f"Adapted parameters: {self.current_bands} bands, {self.current_band_size} per band")
    
    def sample_neighbors(
        self,
        neighbors: torch.Tensor,
        distances: torch.Tensor,
        max_neighbors: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample neighbors using current adaptive parameters.
        
        Args:
            neighbors: Neighbor indices
            distances: Distances to neighbors
            max_neighbors: Maximum number of neighbors
            
        Returns:
            Tuple of (sampled_neighbors, sampled_distances)
        """
        # Create sampler with current parameters
        sampler = RadialBandSampler(
            num_bands=self.current_bands,
            band_size=self.current_band_size,
            sampling_strategy="uniform"
        )
        
        # Sample neighbors
        sampled_neighbors, sampled_distances = sampler.sample_neighbors(
            neighbors, distances, max_neighbors
        )
        
        # Update statistics
        efficiency = len(sampled_neighbors) / len(neighbors) if len(neighbors) > 0 else 0.0
        self.update_statistics(distances, efficiency)
        
        # Adapt parameters periodically
        if len(self.sampling_efficiency) % 50 == 0:
            self.adapt_parameters()
        
        return sampled_neighbors, sampled_distances


def create_sampler(
    sampler_type: str = "radial_band",
    **kwargs
) -> Union[RadialBandSampler, ProgressiveSampler, AdaptiveSampler]:
    """
    Create a sampler instance.
    
    Args:
        sampler_type: Type of sampler ("radial_band", "progressive", "adaptive")
        **kwargs: Additional arguments for sampler
        
    Returns:
        Sampler instance
    """
    if sampler_type == "radial_band":
        return RadialBandSampler(**kwargs)
    elif sampler_type == "progressive":
        return ProgressiveSampler(**kwargs)
    elif sampler_type == "adaptive":
        return AdaptiveSampler(**kwargs)
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")
