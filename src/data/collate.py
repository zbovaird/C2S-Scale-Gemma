"""
Collation functions for batching graph and text data.

Handles neighborhood sampling, padding, and batching for the hybrid model.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_text_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate batch of text data.
    
    Args:
        batch: List of text samples
        
    Returns:
        Batched text data
    """
    # Stack tensors
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    # Collect metadata
    cell_ids = [item['cell_id'] for item in batch]
    cell_types = [item['cell_type'] for item in batch]
    tissues = [item['tissue'] for item in batch]
    n_genes = torch.tensor([item['n_genes'] for item in batch])
    total_counts = torch.tensor([item['total_counts'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'cell_ids': cell_ids,
        'cell_types': cell_types,
        'tissues': tissues,
        'n_genes': n_genes,
        'total_counts': total_counts
    }


def collate_graph_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate batch of graph data.
    
    Args:
        batch: List of graph samples
        
    Returns:
        Batched graph data
    """
    result = {
        'cell_ids': [item['cell_id'] for item in batch],
        'cell_types': [item['cell_type'] for item in batch],
        'tissues': [item['tissue'] for item in batch]
    }
    
    # Handle different graph types
    graph_types = ['knn', 'lr', 'grn']
    
    for graph_type in graph_types:
        neighbor_key = f'{graph_type}_neighbors'
        interaction_key = f'{graph_type}_interactions'
        edge_key = f'{graph_type}_edges'
        
        if neighbor_key in batch[0]:
            # Pad neighbor lists
            neighbors = [item[neighbor_key] for item in batch]
            result[f'{graph_type}_neighbors'] = pad_sequence(
                neighbors, batch_first=True, padding_value=-1
            )
            result[f'{graph_type}_neighbor_masks'] = create_padding_mask(neighbors)
        
        if interaction_key in batch[0]:
            # Pad interaction matrices
            interactions = [item[interaction_key] for item in batch]
            result[f'{graph_type}_interactions'] = pad_sequence(
                interactions, batch_first=True, padding_value=-1
            )
        
        if edge_key in batch[0]:
            # Pad edge lists
            edges = [item[edge_key] for item in batch]
            result[f'{graph_type}_edges'] = pad_sequence(
                edges, batch_first=True, padding_value=-1
            )
            result[f'{graph_type}_edge_masks'] = create_padding_mask(edges)
    
    return result


def collate_paired_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate batch of paired text and graph data.
    
    Args:
        batch: List of paired samples
        
    Returns:
        Batched paired data
    """
    # Collate text data
    text_batch = collate_text_batch(batch)
    
    # Collate graph data
    graph_batch = collate_graph_batch(batch)
    
    # Combine results
    result = {**text_batch, **graph_batch}
    
    return result


def create_padding_mask(sequences: List[torch.Tensor], padding_value: int = -1) -> torch.Tensor:
    """
    Create padding mask for sequences.
    
    Args:
        sequences: List of sequences
        padding_value: Value used for padding
        
    Returns:
        Padding mask tensor
    """
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    
    mask = torch.ones(batch_size, max_len, dtype=torch.bool)
    
    for i, seq in enumerate(sequences):
        mask[i, :len(seq)] = False
    
    return mask


def sample_neighborhood(
    neighbors: torch.Tensor,
    max_neighbors: int = 128,
    min_neighbors: int = 32,
    strategy: str = "random"
) -> torch.Tensor:
    """
    Sample neighborhood from full neighbor list.
    
    Args:
        neighbors: Full neighbor list
        max_neighbors: Maximum number of neighbors to keep
        min_neighbors: Minimum number of neighbors to keep
        strategy: Sampling strategy ("random", "first", "last")
        
    Returns:
        Sampled neighbors
    """
    if len(neighbors) <= max_neighbors:
        return neighbors
    
    if strategy == "random":
        # Random sampling
        indices = torch.randperm(len(neighbors))[:max_neighbors]
        return neighbors[indices]
    elif strategy == "first":
        # Keep first neighbors
        return neighbors[:max_neighbors]
    elif strategy == "last":
        # Keep last neighbors
        return neighbors[-max_neighbors:]
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")


def radial_band_sampling(
    neighbors: torch.Tensor,
    distances: torch.Tensor,
    num_bands: int = 5,
    band_size: int = 20
) -> torch.Tensor:
    """
    Sample neighbors using radial band strategy.
    
    Args:
        neighbors: Neighbor indices
        distances: Distances to neighbors
        num_bands: Number of radial bands
        band_size: Size of each band
        
    Returns:
        Sampled neighbors
    """
    if len(neighbors) <= num_bands * band_size:
        return neighbors
    
    # Sort by distance
    sorted_indices = distances.argsort()
    sorted_neighbors = neighbors[sorted_indices]
    
    # Sample from each band
    sampled_neighbors = []
    for i in range(num_bands):
        start_idx = i * band_size
        end_idx = min((i + 1) * band_size, len(sorted_neighbors))
        
        if start_idx < len(sorted_neighbors):
            band_neighbors = sorted_neighbors[start_idx:end_idx]
            sampled_neighbors.append(band_neighbors)
    
    if sampled_neighbors:
        return torch.cat(sampled_neighbors)
    else:
        return neighbors[:band_size]


def create_graph_batch(
    cell_ids: List[str],
    graph_data: Dict[str, torch.Tensor],
    max_neighbors: int = 128
) -> Dict[str, torch.Tensor]:
    """
    Create graph batch from cell IDs and graph data.
    
    Args:
        cell_ids: List of cell IDs
        graph_data: Graph data dictionary
        max_neighbors: Maximum neighbors per cell
        
    Returns:
        Batched graph data
    """
    batch_size = len(cell_ids)
    
    result = {
        'cell_ids': cell_ids,
        'batch_size': batch_size
    }
    
    # Process each graph type
    for graph_type in ['knn', 'lr', 'grn']:
        if f'{graph_type}_neighbors' in graph_data:
            neighbors = graph_data[f'{graph_type}_neighbors']
            
            # Sample neighborhoods
            sampled_neighbors = []
            for i in range(batch_size):
                cell_neighbors = neighbors[i]
                # Remove padding values
                valid_neighbors = cell_neighbors[cell_neighbors != -1]
                
                if len(valid_neighbors) > max_neighbors:
                    sampled = sample_neighborhood(valid_neighbors, max_neighbors)
                else:
                    sampled = valid_neighbors
                
                sampled_neighbors.append(sampled)
            
            # Pad to same length
            result[f'{graph_type}_neighbors'] = pad_sequence(
                sampled_neighbors, batch_first=True, padding_value=-1
            )
            result[f'{graph_type}_neighbor_masks'] = create_padding_mask(sampled_neighbors)
    
    return result


def collate_with_neighborhood_sampling(
    batch: List[Dict[str, torch.Tensor]],
    max_neighbors: int = 128,
    sampling_strategy: str = "random"
) -> Dict[str, torch.Tensor]:
    """
    Collate batch with neighborhood sampling.
    
    Args:
        batch: List of samples
        max_neighbors: Maximum neighbors per cell
        sampling_strategy: Sampling strategy
        
    Returns:
        Batched data with sampled neighborhoods
    """
    # Standard collation
    result = collate_paired_batch(batch)
    
    # Apply neighborhood sampling
    for graph_type in ['knn', 'lr', 'grn']:
        neighbor_key = f'{graph_type}_neighbors'
        if neighbor_key in result:
            neighbors = result[neighbor_key]
            batch_size = neighbors.shape[0]
            
            # Sample neighborhoods for each cell
            sampled_neighbors = []
            for i in range(batch_size):
                cell_neighbors = neighbors[i]
                # Remove padding values
                valid_neighbors = cell_neighbors[cell_neighbors != -1]
                
                if len(valid_neighbors) > max_neighbors:
                    sampled = sample_neighborhood(
                        valid_neighbors, max_neighbors, strategy=sampling_strategy
                    )
                else:
                    sampled = valid_neighbors
                
                sampled_neighbors.append(sampled)
            
            # Update with sampled neighbors
            result[neighbor_key] = pad_sequence(
                sampled_neighbors, batch_first=True, padding_value=-1
            )
            result[f'{graph_type}_neighbor_masks'] = create_padding_mask(sampled_neighbors)
    
    return result
