"""
UHG Graph Neural Network layers.

Implements UHG-GraphSAGE and UHG-GIN layers using UHG primitives.
"""

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from uhg.layers import UHGConv, UHGLayerNorm
from uhg.nn import ProjectiveSAGEConv
from uhg.projective import ProjectiveUHG

logger = logging.getLogger(__name__)


class UHGGraphSAGELayer(nn.Module):
    """
    UHG GraphSAGE layer using UHG primitives.
    
    Implements GraphSAGE aggregation in hyperbolic space.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        aggregator: str = "mean",
        dropout: float = 0.1,
        use_uhg_norm: bool = True
    ):
        """
        Initialize UHG GraphSAGE layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            aggregator: Aggregation method ("mean", "max", "sum")
            dropout: Dropout probability
            use_uhg_norm: Whether to use UHG layer normalization
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator = aggregator
        self.dropout = dropout
        self.use_uhg_norm = use_uhg_norm
        
        # UHG operations
        self.uhg = ProjectiveUHG()
        
        # Linear transformation for self features
        self.self_linear = nn.Linear(in_features, out_features)
        
        # Linear transformation for neighbor features
        self.neighbor_linear = nn.Linear(in_features, out_features)
        
        # UHG layer normalization
        if use_uhg_norm:
            self.uhg_norm = UHGLayerNorm(out_features)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        logger.info(f"Initialized UHG GraphSAGE layer: {in_features} -> {out_features}")
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through UHG GraphSAGE layer.
        
        Args:
            x: Node features [N, in_features]
            edge_index: Edge indices [2, E]
            edge_weight: Edge weights [E] (optional)
            
        Returns:
            Updated node features [N, out_features]
        """
        # Transform self features
        self_features = self.self_linear(x)
        
        # Aggregate neighbor features
        neighbor_features = self._aggregate_neighbors(x, edge_index, edge_weight)
        
        # Transform neighbor features
        neighbor_features = self.neighbor_linear(neighbor_features)
        
        # Combine self and neighbor features
        # In UHG, we use projective average for combination
        combined_features = self._combine_features(self_features, neighbor_features)
        
        # Apply UHG normalization
        if self.use_uhg_norm:
            combined_features = self.uhg_norm(combined_features)
        
        # Apply dropout
        combined_features = self.dropout_layer(combined_features)
        
        return combined_features
    
    def _aggregate_neighbors(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Aggregate neighbor features using UHG operations.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_weight: Edge weights
            
        Returns:
            Aggregated neighbor features
        """
        num_nodes = x.size(0)
        aggregated = torch.zeros_like(x)
        
        # Get source and target indices
        source_indices = edge_index[0]
        target_indices = edge_index[1]
        
        # Aggregate neighbors for each node
        for node_idx in range(num_nodes):
            # Find neighbors of this node
            neighbor_mask = target_indices == node_idx
            if neighbor_mask.sum() == 0:
                # No neighbors, use zero features
                aggregated[node_idx] = torch.zeros_like(x[node_idx])
                continue
            
            neighbor_indices = source_indices[neighbor_mask]
            neighbor_features = x[neighbor_indices]
            
            if edge_weight is not None:
                neighbor_weights = edge_weight[neighbor_mask]
                # Normalize weights
                neighbor_weights = F.softmax(neighbor_weights, dim=0)
            else:
                neighbor_weights = torch.ones(len(neighbor_indices), device=x.device)
                neighbor_weights = neighbor_weights / len(neighbor_indices)
            
            # Aggregate using UHG projective average
            aggregated[node_idx] = self.uhg.projective_average(
                neighbor_features, neighbor_weights
            )
        
        return aggregated
    
    def _combine_features(
        self,
        self_features: torch.Tensor,
        neighbor_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine self and neighbor features.
        
        Args:
            self_features: Self features
            neighbor_features: Neighbor features
            
        Returns:
            Combined features
        """
        # Use UHG projective average to combine features
        # This preserves hyperbolic geometry properties
        combined_features = self.uhg.projective_average(
            torch.stack([self_features, neighbor_features], dim=1),
            torch.tensor([0.5, 0.5], device=self_features.device)
        )
        
        return combined_features


class UHGGINLayer(nn.Module):
    """
    UHG Graph Isomorphism Network layer.
    
    Implements GIN aggregation in hyperbolic space.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        eps: float = 0.0,
        dropout: float = 0.1,
        use_uhg_norm: bool = True
    ):
        """
        Initialize UHG GIN layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            eps: Epsilon parameter for GIN
            dropout: Dropout probability
            use_uhg_norm: Whether to use UHG layer normalization
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.dropout = dropout
        self.use_uhg_norm = use_uhg_norm
        
        # UHG operations
        self.uhg = ProjectiveUHG()
        
        # MLP for feature transformation
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )
        
        # UHG layer normalization
        if use_uhg_norm:
            self.uhg_norm = UHGLayerNorm(out_features)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        logger.info(f"Initialized UHG GIN layer: {in_features} -> {out_features}")
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through UHG GIN layer.
        
        Args:
            x: Node features [N, in_features]
            edge_index: Edge indices [2, E]
            edge_weight: Edge weights [E] (optional)
            
        Returns:
            Updated node features [N, out_features]
        """
        # Aggregate neighbor features
        neighbor_features = self._aggregate_neighbors(x, edge_index, edge_weight)
        
        # GIN combination: (1 + eps) * x + neighbor_features
        combined_features = (1 + self.eps) * x + neighbor_features
        
        # Apply MLP
        combined_features = self.mlp(combined_features)
        
        # Apply UHG normalization
        if self.use_uhg_norm:
            combined_features = self.uhg_norm(combined_features)
        
        # Apply dropout
        combined_features = self.dropout_layer(combined_features)
        
        return combined_features
    
    def _aggregate_neighbors(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Aggregate neighbor features using UHG operations.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_weight: Edge weights
            
        Returns:
            Aggregated neighbor features
        """
        num_nodes = x.size(0)
        aggregated = torch.zeros_like(x)
        
        # Get source and target indices
        source_indices = edge_index[0]
        target_indices = edge_index[1]
        
        # Aggregate neighbors for each node
        for node_idx in range(num_nodes):
            # Find neighbors of this node
            neighbor_mask = target_indices == node_idx
            if neighbor_mask.sum() == 0:
                # No neighbors, use zero features
                aggregated[node_idx] = torch.zeros_like(x[node_idx])
                continue
            
            neighbor_indices = source_indices[neighbor_mask]
            neighbor_features = x[neighbor_indices]
            
            if edge_weight is not None:
                neighbor_weights = edge_weight[neighbor_mask]
                # Normalize weights
                neighbor_weights = F.softmax(neighbor_weights, dim=0)
            else:
                neighbor_weights = torch.ones(len(neighbor_indices), device=x.device)
                neighbor_weights = neighbor_weights / len(neighbor_indices)
            
            # Aggregate using UHG projective average
            aggregated[node_idx] = self.uhg.projective_average(
                neighbor_features, neighbor_weights
            )
        
        return aggregated


class UHGAttentionLayer(nn.Module):
    """
    UHG attention layer for graph neural networks.
    
    Implements attention mechanism in hyperbolic space.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize UHG attention layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"
        
        # UHG operations
        self.uhg = ProjectiveUHG()
        
        # Linear transformations
        self.query_linear = nn.Linear(in_features, out_features)
        self.key_linear = nn.Linear(in_features, out_features)
        self.value_linear = nn.Linear(in_features, out_features)
        self.output_linear = nn.Linear(out_features, out_features)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        logger.info(f"Initialized UHG attention layer: {in_features} -> {out_features}, {num_heads} heads")
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through UHG attention layer.
        
        Args:
            x: Node features [N, in_features]
            edge_index: Edge indices [2, E]
            edge_weight: Edge weights [E] (optional)
            
        Returns:
            Updated node features [N, out_features]
        """
        # Compute queries, keys, and values
        queries = self.query_linear(x)
        keys = self.key_linear(x)
        values = self.value_linear(x)
        
        # Reshape for multi-head attention
        batch_size = x.size(0)
        queries = queries.view(batch_size, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, self.num_heads, self.head_dim)
        values = values.view(batch_size, self.num_heads, self.head_dim)
        
        # Compute attention scores using UHG distance
        attention_scores = self._compute_attention_scores(queries, keys, edge_index)
        
        # Apply attention weights
        attended_values = self._apply_attention(attention_scores, values, edge_index)
        
        # Concatenate heads
        attended_values = attended_values.view(batch_size, self.out_features)
        
        # Output projection
        output = self.output_linear(attended_values)
        
        # Apply dropout
        output = self.dropout_layer(output)
        
        return output
    
    def _compute_attention_scores(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention scores using UHG distance.
        
        Args:
            queries: Query features [N, num_heads, head_dim]
            keys: Key features [N, num_heads, head_dim]
            edge_index: Edge indices [2, E]
            
        Returns:
            Attention scores [E, num_heads]
        """
        source_indices = edge_index[0]
        target_indices = edge_index[1]
        
        # Get query and key features for edges
        query_features = queries[source_indices]  # [E, num_heads, head_dim]
        key_features = keys[target_indices]  # [E, num_heads, head_dim]
        
        # Compute UHG distances between query and key features
        attention_scores = []
        for head_idx in range(self.num_heads):
            head_queries = query_features[:, head_idx, :]  # [E, head_dim]
            head_keys = key_features[:, head_idx, :]  # [E, head_dim]
            
            # Compute UHG distances
            distances = self.uhg.distance(head_queries, head_keys)
            
            # Convert distances to attention scores (negative distances for higher attention)
            scores = -distances
            attention_scores.append(scores)
        
        attention_scores = torch.stack(attention_scores, dim=1)  # [E, num_heads]
        
        return attention_scores
    
    def _apply_attention(
        self,
        attention_scores: torch.Tensor,
        values: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply attention weights to values.
        
        Args:
            attention_scores: Attention scores [E, num_heads]
            values: Value features [N, num_heads, head_dim]
            edge_index: Edge indices [2, E]
            
        Returns:
            Attended values [N, num_heads, head_dim]
        """
        num_nodes = values.size(0)
        attended_values = torch.zeros_like(values)
        
        source_indices = edge_index[0]
        target_indices = edge_index[1]
        
        # Apply attention for each node
        for node_idx in range(num_nodes):
            # Find edges where this node is the target
            edge_mask = target_indices == node_idx
            if edge_mask.sum() == 0:
                # No incoming edges, use zero values
                attended_values[node_idx] = torch.zeros_like(values[node_idx])
                continue
            
            # Get attention scores and source indices for this node
            node_attention_scores = attention_scores[edge_mask]  # [num_edges, num_heads]
            source_nodes = source_indices[edge_mask]  # [num_edges]
            
            # Get value features for source nodes
            source_values = values[source_nodes]  # [num_edges, num_heads, head_dim]
            
            # Apply softmax to attention scores
            attention_weights = F.softmax(node_attention_scores, dim=0)  # [num_edges, num_heads]
            
            # Weighted sum of values
            for head_idx in range(self.num_heads):
                head_weights = attention_weights[:, head_idx].unsqueeze(-1)  # [num_edges, 1]
                head_values = source_values[:, head_idx, :]  # [num_edges, head_dim]
                
                # Use UHG projective average for weighted combination
                attended_values[node_idx, head_idx, :] = self.uhg.projective_average(
                    head_values, head_weights.squeeze(-1)
                )
        
        return attended_values


def create_uhg_layer(
    layer_type: str,
    in_features: int,
    out_features: int,
    **kwargs
) -> Union[UHGGraphSAGELayer, UHGGINLayer, UHGAttentionLayer]:
    """
    Create a UHG layer instance.
    
    Args:
        layer_type: Type of layer ("graphsage", "gin", "attention")
        in_features: Input feature dimension
        out_features: Output feature dimension
        **kwargs: Additional arguments for layer
        
    Returns:
        UHG layer instance
    """
    if layer_type == "graphsage":
        return UHGGraphSAGELayer(in_features, out_features, **kwargs)
    elif layer_type == "gin":
        return UHGGINLayer(in_features, out_features, **kwargs)
    elif layer_type == "attention":
        return UHGAttentionLayer(in_features, out_features, **kwargs)
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")
