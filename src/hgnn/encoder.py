"""
UHG Graph Neural Network encoder.

Implements multi-layer HGNN encoder with UHG operations.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from uhg.projective import ProjectiveUHG
from uhg.layers import UHGLayerNorm

from .layers import UHGGraphSAGELayer, UHGGINLayer, UHGAttentionLayer, create_uhg_layer

logger = logging.getLogger(__name__)


class UHGEncoder(nn.Module):
    """
    Multi-layer UHG Graph Neural Network encoder.
    
    Encodes graph data into hyperbolic embeddings using UHG operations.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        layer_type: str = "graphsage",
        dropout: float = 0.1,
        use_uhg_norm: bool = True,
        residual_connections: bool = True,
        pooling_method: str = "projective_average"
    ):
        """
        Initialize UHG encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of layers
            layer_type: Type of GNN layer ("graphsage", "gin", "attention")
            dropout: Dropout probability
            use_uhg_norm: Whether to use UHG layer normalization
            residual_connections: Whether to use residual connections
            pooling_method: Method for neighborhood pooling ("projective_average", "mean", "max")
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.layer_type = layer_type
        self.dropout = dropout
        self.use_uhg_norm = use_uhg_norm
        self.residual_connections = residual_connections
        self.pooling_method = pooling_method
        
        # UHG operations
        self.uhg = ProjectiveUHG()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = create_uhg_layer(
                layer_type=layer_type,
                in_features=hidden_dim,
                out_features=hidden_dim,
                dropout=dropout,
                use_uhg_norm=use_uhg_norm
            )
            self.layers.append(layer)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # Final UHG normalization
        if use_uhg_norm:
            self.final_norm = UHGLayerNorm(output_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        logger.info(f"Initialized UHG encoder: {input_dim} -> {hidden_dim} -> {output_dim}, {num_layers} layers")
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through UHG encoder.
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge indices [2, E]
            edge_weight: Edge weights [E] (optional)
            batch: Batch assignment [N] (optional)
            
        Returns:
            Dictionary containing:
                - node_embeddings: Node-level embeddings [N, output_dim]
                - graph_embeddings: Graph-level embeddings [B, output_dim] (if batch provided)
                - layer_outputs: Outputs from each layer [num_layers, N, hidden_dim]
        """
        # Input projection
        h = self.input_projection(x)
        
        # Store layer outputs
        layer_outputs = []
        
        # Forward pass through layers
        for i, layer in enumerate(self.layers):
            # Apply layer
            h_new = layer(h, edge_index, edge_weight)
            
            # Residual connection
            if self.residual_connections and h.size(-1) == h_new.size(-1):
                h = h + h_new
            else:
                h = h_new
            
            # Apply dropout
            h = self.dropout_layer(h)
            
            layer_outputs.append(h)
        
        # Output projection
        node_embeddings = self.output_projection(h)
        
        # Final normalization
        if self.use_uhg_norm:
            node_embeddings = self.final_norm(node_embeddings)
        
        # Graph-level pooling
        graph_embeddings = None
        if batch is not None:
            graph_embeddings = self._pool_neighborhoods(node_embeddings, batch)
        
        return {
            'node_embeddings': node_embeddings,
            'graph_embeddings': graph_embeddings,
            'layer_outputs': torch.stack(layer_outputs)
        }
    
    def _pool_neighborhoods(
        self,
        node_embeddings: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool node embeddings to graph-level embeddings.
        
        Args:
            node_embeddings: Node embeddings [N, output_dim]
            batch: Batch assignment [N]
            
        Returns:
            Graph embeddings [B, output_dim]
        """
        num_graphs = batch.max().item() + 1
        graph_embeddings = []
        
        for graph_idx in range(num_graphs):
            # Get nodes belonging to this graph
            node_mask = batch == graph_idx
            graph_nodes = node_embeddings[node_mask]
            
            if len(graph_nodes) == 0:
                # Empty graph, use zero embedding
                graph_embeddings.append(torch.zeros_like(node_embeddings[0]))
                continue
            
            # Pool using specified method
            if self.pooling_method == "projective_average":
                # Use UHG projective average
                weights = torch.ones(len(graph_nodes), device=node_embeddings.device)
                weights = weights / len(graph_nodes)
                graph_embedding = self.uhg.projective_average(graph_nodes, weights)
            elif self.pooling_method == "mean":
                graph_embedding = torch.mean(graph_nodes, dim=0)
            elif self.pooling_method == "max":
                graph_embedding = torch.max(graph_nodes, dim=0)[0]
            else:
                raise ValueError(f"Unknown pooling method: {self.pooling_method}")
            
            graph_embeddings.append(graph_embedding)
        
        return torch.stack(graph_embeddings)
    
    def encode_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode nodes to hyperbolic embeddings.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_weight: Edge weights
            
        Returns:
            Node embeddings
        """
        outputs = self.forward(x, edge_index, edge_weight)
        return outputs['node_embeddings']
    
    def encode_graphs(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode graphs to hyperbolic embeddings.
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment
            edge_weight: Edge weights
            
        Returns:
            Graph embeddings
        """
        outputs = self.forward(x, edge_index, edge_weight, batch)
        return outputs['graph_embeddings']
    
    def get_layer_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Get embeddings from each layer.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_weight: Edge weights
            
        Returns:
            List of embeddings from each layer
        """
        outputs = self.forward(x, edge_index, edge_weight)
        return [outputs['layer_outputs'][i] for i in range(self.num_layers)]


class UHGMultiScaleEncoder(nn.Module):
    """
    Multi-scale UHG encoder that captures information at different scales.
    
    Combines multiple UHG encoders with different receptive fields.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        layer_type: str = "graphsage",
        num_scales: int = 3,
        scale_factors: List[float] = [1.0, 0.5, 0.25],
        dropout: float = 0.1,
        use_uhg_norm: bool = True
    ):
        """
        Initialize multi-scale UHG encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of layers per scale
            layer_type: Type of GNN layer
            num_scales: Number of scales
            scale_factors: Scale factors for each scale
            dropout: Dropout probability
            use_uhg_norm: Whether to use UHG layer normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_scales = num_scales
        self.scale_factors = scale_factors
        
        # UHG operations
        self.uhg = ProjectiveUHG()
        
        # Multi-scale encoders
        self.encoders = nn.ModuleList()
        for i in range(num_scales):
            encoder = UHGEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,  # Will be projected later
                num_layers=num_layers,
                layer_type=layer_type,
                dropout=dropout,
                use_uhg_norm=use_uhg_norm
            )
            self.encoders.append(encoder)
        
        # Scale fusion
        self.scale_fusion = nn.Linear(hidden_dim * num_scales, output_dim)
        
        # Final normalization
        if use_uhg_norm:
            self.final_norm = UHGLayerNorm(output_dim)
        
        logger.info(f"Initialized multi-scale UHG encoder: {num_scales} scales")
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-scale encoder.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_weight: Edge weights
            batch: Batch assignment
            
        Returns:
            Dictionary containing multi-scale embeddings
        """
        scale_embeddings = []
        
        # Encode at each scale
        for i, encoder in enumerate(self.encoders):
            # Apply scale factor to edge weights
            scaled_edge_weight = edge_weight
            if edge_weight is not None:
                scaled_edge_weight = edge_weight * self.scale_factors[i]
            
            # Encode at this scale
            scale_output = encoder.encode_nodes(x, edge_index, scaled_edge_weight)
            scale_embeddings.append(scale_output)
        
        # Concatenate scale embeddings
        multi_scale_embeddings = torch.cat(scale_embeddings, dim=-1)
        
        # Fuse scales
        fused_embeddings = self.scale_fusion(multi_scale_embeddings)
        
        # Final normalization
        if self.use_uhg_norm:
            fused_embeddings = self.final_norm(fused_embeddings)
        
        # Graph-level pooling
        graph_embeddings = None
        if batch is not None:
            graph_embeddings = self._pool_neighborhoods(fused_embeddings, batch)
        
        return {
            'node_embeddings': fused_embeddings,
            'graph_embeddings': graph_embeddings,
            'scale_embeddings': scale_embeddings
        }
    
    def _pool_neighborhoods(
        self,
        node_embeddings: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool node embeddings to graph-level embeddings.
        
        Args:
            node_embeddings: Node embeddings
            batch: Batch assignment
            
        Returns:
            Graph embeddings
        """
        num_graphs = batch.max().item() + 1
        graph_embeddings = []
        
        for graph_idx in range(num_graphs):
            node_mask = batch == graph_idx
            graph_nodes = node_embeddings[node_mask]
            
            if len(graph_nodes) == 0:
                graph_embeddings.append(torch.zeros_like(node_embeddings[0]))
                continue
            
            # Use UHG projective average
            weights = torch.ones(len(graph_nodes), device=node_embeddings.device)
            weights = weights / len(graph_nodes)
            graph_embedding = self.uhg.projective_average(graph_nodes, weights)
            graph_embeddings.append(graph_embedding)
        
        return torch.stack(graph_embeddings)


def create_uhg_encoder(
    encoder_type: str = "single_scale",
    input_dim: int = 64,
    hidden_dim: int = 256,
    output_dim: int = 128,
    num_layers: int = 3,
    layer_type: str = "graphsage",
    **kwargs
) -> Union[UHGEncoder, UHGMultiScaleEncoder]:
    """
    Create a UHG encoder instance.
    
    Args:
        encoder_type: Type of encoder ("single_scale", "multi_scale")
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of layers
        layer_type: Type of GNN layer
        **kwargs: Additional arguments
        
    Returns:
        UHG encoder instance
    """
    if encoder_type == "single_scale":
        return UHGEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            layer_type=layer_type,
            **kwargs
        )
    elif encoder_type == "multi_scale":
        return UHGMultiScaleEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            layer_type=layer_type,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
