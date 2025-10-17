"""
Complete UHG-HGNN Encoder for C2S-Scale-Gemma Hybrid Model.

This module implements the complete UHG-HGNN encoder that integrates:
- UHG layers (GraphSAGE, GIN, Attention)
- Radial projector (UHG → Euclidean)
- Multi-scale processing
- Contrastive learning capabilities
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from uhg.projective import ProjectiveUHG

from .layers import UHGGraphSAGELayer, UHGGINLayer, UHGAttentionLayer, create_uhg_layer
from .encoder import UHGEncoder, UHGMultiScaleEncoder, create_uhg_encoder
from .losses import UHGContrastiveLoss, UHGInfoNCELoss, UHGTripletLoss, create_uhg_loss
from ..uhg_adapters.radial_projector import RadialProjector

logger = logging.getLogger(__name__)


class UHGHGNNEncoder(nn.Module):
    """
    Complete UHG-HGNN encoder for the C2S-Scale-Gemma hybrid model.
    
    This encoder combines:
    - UHG graph neural network layers
    - Radial projection to Euclidean space
    - Multi-scale processing capabilities
    - Contrastive learning support
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        layer_type: str = "graphsage",
        encoder_type: str = "single_scale",
        dropout: float = 0.1,
        use_uhg_norm: bool = True,
        residual_connections: bool = True,
        pooling_method: str = "projective_average",
        # Radial projection parameters
        projection_type: str = "monotone_radial",
        preserve_angular: bool = True,
        # Multi-scale parameters
        num_scales: int = 3,
        scale_factors: List[float] = [1.0, 0.5, 0.25],
        # Contrastive learning parameters
        contrastive_temperature: float = 0.07,
        contrastive_margin: float = 1.0,
        hard_negative_mining: bool = True
    ):
        """
        Initialize UHG-HGNN encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of GNN layers
            layer_type: Type of GNN layer ("graphsage", "gin", "attention")
            encoder_type: Type of encoder ("single_scale", "multi_scale")
            dropout: Dropout probability
            use_uhg_norm: Whether to use UHG layer normalization
            residual_connections: Whether to use residual connections
            pooling_method: Method for neighborhood pooling
            projection_type: Type of radial projection
            preserve_angular: Whether to preserve angular information
            num_scales: Number of scales for multi-scale encoder
            scale_factors: Scale factors for each scale
            contrastive_temperature: Temperature for contrastive loss
            contrastive_margin: Margin for contrastive loss
            hard_negative_mining: Whether to use hard negative mining
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.layer_type = layer_type
        self.encoder_type = encoder_type
        self.dropout = dropout
        self.use_uhg_norm = use_uhg_norm
        self.residual_connections = residual_connections
        self.pooling_method = pooling_method
        self.projection_type = projection_type
        self.preserve_angular = preserve_angular
        self.num_scales = num_scales
        self.scale_factors = scale_factors
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_margin = contrastive_margin
        self.hard_negative_mining = hard_negative_mining
        
        # UHG operations
        self.uhg = ProjectiveUHG()
        
        # Create UHG encoder
        encoder_kwargs = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'output_dim': hidden_dim,  # Will be projected later
            'num_layers': num_layers,
            'layer_type': layer_type,
            'dropout': dropout,
            'use_uhg_norm': use_uhg_norm,
            'residual_connections': residual_connections,
            'pooling_method': pooling_method
        }
        
        if encoder_type == "multi_scale":
            encoder_kwargs.update({
                'num_scales': num_scales,
                'scale_factors': scale_factors
            })
        
        self.uhg_encoder = create_uhg_encoder(encoder_type, **encoder_kwargs)
        
        # Radial projector: UHG → Euclidean
        self.radial_projector = RadialProjector(
            input_dim=hidden_dim,
            output_dim=output_dim,
            projection_type=projection_type,
            preserve_angular=preserve_angular
        )
        
        # Contrastive loss for self-supervised learning
        self.contrastive_loss = UHGContrastiveLoss(
            temperature=contrastive_temperature,
            margin=contrastive_margin,
            hard_negative_mining=hard_negative_mining
        )
        
        # InfoNCE loss for alignment
        self.infonce_loss = UHGInfoNCELoss(temperature=contrastive_temperature)
        
        logger.info(f"Initialized UHG-HGNN encoder: {input_dim} -> {hidden_dim} -> {output_dim}")
        logger.info(f"Encoder type: {encoder_type}, Layer type: {layer_type}")
        logger.info(f"Projection type: {projection_type}, Preserve angular: {preserve_angular}")
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_projections: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through UHG-HGNN encoder.
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge indices [2, E]
            edge_weight: Edge weights [E] (optional)
            batch: Batch assignment [N] (optional)
            return_projections: Whether to return intermediate projections
            
        Returns:
            Dictionary containing:
                - hyperbolic_embeddings: Embeddings in UHG space [N, hidden_dim]
                - euclidean_embeddings: Projected embeddings in Euclidean space [N, output_dim]
                - graph_embeddings: Graph-level embeddings [B, output_dim] (if batch provided)
                - layer_outputs: Outputs from each layer (if return_projections)
        """
        # Encode using UHG encoder
        encoder_outputs = self.uhg_encoder(x, edge_index, edge_weight, batch)
        
        # Get hyperbolic embeddings
        hyperbolic_embeddings = encoder_outputs['node_embeddings']
        
        # Project to Euclidean space
        euclidean_embeddings = self.radial_projector(hyperbolic_embeddings)
        
        # Project graph embeddings if available
        graph_embeddings = None
        if encoder_outputs['graph_embeddings'] is not None:
            graph_embeddings = self.radial_projector(encoder_outputs['graph_embeddings'])
        
        # Prepare outputs
        outputs = {
            'hyperbolic_embeddings': hyperbolic_embeddings,
            'euclidean_embeddings': euclidean_embeddings,
            'graph_embeddings': graph_embeddings
        }
        
        # Add layer outputs if requested
        if return_projections and 'layer_outputs' in encoder_outputs:
            # Project layer outputs to Euclidean space
            projected_layer_outputs = []
            for layer_output in encoder_outputs['layer_outputs']:
                projected_layer_output = self.radial_projector(layer_output)
                projected_layer_outputs.append(projected_layer_output)
            
            outputs['layer_outputs'] = torch.stack(projected_layer_outputs)
            outputs['hyperbolic_layer_outputs'] = encoder_outputs['layer_outputs']
        
        # Add scale embeddings for multi-scale encoder
        if self.encoder_type == "multi_scale" and 'scale_embeddings' in encoder_outputs:
            projected_scale_embeddings = []
            for scale_embedding in encoder_outputs['scale_embeddings']:
                projected_scale_embedding = self.radial_projector(scale_embedding)
                projected_scale_embeddings.append(projected_scale_embedding)
            
            outputs['scale_embeddings'] = projected_scale_embeddings
            outputs['hyperbolic_scale_embeddings'] = encoder_outputs['scale_embeddings']
        
        return outputs
    
    def encode_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode nodes to Euclidean embeddings.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_weight: Edge weights
            
        Returns:
            Euclidean node embeddings
        """
        outputs = self.forward(x, edge_index, edge_weight)
        return outputs['euclidean_embeddings']
    
    def encode_graphs(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode graphs to Euclidean embeddings.
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment
            edge_weight: Edge weights
            
        Returns:
            Euclidean graph embeddings
        """
        outputs = self.forward(x, edge_index, edge_weight, batch)
        return outputs['graph_embeddings']
    
    def compute_contrastive_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        positive_pairs: Optional[torch.Tensor] = None,
        negative_pairs: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive loss for self-supervised learning.
        
        Args:
            embeddings: Node embeddings in UHG space
            labels: Node labels
            positive_pairs: Positive pairs (optional)
            negative_pairs: Negative pairs (optional)
            
        Returns:
            Dictionary containing loss components
        """
        return self.contrastive_loss(embeddings, labels, positive_pairs, negative_pairs)
    
    def compute_infonce_loss(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        positive_indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute InfoNCE loss for alignment.
        
        Args:
            query_embeddings: Query embeddings in UHG space
            key_embeddings: Key embeddings in UHG space
            positive_indices: Positive key indices for each query
            
        Returns:
            Dictionary containing loss components
        """
        return self.infonce_loss(query_embeddings, key_embeddings, positive_indices)
    
    def get_monotonicity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get monotonicity loss from radial projector.
        
        Args:
            x: Input features
            
        Returns:
            Monotonicity loss
        """
        # Get hyperbolic embeddings
        encoder_outputs = self.uhg_encoder(x, torch.empty(2, 0, dtype=torch.long, device=x.device))
        hyperbolic_embeddings = encoder_outputs['node_embeddings']
        
        return self.radial_projector.get_monotonicity_loss(hyperbolic_embeddings)
    
    def get_angular_preservation_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get angular preservation loss from radial projector.
        
        Args:
            x: Input features
            
        Returns:
            Angular preservation loss
        """
        # Get hyperbolic embeddings
        encoder_outputs = self.uhg_encoder(x, torch.empty(2, 0, dtype=torch.long, device=x.device))
        hyperbolic_embeddings = encoder_outputs['node_embeddings']
        
        return self.radial_projector.get_angular_preservation_loss(hyperbolic_embeddings)
    
    def get_model_info(self) -> Dict[str, Union[int, str, bool, List[float]]]:
        """
        Get model information.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'layer_type': self.layer_type,
            'encoder_type': self.encoder_type,
            'dropout': self.dropout,
            'use_uhg_norm': self.use_uhg_norm,
            'residual_connections': self.residual_connections,
            'pooling_method': self.pooling_method,
            'projection_type': self.projection_type,
            'preserve_angular': self.preserve_angular,
            'num_scales': self.num_scales,
            'scale_factors': self.scale_factors,
            'contrastive_temperature': self.contrastive_temperature,
            'contrastive_margin': self.contrastive_margin,
            'hard_negative_mining': self.hard_negative_mining
        }


def create_uhg_hgnn_encoder(
    input_dim: int = 2000,
    hidden_dim: int = 256,
    output_dim: int = 128,
    num_layers: int = 3,
    layer_type: str = "graphsage",
    encoder_type: str = "single_scale",
    **kwargs
) -> UHGHGNNEncoder:
    """
    Create a UHG-HGNN encoder instance.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of GNN layers
        layer_type: Type of GNN layer
        encoder_type: Type of encoder
        **kwargs: Additional arguments
        
    Returns:
        UHG-HGNN encoder instance
    """
    return UHGHGNNEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        layer_type=layer_type,
        encoder_type=encoder_type,
        **kwargs
    )


# Convenience functions for different configurations
def create_colab_hgnn_encoder(**kwargs) -> UHGHGNNEncoder:
    """Create UHG-HGNN encoder optimized for Colab (7B model)."""
    default_kwargs = {
        'input_dim': 2000,
        'hidden_dim': 256,
        'output_dim': 128,
        'num_layers': 3,
        'layer_type': 'graphsage',
        'encoder_type': 'single_scale',
        'dropout': 0.1,
        'use_uhg_norm': True,
        'residual_connections': True,
        'pooling_method': 'projective_average',
        'projection_type': 'monotone_radial',
        'preserve_angular': True,
        'contrastive_temperature': 0.07,
        'contrastive_margin': 1.0,
        'hard_negative_mining': True
    }
    default_kwargs.update(kwargs)
    return create_uhg_hgnn_encoder(**default_kwargs)


def create_vertex_hgnn_encoder(**kwargs) -> UHGHGNNEncoder:
    """Create UHG-HGNN encoder optimized for Vertex AI (27B model)."""
    default_kwargs = {
        'input_dim': 2000,
        'hidden_dim': 384,
        'output_dim': 256,
        'num_layers': 4,
        'layer_type': 'graphsage',
        'encoder_type': 'multi_scale',
        'num_scales': 3,
        'scale_factors': [1.0, 0.5, 0.25],
        'dropout': 0.1,
        'use_uhg_norm': True,
        'residual_connections': True,
        'pooling_method': 'projective_average',
        'projection_type': 'monotone_radial',
        'preserve_angular': True,
        'contrastive_temperature': 0.07,
        'contrastive_margin': 1.0,
        'hard_negative_mining': True
    }
    default_kwargs.update(kwargs)
    return create_uhg_hgnn_encoder(**default_kwargs)
