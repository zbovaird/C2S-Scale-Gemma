"""
Fusion heads for combining text and graph embeddings.

Implements various fusion strategies for the hybrid model.
"""

import logging
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MLPFusionHead(nn.Module):
    """
    MLP fusion head for combining text and graph embeddings.
    
    Implements a simple MLP-based fusion strategy.
    """
    
    def __init__(
        self,
        text_dim: int,
        graph_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        """
        Initialize MLP fusion head.
        
        Args:
            text_dim: Text embedding dimension
            graph_dim: Graph embedding dimension
            output_dim: Output dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of layers
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()
        
        self.text_dim = text_dim
        self.graph_dim = graph_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or max(text_dim, graph_dim)
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        
        # Input projection
        self.input_projection = nn.Linear(text_dim + graph_dim, self.hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        
        # Output projection
        self.output_projection = nn.Linear(self.hidden_dim, output_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Activation
        if activation == "relu":
            self.activation_fn = nn.ReLU()
        elif activation == "gelu":
            self.activation_fn = nn.GELU()
        elif activation == "tanh":
            self.activation_fn = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        logger.info(f"Initialized MLP fusion head: {text_dim} + {graph_dim} -> {output_dim}")
    
    def forward(
        self,
        text_embeddings: torch.Tensor,
        graph_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through MLP fusion head.
        
        Args:
            text_embeddings: Text embeddings [batch_size, text_dim]
            graph_embeddings: Graph embeddings [batch_size, graph_dim]
            
        Returns:
            Fused embeddings [batch_size, output_dim]
        """
        # Concatenate embeddings
        fused = torch.cat([text_embeddings, graph_embeddings], dim=-1)
        
        # Input projection
        fused = self.input_projection(fused)
        fused = self.activation_fn(fused)
        fused = self.dropout_layer(fused)
        
        # Hidden layers
        for layer in self.hidden_layers:
            fused = layer(fused)
            fused = self.activation_fn(fused)
            fused = self.dropout_layer(fused)
        
        # Output projection
        fused = self.output_projection(fused)
        
        return fused


class AttentionFusionHead(nn.Module):
    """
    Attention-based fusion head for combining text and graph embeddings.
    
    Uses attention mechanism to dynamically weight and combine embeddings.
    """
    
    def __init__(
        self,
        text_dim: int,
        graph_dim: int,
        output_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_cross_attention: bool = True
    ):
        """
        Initialize attention fusion head.
        
        Args:
            text_dim: Text embedding dimension
            graph_dim: Graph embedding dimension
            output_dim: Output dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_cross_attention: Whether to use cross-attention
        """
        super().__init__()
        
        self.text_dim = text_dim
        self.graph_dim = graph_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_cross_attention = use_cross_attention
        
        # Project embeddings to common dimension
        self.common_dim = max(text_dim, graph_dim)
        self.text_projection = nn.Linear(text_dim, self.common_dim)
        self.graph_projection = nn.Linear(graph_dim, self.common_dim)
        
        # Self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.common_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention
        if use_cross_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.common_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        
        # Output projection
        self.output_projection = nn.Linear(self.common_dim, output_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.common_dim)
        
        logger.info(f"Initialized attention fusion head: {text_dim} + {graph_dim} -> {output_dim}")
    
    def forward(
        self,
        text_embeddings: torch.Tensor,
        graph_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through attention fusion head.
        
        Args:
            text_embeddings: Text embeddings [batch_size, text_dim]
            graph_embeddings: Graph embeddings [batch_size, graph_dim]
            
        Returns:
            Fused embeddings [batch_size, output_dim]
        """
        batch_size = text_embeddings.size(0)
        
        # Project embeddings to common dimension
        text_proj = self.text_projection(text_embeddings)  # [batch_size, common_dim]
        graph_proj = self.graph_projection(graph_embeddings)  # [batch_size, common_dim]
        
        # Stack embeddings for attention
        stacked_embeddings = torch.stack([text_proj, graph_proj], dim=1)  # [batch_size, 2, common_dim]
        
        # Self-attention
        attn_output, _ = self.self_attention(
            query=stacked_embeddings,
            key=stacked_embeddings,
            value=stacked_embeddings
        )
        
        # Layer normalization
        attn_output = self.layer_norm(attn_output)
        
        # Cross-attention
        if self.use_cross_attention:
            cross_attn_output, _ = self.cross_attention(
                query=attn_output,
                key=stacked_embeddings,
                value=stacked_embeddings
            )
            attn_output = attn_output + cross_attn_output
        
        # Pool over sequence dimension
        fused = torch.mean(attn_output, dim=1)  # [batch_size, common_dim]
        
        # Output projection
        fused = self.output_projection(fused)
        
        return fused


class GatedFusionHead(nn.Module):
    """
    Gated fusion head for combining text and graph embeddings.
    
    Uses gating mechanism to control the contribution of each modality.
    """
    
    def __init__(
        self,
        text_dim: int,
        graph_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Initialize gated fusion head.
        
        Args:
            text_dim: Text embedding dimension
            graph_dim: Graph embedding dimension
            output_dim: Output dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.text_dim = text_dim
        self.graph_dim = graph_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or max(text_dim, graph_dim)
        self.dropout = dropout
        
        # Gating network
        self.gate_network = nn.Sequential(
            nn.Linear(text_dim + graph_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 2),  # 2 gates for text and graph
            nn.Softmax(dim=-1)
        )
        
        # Projection networks
        self.text_projection = nn.Linear(text_dim, self.hidden_dim)
        self.graph_projection = nn.Linear(graph_dim, self.hidden_dim)
        
        # Output projection
        self.output_projection = nn.Linear(self.hidden_dim, output_dim)
        
        logger.info(f"Initialized gated fusion head: {text_dim} + {graph_dim} -> {output_dim}")
    
    def forward(
        self,
        text_embeddings: torch.Tensor,
        graph_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through gated fusion head.
        
        Args:
            text_embeddings: Text embeddings [batch_size, text_dim]
            graph_embeddings: Graph embeddings [batch_size, graph_dim]
            
        Returns:
            Fused embeddings [batch_size, output_dim]
        """
        # Concatenate embeddings for gating
        concat_embeddings = torch.cat([text_embeddings, graph_embeddings], dim=-1)
        
        # Compute gates
        gates = self.gate_network(concat_embeddings)  # [batch_size, 2]
        text_gate = gates[:, 0:1]  # [batch_size, 1]
        graph_gate = gates[:, 1:2]  # [batch_size, 1]
        
        # Project embeddings
        text_proj = self.text_projection(text_embeddings)  # [batch_size, hidden_dim]
        graph_proj = self.graph_projection(graph_embeddings)  # [batch_size, hidden_dim]
        
        # Apply gates
        gated_text = text_proj * text_gate
        gated_graph = graph_proj * graph_gate
        
        # Combine gated embeddings
        fused = gated_text + gated_graph
        
        # Output projection
        fused = self.output_projection(fused)
        
        return fused


class ResidualFusionHead(nn.Module):
    """
    Residual fusion head for combining text and graph embeddings.
    
    Uses residual connections for stable training.
    """
    
    def __init__(
        self,
        text_dim: int,
        graph_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize residual fusion head.
        
        Args:
            text_dim: Text embedding dimension
            graph_dim: Graph embedding dimension
            output_dim: Output dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.text_dim = text_dim
        self.graph_dim = graph_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or max(text_dim, graph_dim)
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_projection = nn.Linear(text_dim + graph_dim, self.hidden_dim)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(num_layers):
            block = ResidualBlock(self.hidden_dim, dropout)
            self.residual_blocks.append(block)
        
        # Output projection
        self.output_projection = nn.Linear(self.hidden_dim, output_dim)
        
        logger.info(f"Initialized residual fusion head: {text_dim} + {graph_dim} -> {output_dim}")
    
    def forward(
        self,
        text_embeddings: torch.Tensor,
        graph_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through residual fusion head.
        
        Args:
            text_embeddings: Text embeddings [batch_size, text_dim]
            graph_embeddings: Graph embeddings [batch_size, graph_dim]
            
        Returns:
            Fused embeddings [batch_size, output_dim]
        """
        # Concatenate embeddings
        fused = torch.cat([text_embeddings, graph_embeddings], dim=-1)
        
        # Input projection
        fused = self.input_projection(fused)
        
        # Residual blocks
        for block in self.residual_blocks:
            fused = block(fused)
        
        # Output projection
        fused = self.output_projection(fused)
        
        return fused


class ResidualBlock(nn.Module):
    """
    Residual block for fusion head.
    
    Implements residual connection with layer normalization.
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        """
        Initialize residual block.
        
        Args:
            hidden_dim: Hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Layers
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.
        
        Args:
            x: Input tensor [batch_size, hidden_dim]
            
        Returns:
            Output tensor [batch_size, hidden_dim]
        """
        # Residual connection
        residual = x
        
        # First linear layer
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout_layer(x)
        
        # Second linear layer
        x = self.linear2(x)
        
        # Add residual connection
        x = x + residual
        
        # Layer normalization
        x = self.layer_norm(x)
        
        return x


def create_fusion_head(
    head_type: str = "mlp",
    text_dim: int = 2048,
    graph_dim: int = 256,
    output_dim: int = 512,
    **kwargs
) -> Union[MLPFusionHead, AttentionFusionHead, GatedFusionHead, ResidualFusionHead]:
    """
    Create fusion head instance.
    
    Args:
        head_type: Type of fusion head ("mlp", "attention", "gated", "residual")
        text_dim: Text embedding dimension
        graph_dim: Graph embedding dimension
        output_dim: Output dimension
        **kwargs: Additional arguments
        
    Returns:
        Fusion head instance
    """
    if head_type == "mlp":
        return MLPFusionHead(text_dim, graph_dim, output_dim, **kwargs)
    elif head_type == "attention":
        return AttentionFusionHead(text_dim, graph_dim, output_dim, **kwargs)
    elif head_type == "gated":
        return GatedFusionHead(text_dim, graph_dim, output_dim, **kwargs)
    elif head_type == "residual":
        return ResidualFusionHead(text_dim, graph_dim, output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown fusion head type: {head_type}")
