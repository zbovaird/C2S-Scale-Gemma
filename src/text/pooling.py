"""
Sequence pooling for text embeddings.

Implements various pooling strategies for aggregating sequence outputs.
"""

import logging
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SequencePooler(nn.Module):
    """
    Sequence pooler for text embeddings.
    
    Implements various pooling strategies for aggregating sequence outputs.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        pooling_method: str = "mean",
        use_cls_token: bool = False,
        cls_token_id: Optional[int] = None
    ):
        """
        Initialize sequence pooler.
        
        Args:
            hidden_dim: Hidden dimension of the model
            pooling_method: Pooling method ("mean", "max", "cls", "last", "attention")
            use_cls_token: Whether to use CLS token for pooling
            cls_token_id: CLS token ID (if using CLS token)
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.pooling_method = pooling_method
        self.use_cls_token = use_cls_token
        self.cls_token_id = cls_token_id
        
        # Attention pooling
        if pooling_method == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Learnable pooling weights
        if pooling_method == "weighted_mean":
            self.pooling_weights = nn.Parameter(torch.ones(1))
        
        logger.info(f"Initialized sequence pooler: {pooling_method}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through sequence pooler.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            input_ids: Input token IDs [batch_size, seq_len]
            
        Returns:
            Pooled embeddings [batch_size, hidden_dim]
        """
        if self.pooling_method == "mean":
            return self._mean_pooling(hidden_states, attention_mask)
        elif self.pooling_method == "max":
            return self._max_pooling(hidden_states, attention_mask)
        elif self.pooling_method == "cls":
            return self._cls_pooling(hidden_states, input_ids)
        elif self.pooling_method == "last":
            return self._last_pooling(hidden_states, attention_mask)
        elif self.pooling_method == "attention":
            return self._attention_pooling(hidden_states, attention_mask)
        elif self.pooling_method == "weighted_mean":
            return self._weighted_mean_pooling(hidden_states, attention_mask)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
    
    def _mean_pooling(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Mean pooling over sequence length.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Mean pooled embeddings [batch_size, hidden_dim]
        """
        if attention_mask is not None:
            # Mask out padding tokens
            attention_mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            masked_hidden_states = hidden_states * attention_mask
            
            # Sum over sequence length
            sum_hidden_states = torch.sum(masked_hidden_states, dim=1)
            
            # Divide by number of non-padding tokens
            seq_lengths = torch.sum(attention_mask, dim=1)
            pooled = sum_hidden_states / (seq_lengths + 1e-8)
        else:
            # Simple mean pooling
            pooled = torch.mean(hidden_states, dim=1)
        
        return pooled
    
    def _max_pooling(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Max pooling over sequence length.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Max pooled embeddings [batch_size, hidden_dim]
        """
        if attention_mask is not None:
            # Mask out padding tokens
            attention_mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            masked_hidden_states = hidden_states * attention_mask
            
            # Set padding tokens to very negative values
            masked_hidden_states = masked_hidden_states.masked_fill(
                attention_mask == 0, -1e9
            )
            
            # Max pooling
            pooled = torch.max(masked_hidden_states, dim=1)[0]
        else:
            # Simple max pooling
            pooled = torch.max(hidden_states, dim=1)[0]
        
        return pooled
    
    def _cls_pooling(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        CLS token pooling.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_dim]
            input_ids: Input token IDs [batch_size, seq_len]
            
        Returns:
            CLS token embeddings [batch_size, hidden_dim]
        """
        if self.use_cls_token and self.cls_token_id is not None and input_ids is not None:
            # Find CLS token positions
            cls_positions = (input_ids == self.cls_token_id).nonzero(as_tuple=True)
            
            if len(cls_positions[0]) > 0:
                # Extract CLS token embeddings
                batch_indices = cls_positions[0]
                seq_indices = cls_positions[1]
                
                pooled = torch.zeros(hidden_states.size(0), hidden_states.size(-1), device=hidden_states.device)
                pooled[batch_indices] = hidden_states[batch_indices, seq_indices]
                
                return pooled
        
        # Fallback to first token
        return hidden_states[:, 0, :]
    
    def _last_pooling(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Last token pooling.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Last token embeddings [batch_size, hidden_dim]
        """
        if attention_mask is not None:
            # Find last non-padding token for each sequence
            seq_lengths = torch.sum(attention_mask, dim=1) - 1  # -1 for 0-indexing
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            
            pooled = hidden_states[batch_indices, seq_lengths]
        else:
            # Use last token
            pooled = hidden_states[:, -1, :]
        
        return pooled
    
    def _attention_pooling(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Attention-based pooling.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Attention pooled embeddings [batch_size, hidden_dim]
        """
        batch_size = hidden_states.size(0)
        
        # Create query
        query = self.query.expand(batch_size, -1, -1)
        
        # Apply attention
        attn_output, _ = self.attention(
            query=query,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=attention_mask == 0 if attention_mask is not None else None
        )
        
        # Squeeze to get pooled embeddings
        pooled = attn_output.squeeze(1)
        
        return pooled
    
    def _weighted_mean_pooling(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Weighted mean pooling.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Weighted mean pooled embeddings [batch_size, hidden_dim]
        """
        if attention_mask is not None:
            # Create weights based on attention mask
            weights = attention_mask.float() * self.pooling_weights
            
            # Normalize weights
            weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-8)
            
            # Weighted sum
            pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)
        else:
            # Simple weighted mean
            weights = torch.softmax(self.pooling_weights, dim=0)
            pooled = torch.sum(hidden_states * weights, dim=1)
        
        return pooled


class MultiHeadPooler(nn.Module):
    """
    Multi-head pooler for different pooling strategies.
    
    Combines multiple pooling methods for robust representation.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        pooling_methods: list = ["mean", "max", "cls"],
        fusion_method: str = "concat"
    ):
        """
        Initialize multi-head pooler.
        
        Args:
            hidden_dim: Hidden dimension of the model
            pooling_methods: List of pooling methods to use
            fusion_method: Method to fuse pooled embeddings ("concat", "add", "attention")
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.pooling_methods = pooling_methods
        self.fusion_method = fusion_method
        
        # Create individual poolers
        self.poolers = nn.ModuleDict()
        for method in pooling_methods:
            self.poolers[method] = SequencePooler(
                hidden_dim=hidden_dim,
                pooling_method=method
            )
        
        # Fusion layer
        if fusion_method == "concat":
            self.fusion_layer = nn.Linear(hidden_dim * len(pooling_methods), hidden_dim)
        elif fusion_method == "add":
            self.fusion_layer = nn.Linear(hidden_dim, hidden_dim)
        elif fusion_method == "attention":
            self.fusion_layer = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            self.fusion_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        logger.info(f"Initialized multi-head pooler: {pooling_methods} -> {fusion_method}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through multi-head pooler.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            input_ids: Input token IDs [batch_size, seq_len]
            
        Returns:
            Fused pooled embeddings [batch_size, hidden_dim]
        """
        # Apply each pooling method
        pooled_embeddings = []
        for method in self.pooling_methods:
            pooled = self.poolers[method](hidden_states, attention_mask, input_ids)
            pooled_embeddings.append(pooled)
        
        # Fuse embeddings
        if self.fusion_method == "concat":
            # Concatenate all pooled embeddings
            fused = torch.cat(pooled_embeddings, dim=-1)
            fused = self.fusion_layer(fused)
        elif self.fusion_method == "add":
            # Add all pooled embeddings
            fused = torch.stack(pooled_embeddings, dim=1)  # [batch_size, num_methods, hidden_dim]
            fused = torch.sum(fused, dim=1)  # [batch_size, hidden_dim]
            fused = self.fusion_layer(fused)
        elif self.fusion_method == "attention":
            # Attention-based fusion
            stacked_embeddings = torch.stack(pooled_embeddings, dim=1)  # [batch_size, num_methods, hidden_dim]
            
            batch_size = stacked_embeddings.size(0)
            query = self.fusion_query.expand(batch_size, -1, -1)
            
            attn_output, _ = self.fusion_layer(
                query=query,
                key=stacked_embeddings,
                value=stacked_embeddings
            )
            
            fused = attn_output.squeeze(1)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return fused


def create_sequence_pooler(
    hidden_dim: int,
    pooling_method: str = "mean",
    use_cls_token: bool = False,
    cls_token_id: Optional[int] = None
) -> SequencePooler:
    """
    Create sequence pooler instance.
    
    Args:
        hidden_dim: Hidden dimension
        pooling_method: Pooling method
        use_cls_token: Whether to use CLS token
        cls_token_id: CLS token ID
        
    Returns:
        Sequence pooler instance
    """
    return SequencePooler(
        hidden_dim=hidden_dim,
        pooling_method=pooling_method,
        use_cls_token=use_cls_token,
        cls_token_id=cls_token_id
    )


def create_multi_head_pooler(
    hidden_dim: int,
    pooling_methods: list = ["mean", "max", "cls"],
    fusion_method: str = "concat"
) -> MultiHeadPooler:
    """
    Create multi-head pooler instance.
    
    Args:
        hidden_dim: Hidden dimension
        pooling_methods: List of pooling methods
        fusion_method: Fusion method
        
    Returns:
        Multi-head pooler instance
    """
    return MultiHeadPooler(
        hidden_dim=hidden_dim,
        pooling_methods=pooling_methods,
        fusion_method=fusion_method
    )
