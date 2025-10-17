"""
LoRA adapters for Gemma model.

Implements LoRA (Low-Rank Adaptation) adapters for efficient fine-tuning.
"""

import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class LoRAAdapter(nn.Module):
    """
    LoRA adapter module.
    
    Implements Low-Rank Adaptation for efficient fine-tuning.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.05,
        bias: str = "none"
    ):
        """
        Initialize LoRA adapter.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            rank: LoRA rank
            alpha: LoRA alpha parameter
            dropout: Dropout probability
            bias: Bias type ("none", "all", "lora_only")
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.bias = bias
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Bias
        if bias == "all":
            self.bias = nn.Parameter(torch.zeros(out_features))
        elif bias == "lora_only":
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        
        logger.info(f"Initialized LoRA adapter: {in_features} -> {out_features}, rank={rank}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA adapter.
        
        Args:
            x: Input tensor [..., in_features]
            
        Returns:
            Output tensor [..., out_features]
        """
        # LoRA computation: x @ A^T @ B^T
        # A: [rank, in_features], B: [out_features, rank]
        # x: [..., in_features]
        
        # Apply LoRA A
        lora_A_out = torch.matmul(x, self.lora_A.t())  # [..., rank]
        
        # Apply dropout
        lora_A_out = self.dropout_layer(lora_A_out)
        
        # Apply LoRA B
        lora_out = torch.matmul(lora_A_out, self.lora_B.t())  # [..., out_features]
        
        # Scale by alpha/rank
        lora_out = lora_out * (self.alpha / self.rank)
        
        # Add bias if specified
        if self.bias is not None:
            lora_out = lora_out + self.bias
        
        return lora_out
    
    def get_scaling_factor(self) -> float:
        """Get LoRA scaling factor."""
        return self.alpha / self.rank


class LoRAManager:
    """
    Manager for LoRA adapters.
    
    Handles creation, application, and management of LoRA adapters.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        lora_config: Dict,
        target_modules: Optional[List[str]] = None
    ):
        """
        Initialize LoRA manager.
        
        Args:
            model: Pre-trained model
            lora_config: LoRA configuration
            target_modules: Target modules for LoRA
        """
        self.model = model
        self.lora_config = lora_config
        self.target_modules = target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        
        self.lora_model = None
        self.applied_adapters = {}
        
        logger.info(f"Initialized LoRA manager with target modules: {self.target_modules}")
    
    def apply_lora(self) -> PreTrainedModel:
        """
        Apply LoRA adapters to the model.
        
        Returns:
            Model with LoRA adapters
        """
        # Create LoRA configuration
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_config.get('r', 16),
            lora_alpha=self.lora_config.get('alpha', 16),
            lora_dropout=self.lora_config.get('dropout', 0.05),
            target_modules=self.target_modules,
            bias=self.lora_config.get('bias', 'none'),
            inference_mode=False
        )
        
        # Apply LoRA
        self.lora_model = get_peft_model(self.model, peft_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.lora_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.lora_model.parameters())
        
        logger.info(f"Applied LoRA adapters: {trainable_params:,} trainable parameters out of {total_params:,} total")
        
        return self.lora_model
    
    def add_adapter(
        self,
        adapter_name: str,
        target_modules: Optional[List[str]] = None,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.05
    ) -> None:
        """
        Add a new LoRA adapter.
        
        Args:
            adapter_name: Name of the adapter
            target_modules: Target modules for this adapter
            rank: LoRA rank
            alpha: LoRA alpha
            dropout: Dropout probability
        """
        if target_modules is None:
            target_modules = self.target_modules
        
        # Create adapter configuration
        adapter_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias=self.lora_config.get('bias', 'none'),
            inference_mode=False
        )
        
        # Add adapter
        self.lora_model.add_adapter(adapter_name, adapter_config)
        self.applied_adapters[adapter_name] = adapter_config
        
        logger.info(f"Added LoRA adapter: {adapter_name}")
    
    def set_active_adapters(self, adapter_names: List[str]) -> None:
        """
        Set active adapters.
        
        Args:
            adapter_names: List of adapter names to activate
        """
        self.lora_model.set_adapter(adapter_names)
        logger.info(f"Set active adapters: {adapter_names}")
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """
        Get trainable parameters.
        
        Returns:
            List of trainable parameters
        """
        return [p for p in self.lora_model.parameters() if p.requires_grad]
    
    def save_adapters(self, save_path: str) -> None:
        """
        Save LoRA adapters.
        
        Args:
            save_path: Path to save adapters
        """
        self.lora_model.save_pretrained(save_path)
        logger.info(f"Saved LoRA adapters to {save_path}")
    
    def load_adapters(self, load_path: str) -> None:
        """
        Load LoRA adapters.
        
        Args:
            load_path: Path to load adapters from
        """
        self.lora_model = PeftModel.from_pretrained(self.model, load_path)
        logger.info(f"Loaded LoRA adapters from {load_path}")


class LoRAFusionAdapter(nn.Module):
    """
    LoRA fusion adapter for combining multiple modalities.
    
    Implements LoRA adapters specifically designed for fusion tasks.
    """
    
    def __init__(
        self,
        text_dim: int,
        graph_dim: int,
        output_dim: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.05,
        fusion_method: str = "concat"
    ):
        """
        Initialize LoRA fusion adapter.
        
        Args:
            text_dim: Text embedding dimension
            graph_dim: Graph embedding dimension
            output_dim: Output dimension
            rank: LoRA rank
            alpha: LoRA alpha
            dropout: Dropout probability
            fusion_method: Fusion method ("concat", "add", "multiply")
        """
        super().__init__()
        
        self.text_dim = text_dim
        self.graph_dim = graph_dim
        self.output_dim = output_dim
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.fusion_method = fusion_method
        
        # Fusion layer
        if fusion_method == "concat":
            input_dim = text_dim + graph_dim
        elif fusion_method == "add":
            input_dim = text_dim  # Assume text_dim == graph_dim
        elif fusion_method == "multiply":
            input_dim = text_dim  # Assume text_dim == graph_dim
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # LoRA adapter for fusion
        self.fusion_adapter = LoRAAdapter(
            in_features=input_dim,
            out_features=output_dim,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # Projection layers for different modalities
        self.text_projection = nn.Linear(text_dim, text_dim)
        self.graph_projection = nn.Linear(graph_dim, graph_dim)
        
        logger.info(f"Initialized LoRA fusion adapter: {fusion_method} fusion")
    
    def forward(
        self,
        text_embeddings: torch.Tensor,
        graph_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through fusion adapter.
        
        Args:
            text_embeddings: Text embeddings [batch_size, text_dim]
            graph_embeddings: Graph embeddings [batch_size, graph_dim]
            
        Returns:
            Fused embeddings [batch_size, output_dim]
        """
        # Project embeddings
        text_proj = self.text_projection(text_embeddings)
        graph_proj = self.graph_projection(graph_embeddings)
        
        # Fuse embeddings
        if self.fusion_method == "concat":
            fused = torch.cat([text_proj, graph_proj], dim=-1)
        elif self.fusion_method == "add":
            fused = text_proj + graph_proj
        elif self.fusion_method == "multiply":
            fused = text_proj * graph_proj
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        # Apply LoRA adapter
        output = self.fusion_adapter(fused)
        
        return output


def create_lora_config(
    r: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none"
) -> Dict:
    """
    Create LoRA configuration dictionary.
    
    Args:
        r: LoRA rank
        alpha: LoRA alpha
        dropout: Dropout probability
        target_modules: Target modules
        bias: Bias type
        
    Returns:
        LoRA configuration dictionary
    """
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    
    return {
        'r': r,
        'alpha': alpha,
        'dropout': dropout,
        'target_modules': target_modules,
        'bias': bias
    }


def create_lora_manager(
    model: PreTrainedModel,
    lora_config: Dict,
    target_modules: Optional[List[str]] = None
) -> LoRAManager:
    """
    Create LoRA manager instance.
    
    Args:
        model: Pre-trained model
        lora_config: LoRA configuration
        target_modules: Target modules
        
    Returns:
        LoRA manager instance
    """
    return LoRAManager(model, lora_config, target_modules)


def create_fusion_adapter(
    text_dim: int,
    graph_dim: int,
    output_dim: int,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.05,
    fusion_method: str = "concat"
) -> LoRAFusionAdapter:
    """
    Create LoRA fusion adapter instance.
    
    Args:
        text_dim: Text embedding dimension
        graph_dim: Graph embedding dimension
        output_dim: Output dimension
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: Dropout probability
        fusion_method: Fusion method
        
    Returns:
        LoRA fusion adapter instance
    """
    return LoRAFusionAdapter(
        text_dim=text_dim,
        graph_dim=graph_dim,
        output_dim=output_dim,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        fusion_method=fusion_method
    )
