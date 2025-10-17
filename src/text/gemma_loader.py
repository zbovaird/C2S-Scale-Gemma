"""
Gemma model loader for C2S-Scale-Gemma Hybrid project.

Loads and configures Gemma models with quantization and LoRA adapters.
"""

import logging
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GemmaConfig,
    GemmaForCausalLM,
    GemmaTokenizer
)
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)


class GemmaLoader:
    """
    Gemma model loader with quantization and LoRA support.
    
    Handles loading of Gemma models with 4-bit quantization and LoRA adapters.
    """
    
    def __init__(
        self,
        model_name: str,
        load_4bit: bool = True,
        lora_config: Optional[Dict] = None,
        gradient_checkpointing: bool = True,
        max_seq_length: int = 2048,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Gemma loader.
        
        Args:
            model_name: HuggingFace model name or path
            load_4bit: Whether to use 4-bit quantization
            lora_config: LoRA configuration dictionary
            gradient_checkpointing: Whether to enable gradient checkpointing
            max_seq_length: Maximum sequence length
            cache_dir: Cache directory for model files
        """
        self.model_name = model_name
        self.load_4bit = load_4bit
        self.lora_config = lora_config or {}
        self.gradient_checkpointing = gradient_checkpointing
        self.max_seq_length = max_seq_length
        self.cache_dir = cache_dir
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.lora_model = None
        
        logger.info(f"Initialized Gemma loader for {model_name}")
    
    def load_model(self) -> Tuple[nn.Module, nn.Module]:
        """
        Load Gemma model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading Gemma model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = self._load_tokenizer()
        
        # Load model
        self.model = self._load_model()
        
        # Apply LoRA if configured
        if self.lora_config:
            self.lora_model = self._apply_lora()
            return self.lora_model, self.tokenizer
        
        return self.model, self.tokenizer
    
    def _load_tokenizer(self) -> nn.Module:
        """Load tokenizer."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Failed to load tokenizer with AutoTokenizer: {e}")
            # Fallback to GemmaTokenizer
            tokenizer = GemmaTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size}")
        return tokenizer
    
    def _load_model(self) -> nn.Module:
        """Load model with quantization if specified."""
        # Configure quantization
        quantization_config = None
        if self.load_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
        
        # Load model
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
        except Exception as e:
            logger.warning(f"Failed to load model with AutoModelForCausalLM: {e}")
            # Fallback to GemmaForCausalLM
            model = GemmaForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                cache_dir=self.cache_dir,
                torch_dtype=torch.bfloat16
            )
        
        # Enable gradient checkpointing
        if self.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        # Freeze base model parameters
        self._freeze_base_model(model)
        
        logger.info(f"Loaded model with {sum(p.numel() for p in model.parameters())} parameters")
        return model
    
    def _freeze_base_model(self, model: nn.Module) -> None:
        """Freeze base model parameters."""
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze layer norms if specified
        for module in model.modules():
            if isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
                for param in module.parameters():
                    param.requires_grad = True
        
        logger.info("Frozen base model parameters (except layer norms)")
    
    def _apply_lora(self) -> nn.Module:
        """Apply LoRA adapters to the model."""
        if not self.lora_config:
            return self.model
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_config.get('r', 16),
            lora_alpha=self.lora_config.get('alpha', 16),
            lora_dropout=self.lora_config.get('dropout', 0.05),
            target_modules=self.lora_config.get('target_modules', [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            bias=self.lora_config.get('bias', 'none'),
            inference_mode=False
        )
        
        # Apply LoRA
        lora_model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in lora_model.parameters())
        
        logger.info(f"Applied LoRA adapters: {trainable_params:,} trainable parameters out of {total_params:,} total")
        
        return lora_model
    
    def get_model_info(self) -> Dict[str, Union[int, str]]:
        """
        Get model information.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {}
        
        info = {
            'model_name': self.model_name,
            'vocab_size': self.tokenizer.vocab_size if self.tokenizer else 0,
            'max_seq_length': self.max_seq_length,
            'load_4bit': self.load_4bit,
            'gradient_checkpointing': self.gradient_checkpointing,
            'has_lora': self.lora_model is not None
        }
        
        if self.model is not None:
            info['total_parameters'] = sum(p.numel() for p in self.model.parameters())
            info['trainable_parameters'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return info


class GemmaTextEncoder(nn.Module):
    """
    Text encoder wrapper for Gemma model.
    
    Provides a unified interface for text encoding with Gemma.
    """
    
    def __init__(
        self,
        model_name: str,
        load_4bit: bool = True,
        lora_config: Optional[Dict] = None,
        gradient_checkpointing: bool = True,
        max_seq_length: int = 2048,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Gemma text encoder.
        
        Args:
            model_name: HuggingFace model name
            load_4bit: Whether to use 4-bit quantization
            lora_config: LoRA configuration
            gradient_checkpointing: Whether to enable gradient checkpointing
            max_seq_length: Maximum sequence length
            cache_dir: Cache directory
        """
        super().__init__()
        
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        
        # Load model and tokenizer
        loader = GemmaLoader(
            model_name=model_name,
            load_4bit=load_4bit,
            lora_config=lora_config,
            gradient_checkpointing=gradient_checkpointing,
            max_seq_length=max_seq_length,
            cache_dir=cache_dir
        )
        
        self.model, self.tokenizer = loader.load_model()
        self.model_info = loader.get_model_info()
        
        logger.info(f"Initialized Gemma text encoder: {self.model_info}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the text encoder.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_hidden_states: Whether to return hidden states
            
        Returns:
            Dictionary containing model outputs
        """
        # Prepare inputs
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'output_hidden_states': return_hidden_states,
            'return_dict': True
        }
        
        # Forward pass
        outputs = self.model(**inputs)
        
        return {
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states if return_hidden_states else None,
            'last_hidden_state': outputs.hidden_states[-1] if return_hidden_states else None
        }
    
    def encode_text(
        self,
        text: Union[str, list],
        max_length: Optional[int] = None,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text or list of texts
            max_length: Maximum sequence length
            return_tensors: Return tensor format
            
        Returns:
            Dictionary containing encoded inputs
        """
        if max_length is None:
            max_length = self.max_seq_length
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors=return_tensors
        )
        
        return encoded
    
    def get_hidden_dim(self) -> int:
        """Get hidden dimension of the model."""
        if hasattr(self.model.config, 'hidden_size'):
            return self.model.config.hidden_size
        elif hasattr(self.model.config, 'd_model'):
            return self.model.config.d_model
        else:
            # Default fallback
            return 2048
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.vocab_size
    
    def save_model(self, save_path: str) -> None:
        """
        Save the model and tokenizer.
        
        Args:
            save_path: Path to save the model
        """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Saved model and tokenizer to {save_path}")
    
    def load_model(self, load_path: str) -> None:
        """
        Load the model and tokenizer.
        
        Args:
            load_path: Path to load the model from
        """
        self.model = AutoModelForCausalLM.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        logger.info(f"Loaded model and tokenizer from {load_path}")


def create_gemma_encoder(
    model_name: str,
    load_4bit: bool = True,
    lora_config: Optional[Dict] = None,
    **kwargs
) -> GemmaTextEncoder:
    """
    Create a Gemma text encoder instance.
    
    Args:
        model_name: HuggingFace model name
        load_4bit: Whether to use 4-bit quantization
        lora_config: LoRA configuration
        **kwargs: Additional arguments
        
    Returns:
        Gemma text encoder instance
    """
    return GemmaTextEncoder(
        model_name=model_name,
        load_4bit=load_4bit,
        lora_config=lora_config,
        **kwargs
    )
