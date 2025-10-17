"""
C2S-Scale-Gemma model loader with proper integration.

This module loads the actual C2S-Scale-Gemma model from HuggingFace
and provides proper integration with the hybrid architecture.
"""

import logging
import torch
from typing import Dict, Any, Optional, Tuple, List
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)


class C2SScaleGemmaLoader:
    """
    Loader for C2S-Scale-Gemma model with proper integration.
    
    This class loads the actual C2S-Scale-Gemma model from HuggingFace
    and provides methods for cell type prediction and other tasks.
    """
    
    def __init__(
        self,
        model_name: str = "vandijklab/C2S-Scale-Gemma-2-27B",
        device: torch.device = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        quantization_config: Optional[Dict[str, Any]] = None,
        use_auth_token: Optional[str] = None
    ):
        """
        Initialize the C2S-Scale-Gemma loader.
        
        Args:
            model_name: HuggingFace model name
            device: Device to load model on
            torch_dtype: Data type for model
            quantization_config: Quantization configuration
            use_auth_token: HuggingFace authentication token
        """
        self.model_name = model_name
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.torch_dtype = torch_dtype
        self.quantization_config = quantization_config
        self.use_auth_token = use_auth_token
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model()
        
        logger.info(f"Loaded C2S-Scale-Gemma model: {model_name}")
        logger.info(f"Model device: {self.device}")
        logger.info(f"Model dtype: {self.torch_dtype}")
        
    def _load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load the C2S-Scale-Gemma model and tokenizer."""
        logger.info(f"Loading C2S-Scale-Gemma model from {self.model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.use_auth_token,
            use_auth_token=self.use_auth_token is not None
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model
        if self.quantization_config and self.quantization_config.get('load_in_4bit', False):
            # Use 4-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.quantization_config['load_in_4bit'],
                bnb_4bit_compute_dtype=self.quantization_config.get('bnb_4bit_compute_dtype', torch.bfloat16),
                bnb_4bit_use_double_quant=self.quantization_config.get('bnb_4bit_use_double_quant', True),
                bnb_4bit_quant_type=self.quantization_config.get('bnb_4bit_quant_type', 'nf4')
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                token=self.use_auth_token,
                use_auth_token=self.use_auth_token is not None
            )
        else:
            # Load without quantization
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                token=self.use_auth_token,
                use_auth_token=self.use_auth_token is not None
            )
            
        return model, tokenizer
    
    def create_cell_type_prompt(
        self, 
        cell_sentence: str, 
        num_genes: int = 1000, 
        organism: str = "Homo sapiens"
    ) -> str:
        """
        Create C2S-Scale-Gemma formatted prompt for cell type prediction.
        
        Args:
            cell_sentence: Space-separated gene names ordered by expression
            num_genes: Number of genes in the sentence
            organism: Organism name
            
        Returns:
            Formatted prompt for cell type prediction
        """
        prompt = f"""The following is a list of {num_genes} gene names ordered by descending expression level in a {organism} cell. Your task is to give the cell type which this cell belongs to based on its gene expression.
Cell sentence: {cell_sentence}.
The cell type corresponding to these genes is:"""
        
        return prompt
    
    def create_tissue_prompt(
        self, 
        cell_sentence: str, 
        num_genes: int = 1000, 
        organism: str = "Homo sapiens"
    ) -> str:
        """
        Create C2S-Scale-Gemma formatted prompt for tissue prediction.
        
        Args:
            cell_sentence: Space-separated gene names ordered by expression
            num_genes: Number of genes in the sentence
            organism: Organism name
            
        Returns:
            Formatted prompt for tissue prediction
        """
        prompt = f"""The following is a list of {num_genes} gene names ordered by descending expression level in a {organism} cell. Your task is to give the tissue which this cell belongs to based on its gene expression.
Cell sentence: {cell_sentence}.
The tissue corresponding to these genes is:"""
        
        return prompt
    
    def predict_cell_type(
        self, 
        cell_sentence: str, 
        max_new_tokens: int = 20,
        num_genes: int = 1000,
        organism: str = "Homo sapiens"
    ) -> str:
        """
        Predict cell type using C2S-Scale-Gemma model.
        
        Args:
            cell_sentence: Space-separated gene names ordered by expression
            max_new_tokens: Maximum number of new tokens to generate
            num_genes: Number of genes in the sentence
            organism: Organism name
            
        Returns:
            Predicted cell type
        """
        # Create prompt
        prompt = self.create_cell_type_prompt(cell_sentence, num_genes, organism)
        
        # Tokenize
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **input_ids, 
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract predicted cell type
        predicted_cell_type = response.split("The cell type corresponding to these genes is:")[1].strip()
        
        return predicted_cell_type
    
    def predict_tissue(
        self, 
        cell_sentence: str, 
        max_new_tokens: int = 20,
        num_genes: int = 1000,
        organism: str = "Homo sapiens"
    ) -> str:
        """
        Predict tissue using C2S-Scale-Gemma model.
        
        Args:
            cell_sentence: Space-separated gene names ordered by expression
            max_new_tokens: Maximum number of new tokens to generate
            num_genes: Number of genes in the sentence
            organism: Organism name
            
        Returns:
            Predicted tissue
        """
        # Create prompt
        prompt = self.create_tissue_prompt(cell_sentence, num_genes, organism)
        
        # Tokenize
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **input_ids, 
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract predicted tissue
        predicted_tissue = response.split("The tissue corresponding to these genes is:")[1].strip()
        
        return predicted_tissue
    
    def get_text_embeddings(self, prompts: List[str]) -> torch.Tensor:
        """
        Get text embeddings from C2S-Scale-Gemma model.
        
        Args:
            prompts: List of prompts to encode
            
        Returns:
            Text embeddings tensor
        """
        # Tokenize prompts
        tokenized = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**tokenized, output_hidden_states=True)
            # Use mean pooling of last hidden states
            embeddings = outputs.hidden_states[-1].mean(dim=1)
            
        return embeddings
    
    def add_lora_adapters(
        self, 
        r: int = 16, 
        lora_alpha: int = 32, 
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None
    ):
        """
        Add LoRA adapters to the C2S-Scale-Gemma model.
        
        Args:
            r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            target_modules: Target modules for LoRA
        """
        if target_modules is None:
            # Default target modules for Gemma
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        logger.info(f"Added LoRA adapters with r={r}, alpha={lora_alpha}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'dtype': str(self.torch_dtype),
            'vocab_size': self.tokenizer.vocab_size,
            'max_length': self.tokenizer.model_max_length,
            'quantized': self.quantization_config is not None and self.quantization_config.get('load_in_4bit', False)
        }
