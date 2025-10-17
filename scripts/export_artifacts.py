#!/usr/bin/env python3
"""
Script for exporting artifacts and model weights.

This script exports the trained model weights, configurations, and artifacts
for deployment and sharing.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import shutil
import json
import zipfile

import torch
import torch.nn as nn
from omegaconf import OmegaConf
import mlflow
import mlflow.pytorch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hgnn.encoder import UHGHGNNEncoder
from text.gemma_loader import GemmaLoader
from text.adapters import LoRAAdapter
from fusion.heads import FusionHead
from fusion.trainer import DualEncoderTrainer


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/export_artifacts.log"),
        ],
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from TOML file."""
    config = OmegaConf.load(config_path)
    return OmegaConf.to_container(config, resolve=True)


def export_model_weights(
    checkpoint_path: str,
    output_dir: Path,
    device: torch.device,
) -> None:
    """Export model weights in various formats."""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    
    # Create models
    hgnn_encoder = UHGHGNNEncoder(
        input_dim=config["model"]["hgnn"]["input_dim"],
        hidden_dim=config["model"]["hgnn"]["hidden_dim"],
        output_dim=config["model"]["hgnn"]["output_dim"],
        num_layers=config["model"]["hgnn"]["num_layers"],
        dropout=config["model"]["hgnn"]["dropout"],
        curvature=config["model"]["hgnn"]["curvature"],
        device=device,
    )
    
    gemma_loader = GemmaLoader(
        model_name=config["model"]["text"]["model_name"],
        device=device,
        torch_dtype=torch.bfloat16,
        quantization_config=config["model"]["text"]["quantization"],
    )
    
    gemma_model, tokenizer = gemma_loader.load_model()
    
    lora_adapter = LoRAAdapter(
        model=gemma_model,
        r=config["model"]["text"]["lora"]["r"],
        lora_alpha=config["model"]["text"]["lora"]["alpha"],
        lora_dropout=config["model"]["text"]["lora"]["dropout"],
        target_modules=config["model"]["text"]["lora"]["target_modules"],
    )
    
    fusion_head = FusionHead(
        graph_dim=config["model"]["hgnn"]["output_dim"],
        text_dim=config["model"]["text"]["hidden_size"],
        fusion_dim=config["model"]["fusion"]["dim"],
        dropout=config["model"]["fusion"]["dropout"],
    )
    
    # Load weights
    trainer_state_dict = checkpoint["trainer_state_dict"]
    hgnn_encoder.load_state_dict({
        k.replace("hgnn_encoder.", ""): v 
        for k, v in trainer_state_dict.items() 
        if k.startswith("hgnn_encoder.")
    })
    lora_adapter.load_state_dict({
        k.replace("text_model.", ""): v 
        for k, v in trainer_state_dict.items() 
        if k.startswith("text_model.")
    })
    fusion_head.load_state_dict({
        k.replace("fusion_head.", ""): v 
        for k, v in trainer_state_dict.items() 
        if k.startswith("fusion_head.")
    })
    
    # Export individual components
    torch.save(hgnn_encoder.state_dict(), output_dir / "hgnn_encoder.pt")
    torch.save(lora_adapter.state_dict(), output_dir / "lora_adapter.pt")
    torch.save(fusion_head.state_dict(), output_dir / "fusion_head.pt")
    
    # Export tokenizer
    tokenizer.save_pretrained(output_dir / "tokenizer")
    
    # Export full model
    torch.save(checkpoint, output_dir / "full_checkpoint.pt")


def export_configurations(
    config_path: str,
    output_dir: Path,
) -> None:
    """Export configuration files."""
    
    # Copy original config
    shutil.copy2(config_path, output_dir / "config.toml")
    
    # Export as JSON for easier loading
    config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)


def export_artifacts(
    artifacts_dir: str,
    output_dir: Path,
) -> None:
    """Export training artifacts."""
    
    artifacts_path = Path(artifacts_dir)
    if not artifacts_path.exists():
        return
    
    # Copy MLflow runs
    mlruns_dir = artifacts_path / "mlruns"
    if mlruns_dir.exists():
        shutil.copytree(mlruns_dir, output_dir / "mlruns", dirs_exist_ok=True)
    
    # Copy logs
    logs_dir = Path("logs")
    if logs_dir.exists():
        shutil.copytree(logs_dir, output_dir / "logs", dirs_exist_ok=True)
    
    # Copy other artifacts
    for item in artifacts_path.iterdir():
        if item.is_file() and item.suffix in [".pt", ".pth", ".json", ".csv", ".png", ".jpg"]:
            shutil.copy2(item, output_dir / item.name)


def create_model_card(
    config: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Create model card for the exported model."""
    
    model_card = f"""---
license: cc-by-nc-nd-4.0
tags:
- single-cell
- transcriptomics
- language-models
- hyperbolic-geometry
- graph-neural-networks
- gemma
- lora
- contrastive-learning
---

# C2S-Scale-Gemma Hybrid Model

## Model Description

The C2S-Scale-Gemma Hybrid Model is a dual-encoder + late-fusion architecture that combines:

- **UHG-HGNN Encoder**: Hyperbolic Graph Neural Network for graph signal processing
- **C2S-Scale-Gemma Text Encoder**: Large language model for text processing
- **LoRA Adapters**: Parameter-efficient fine-tuning
- **Contrastive Alignment**: InfoNCE loss with hard negative mining
- **Late Fusion**: Combines graph and text representations

## Model Architecture

### Graph Encoder (UHG-HGNN)
- Input Dimension: {config['model']['hgnn']['input_dim']}
- Hidden Dimension: {config['model']['hgnn']['hidden_dim']}
- Output Dimension: {config['model']['hgnn']['output_dim']}
- Number of Layers: {config['model']['hgnn']['num_layers']}
- Curvature: {config['model']['hgnn']['curvature']}

### Text Encoder (Gemma)
- Model: {config['model']['text']['model_name']}
- LoRA Rank: {config['model']['text']['lora']['r']}
- LoRA Alpha: {config['model']['text']['lora']['alpha']}
- Target Modules: {config['model']['text']['lora']['target_modules']}

### Fusion Head
- Fusion Dimension: {config['model']['fusion']['dim']}
- Dropout: {config['model']['fusion']['dropout']}

## Training Configuration

- Learning Rate: {config['training']['learning_rate']}
- Batch Size: {config['training']['batch_size']}
- Number of Epochs: {config['training']['num_epochs']}
- Contrastive Temperature: {config['training']['contrastive_temperature']}
- Hard Negative Weight: {config['training']['hard_negative_weight']}

## Usage

```python
import torch
from src.hgnn.encoder import UHGHGNNEncoder
from src.text.adapters import LoRAAdapter
from src.fusion.heads import FusionHead

# Load model components
hgnn_encoder = UHGHGNNEncoder(...)
lora_adapter = LoRAAdapter(...)
fusion_head = FusionHead(...)

# Load weights
hgnn_encoder.load_state_dict(torch.load("hgnn_encoder.pt"))
lora_adapter.load_state_dict(torch.load("lora_adapter.pt"))
fusion_head.load_state_dict(torch.load("fusion_head.pt"))
```

## Files

- `hgnn_encoder.pt`: HGNN encoder weights
- `lora_adapter.pt`: LoRA adapter weights
- `fusion_head.pt`: Fusion head weights
- `tokenizer/`: Tokenizer files
- `config.json`: Model configuration
- `full_checkpoint.pt`: Complete checkpoint

## Citation

```bibtex
@article{{c2s_scale_gemma_hybrid,
  title={{C2S-Scale-Gemma Hybrid: Dual-Encoder + Late-Fusion for Single-Cell Transcriptomics}},
  author={{Bovaird, Zach}},
  year={{2024}},
  license={{CC BY-NC-ND 4.0}}
}}
```

## License

This model is licensed under CC BY-NC-ND 4.0. See LICENSE file for details.
"""
    
    with open(output_dir / "README.md", "w") as f:
        f.write(model_card)


def create_deployment_package(
    output_dir: Path,
    package_name: str = "c2s_scale_gemma_model",
) -> None:
    """Create deployment package."""
    
    package_path = output_dir.parent / f"{package_name}.zip"
    
    with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(output_dir)
                zipf.write(file_path, arcname)
    
    return package_path


def main():
    """Main export function."""
    
    parser = argparse.ArgumentParser(description="Export C2S-Scale-Gemma model artifacts")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="artifacts/export", help="Output directory")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Artifacts directory")
    parser.add_argument("--package_name", type=str, default="c2s_scale_gemma_model", help="Package name")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export model weights
    logger.info("Exporting model weights...")
    export_model_weights(args.checkpoint, output_dir, device)
    
    # Export configurations
    logger.info("Exporting configurations...")
    export_configurations(args.config, output_dir)
    
    # Export artifacts
    logger.info("Exporting artifacts...")
    export_artifacts(args.artifacts_dir, output_dir)
    
    # Create model card
    logger.info("Creating model card...")
    create_model_card(config, output_dir)
    
    # Create deployment package
    logger.info("Creating deployment package...")
    package_path = create_deployment_package(output_dir, args.package_name)
    
    logger.info(f"Export completed! Files saved to {output_dir}")
    logger.info(f"Deployment package created: {package_path}")
    
    # List exported files
    logger.info("Exported files:")
    for file_path in output_dir.rglob("*"):
        if file_path.is_file():
            logger.info(f"  - {file_path.relative_to(output_dir)}")


if __name__ == "__main__":
    main()
