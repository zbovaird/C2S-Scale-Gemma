# C2S-Scale-Gemma Hybrid

A dual-encoder hybrid system combining UHG-HGNN for graph signals and C2S-Scale-Gemma text encoder, aligned via contrastive loss and fused through LoRA adapters.

## Overview

This project implements a hybrid architecture that combines:
- **UHG-HGNN**: Hyperbolic Graph Neural Networks for single-cell graph representations
- **C2S-Scale-Gemma**: Large Language Model fine-tuned for single-cell biology
- **Contrastive Alignment**: InfoNCE loss to align graph and text embeddings
- **LoRA Fusion**: Parameter-efficient fusion via Low-Rank Adaptation

## Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/zbovaird/C2S-Scale-Gemma.git
cd C2S-Scale-Gemma

# Install dependencies
uv sync
```

## Quick Start

```bash
# Download and preprocess data
uv run scripts/download_data.py --cfg configs/datasets.toml

# Build graphs
uv run scripts/build_graphs.py --cfg configs/datasets.toml

# Train the hybrid model
uv run scripts/align_dual_encoder.py --cfg configs/colab_7b.toml
```

## Configuration

- `configs/colab_7b.toml`: Colab prototype settings (7B model)
- `configs/vertex_27b.toml`: Vertex AI production settings (27B model)
- `configs/datasets.toml`: Data and graph parameters
- `configs/ablations.toml`: Ablation study configurations

## License

CC BY-NC-ND 4.0
