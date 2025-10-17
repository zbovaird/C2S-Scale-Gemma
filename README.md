# C2S-Scale-Gemma Hybrid Model

A state-of-the-art dual-encoder + late-fusion hybrid model for single-cell transcriptomics analysis, combining hyperbolic graph neural networks with large language models.

## 🚀 Quick Start

### Colab Prototype (A100 GPU)

1. **Open in Colab**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zbovaird/C2S-Scale-Gemma/blob/main/notebooks/colab_prototype.ipynb)

2. **Select A100 GPU**: Runtime → Change runtime type → GPU → A100

3. **Run the notebook**: All cells are optimized for A100 performance

### Local Development

```bash
# Clone repository
git clone https://github.com/zbovaird/C2S-Scale-Gemma.git
cd C2S-Scale-Gemma

# Install dependencies with uv
uv sync

# Download data
uv run scripts/download_data.py

# Build graphs
uv run scripts/build_graphs.py --cfg configs/colab_7b.toml

# Train model
uv run scripts/align_dual_encoder.py --cfg configs/colab_7b.toml
```

## 🏗️ Architecture

The C2S-Scale-Gemma hybrid model combines:

- **UHG-HGNN Encoder**: Hyperbolic Graph Neural Network for graph signal processing
- **C2S-Scale-Gemma Text Encoder**: Large language model for text processing  
- **LoRA Adapters**: Parameter-efficient fine-tuning
- **Contrastive Alignment**: InfoNCE loss with hard negative mining
- **Late Fusion**: Combines graph and text representations

### Key Features

- **Hyperbolic Geometry**: UHG operations for hierarchical cell relationships
- **4-bit Quantization**: Efficient memory usage with bitsandbytes
- **Flash Attention**: Faster attention computation
- **Gradient Accumulation**: Effective larger batch sizes
- **A100 Optimizations**: TensorFloat-32, cuDNN benchmarking

## 📊 Performance

### A100 GPU Performance
- **Model Size**: Gemma-9B with LoRA adapters
- **Training Time**: ~2-3 hours for 10 epochs
- **Memory Usage**: ~60-70GB GPU memory
- **Throughput**: ~100-150 samples/second
- **Expected Results**: ARI > 0.7, NMI > 0.8

### Model Variants
- **Colab (7B)**: `google/gemma-2-7b` - 2-3GB memory
- **A100 (9B)**: `google/gemma-2-9b` - 60-70GB memory  
- **Vertex AI (27B)**: `vandijklab/C2S-Scale-Gemma-2-27B` - Production scale

## 🛠️ Installation

### Requirements
- Python 3.9+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM (32GB+ recommended for A100)
- A100 GPU (recommended) or compatible GPU

### Dependencies
```toml
# Core ML
torch>=2.3.0
transformers>=4.43.0
accelerate>=1.1.0
bitsandbytes>=0.43.0
peft>=0.11.0

# UHG Library (custom)
uhg

# Single-cell analysis
scanpy>=1.9.0
anndata>=0.10.0
umap-learn>=0.5.0

# Graph processing
networkx>=3.2.0
pynndescent>=0.5.0

# Training infrastructure
mlflow>=2.14.0
omegaconf>=2.3.0
wandb>=0.17.0
```

## 📁 Project Structure

```
C2S-Scale-Gemma/
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── graphs/            # Graph construction (kNN, L-R, GRN)
│   ├── hgnn/              # Hyperbolic GNN encoder
│   ├── text/              # Gemma text encoder with LoRA
│   ├── fusion/            # Dual-encoder alignment and fusion
│   ├── eval/              # Evaluation tasks and metrics
│   └── uhg_adapters/      # UHG library adapters
├── scripts/               # Executable workflows
├── configs/               # TOML configuration files
├── notebooks/             # Colab prototype
└── docs/                  # Documentation
```

## 🔧 Configuration

### Colab Configuration (`configs/colab_7b.toml`)
```toml
[model.hgnn]
hidden_dim = 512
output_dim = 256
num_layers = 3

[model.text]
model_name = "google/gemma-2-7b"
max_length = 512

[training]
batch_size = 8
learning_rate = 1e-4
num_epochs = 5
```

### A100 Configuration (`configs/colab_7b.toml`)
```toml
[model.hgnn]
hidden_dim = 768
output_dim = 384
num_layers = 4

[model.text]
model_name = "google/gemma-2-9b"
max_length = 1024

[training]
batch_size = 16
learning_rate = 2e-4
num_epochs = 10
gradient_accumulation_steps = 2
```

## 🚀 Usage

### Training Pipeline

1. **Data Download**
```bash
uv run scripts/download_data.py
```

2. **Graph Construction**
```bash
uv run scripts/build_graphs.py --cfg configs/colab_7b.toml
```

3. **HGNN Pretraining**
```bash
uv run scripts/pretrain_hgnn.py --cfg configs/colab_7b.toml
```

4. **Dual-Encoder Alignment**
```bash
uv run scripts/align_dual_encoder.py --cfg configs/colab_7b.toml
```

5. **Fine-tuning**
```bash
uv run scripts/finetune_lora.py --cfg configs/colab_7b.toml
```

6. **Evaluation**
```bash
uv run scripts/evaluate.py --cfg configs/colab_7b.toml
```

### Colab Quick Start

```python
# Install dependencies
!pip install uhg torch transformers accelerate peft bitsandbytes

# Run the complete pipeline
# (See notebooks/colab_prototype.ipynb for full implementation)
```

## 📈 Evaluation

### Standard Tasks
- **Cell Type Classification**: Accuracy, F1-score
- **Tissue Prediction**: Cross-tissue generalization
- **Gene Expression Prediction**: MSE, R²
- **Clustering Quality**: ARI, NMI

### Graph-Sensitive Tasks
- **Ligand-Receptor Link Prediction**: AUROC, AP
- **OOD Generalization**: Leave-one-tissue-out
- **Counterfactual Perturbation**: Drug response prediction

### Cross-Modal Alignment
- **Representation Similarity**: Cosine similarity
- **Retrieval Performance**: Recall@K
- **Visualization Quality**: t-SNE, UMAP

## 🔬 Research Applications

- **Single-Cell Analysis**: Cell type identification, trajectory inference
- **Drug Discovery**: Perturbation response prediction
- **Disease Modeling**: Pathological state classification
- **Biological QA**: Natural language queries about cells

## 📚 Citation

```bibtex
@article{c2s_scale_gemma_hybrid,
  title={C2S-Scale-Gemma Hybrid: Dual-Encoder + Late-Fusion for Single-Cell Transcriptomics},
  author={Bovaird, Zach},
  year={2024},
  license={CC BY-NC-ND 4.0}
}
```

## 📄 License

This project is licensed under CC BY-NC-ND 4.0. See [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/zbovaird/C2S-Scale-Gemma/issues)
- **Discussions**: [GitHub Discussions](https://github.com/zbovaird/C2S-Scale-Gemma/discussions)
- **Email**: zbovaird@example.com

## 🙏 Acknowledgments

- **Google Research**: Original Cell2Sentence work
- **HuggingFace**: Transformers library and model hosting
- **UHG Library**: Hyperbolic geometry operations
- **Scanpy Community**: Single-cell analysis tools

---

**Ready to revolutionize single-cell transcriptomics?** 🚀

[Get started with Colab](https://colab.research.google.com/github/zbovaird/C2S-Scale-Gemma/blob/main/notebooks/colab_prototype.ipynb) | [View on GitHub](https://github.com/zbovaird/C2S-Scale-Gemma)