# C2S-Scale-Gemma Hybrid Model

A dual-encoder + late-fusion system for single-cell transcriptomics that combines **Cell2Sentence / C2S-Scale-Gemma** with **UHG-based graph learning**.

This repository is now being extended toward an **OKSM / OSKM reprogramming** research workflow focused on:

- modeling productive vs alternative reprogramming trajectories
- treating **POU5F1 / SOX2 / KLF4 / MYC** as first-class biological priors
- using hyperbolic structure to separate branching cell-state transitions
- supporting partial / transient reprogramming analyses relevant to **lifespan and longevity**

The branch-level roadmap for that work lives in [`PROJECT_PLAN.md`](PROJECT_PLAN.md).

## Research Direction

The near-term objective is to turn the current hybrid into a more explicit **C2S + UHG platform for OKSM-driven cellular state analysis**.

That means the codebase is evolving in phases:

1. **Stabilize the dual-encoder stack** so scripts and `src/` modules agree.
2. **Add OKSM-aware data priors** such as anchor genes in cell sentences and graph edge reweighting.
3. **Move alignment and metrics closer to the intended geometry** for trajectory and safety analysis.
4. **Add progress and research visuals** that make reprogramming state, risk, and branch structure interpretable.

Current branch work includes:

- restored script-facing compatibility wrappers for training and evaluation
- a committed project roadmap in [`PROJECT_PLAN.md`](PROJECT_PLAN.md)
- lightweight visualization-prep helpers in [`src/eval/reprogramming_visuals.py`](src/eval/reprogramming_visuals.py)
- targeted tests in [`tests/`](tests/)

## 🚀 Quick Start

### Production Colab Notebook (A100 GPU)

1. **Open Production Notebook**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zbovaird/C2S-Scale-Gemma/blob/main/notebooks/c2s_scale_gemma_production.ipynb)

2. **Select A100 GPU**: Runtime → Change runtime type → GPU → A100

3. **Run the complete pipeline**: Real PBMC data + C2S-Scale-Gemma-2-27B model

### Legacy Colab Prototype

1. **Open Legacy Notebook**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zbovaird/C2S-Scale-Gemma/blob/main/notebooks/colab_prototype.ipynb)

2. **Select A100 GPU**: Runtime → Change runtime type → GPU → A100

3. **Run the prototype**: Dummy data + Gemma-9B model

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

### OKSM-Oriented Additions

The branch is adding three specific capabilities for reprogramming work:

- **OSKM gene registry**: central alias handling for `POU5F1`, `SOX2`, `KLF4`, and `MYC`
- **Sentence anchoring**: optional promotion of OSKM genes in Cell2Sentence inputs
- **Graph reweighting**: optional kNN edge upweighting for OSKM-high neighborhoods

### Key Features

- **Hyperbolic Geometry**: UHG operations for hierarchical cell relationships
- **4-bit Quantization**: Efficient memory usage with bitsandbytes
- **Flash Attention**: Faster attention computation
- **Gradient Accumulation**: Effective larger batch sizes
- **A100 Optimizations**: TensorFloat-32, cuDNN benchmarking
- **Real Biological Data**: PBMC dataset with 437 cells and 4 cell types
- **NaN Handling**: Robust processing of sparse single-cell data

## 📊 Performance

### Production Notebook Results
- **Dataset**: Real PBMC data (437 cells, 1,710 genes)
- **Cell Types**: Monocyte (146), T_cell (144), B_cell (144), Dendritic_cell (3)
- **Average Genes per Cell**: 251.9 (realistic for single-cell data)
- **Model**: C2S-Scale-Gemma-2-27B with proper cell sentence formatting
- **Data Quality**: 85.21% NaN values handled correctly
- **Gene Diversity**: Real biological genes (RXRA, OSCAR, CTD-2006K23.1, etc.)

### A100 GPU Performance
- **Model Size**: C2S-Scale-Gemma-2-27B with LoRA adapters
- **Training Time**: ~2-3 hours for 10 epochs
- **Memory Usage**: ~60-70GB GPU memory
- **Throughput**: ~100-150 samples/second
- **Expected Results**: ARI > 0.7, NMI > 0.8

### Model Variants
- **Production (27B)**: `vandijklab/C2S-Scale-Gemma-2-27B` - Full production model
- **Legacy (9B)**: `google/gemma-2-9b` - Prototype model  
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
├── notebooks/             # Colab notebooks
│   ├── c2s_scale_gemma_production.ipynb  # Production notebook
│   └── colab_prototype.ipynb             # Legacy prototype
└── docs/                  # Documentation
```

## 🔧 Configuration

### OKSM / Reprogramming Options

The data, graph, and perturbation-report pipeline now supports lightweight OKSM-aware configuration:

```toml
[knn_graph]
oskm_reweight_enabled = true
oskm_weight_multiplier = 1.5
oskm_score_threshold = 0.0
oskm_species = "human"
```

Dataset-side sentence anchoring and reprogramming heuristics are available through config and `CellSentenceDataset`:

- `top_genes`
- `oskm_anchor_mode` (`"none"` or `"prepend_present"`)
- `oskm_species`

```toml
[reprogramming.references]
somatic_labels = ["fibroblast", "somatic", "starting_state"]
pluripotent_labels = ["esc", "ipsc", "pluripotent", "stem_cell"]

[reprogramming.window_profile]
partial_window_proximity_min = 0.35
partial_window_proximity_max = 0.75
partial_window_max_risk = 0.60
longevity_safe_proximity_max = 0.65
longevity_safe_max_risk = 0.45
min_rejuvenation_score = 0.30

[reprogramming.marker_panels]
rejuvenation = ["SIRT1", "FOXO3", "PPARGC1A", "TFAM", "NFE2L2"]
pluripotency_risk = ["NANOG", "LIN28A", "DPPA4", "UTF1", "PRDM14"]
```

Named dataset-specific profiles live in `configs/reprogramming_profiles.toml`. Current presets include:

- `gse242423_human_fibroblast_oskm`
- `gse176206_mouse_transient_partial`

Geometry-aware alignment is also configurable through the fusion block:

```toml
[fusion]
align_loss = "infonce"
alignment_mode = "euclidean_cosine" # or "projective_distance"
alignment_dim = 256
text_projection_type = "learned"    # or "linear" / "inverse_chordal"
```

`euclidean_cosine` keeps the original projected-vector baseline. `projective_distance` maps text embeddings into geometry space and aligns them against the HGNN's raw hyperbolic branch before radial projection.

### Production Configuration (`configs/colab_7b.toml`)
```toml
[model.hgnn]
hidden_dim = 768
output_dim = 384
num_layers = 4

[model.text]
model_name = "vandijklab/C2S-Scale-Gemma-2-27B"
max_length = 1024

[training]
batch_size = 16
learning_rate = 2e-4
num_epochs = 10
gradient_accumulation_steps = 2
```

### Legacy Configuration (`configs/colab_7b.toml`)
```toml
[model.hgnn]
hidden_dim = 512
output_dim = 256
num_layers = 3

[model.text]
model_name = "google/gemma-2-9b"
max_length = 512

[training]
batch_size = 8
learning_rate = 1e-4
num_epochs = 5
```

## 🚀 Usage

### Production Notebook (Recommended)

The production notebook (`c2s_scale_gemma_production.ipynb`) includes:

1. **Real Data Integration**: PBMC dataset with proper cell sentence formatting
2. **C2S-Scale-Gemma Model**: Actual model from HuggingFace
3. **NaN Handling**: Robust processing of sparse expression data
4. **UHG-HGNN Encoder**: Complete hyperbolic graph neural network
5. **Hybrid Pipeline**: Graph + Text → Fusion architecture
6. **Cell Type Prediction**: Working with real biological data

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

### OSKM Counterfactual Workflow

Generate an in silico perturbation dataset:

```bash
uv run scripts/perturb_oskm_expression.py \
  --data-path data/raw/reprogramming.h5ad \
  --mode overexpress \
  --factor 2.0 \
  --output-dir artifacts/oskm_perturbation
```

Compare baseline vs perturbed cells in the learned representation space:

```bash
uv run scripts/compare_oskm_perturbation_embeddings.py \
  --config configs/colab_7b.toml \
  --dataset-profile gse242423_human_fibroblast_oskm \
  --checkpoint-path artifacts/align_dual_encoder/final_model.pt \
  --baseline-data-path data/raw/reprogramming.h5ad \
  --perturbed-data-path artifacts/oskm_perturbation/oskm_overexpress.h5ad \
  --output-dir artifacts/oskm_embedding_comparison
```

This exports:

- baseline and perturbed text / graph / fused embedding arrays
- `embedding_shift_summary.json`
- `fused_embedding_shift_frame.json`
- `reprogramming_overlay_summary.json`

The overlay summary now also records the selected dataset profile manifest, resolved reference labels, and the active heuristic window profile.

Generate a static report with plots:

```bash
uv run scripts/generate_oskm_perturbation_report.py \
  --comparison-dir artifacts/oskm_embedding_comparison \
  --perturbation-summary artifacts/oskm_perturbation/oskm_overexpress_summary.json
```

Add alignment ablations by passing extra comparison runs:

```bash
uv run scripts/generate_oskm_perturbation_report.py \
  --comparison-dir artifacts/oskm_embedding_comparison_projective \
  --ablation-comparison-dir artifacts/oskm_embedding_comparison_euclidean
```

Or generate both comparison runs in one command when you already have separate Euclidean and projective checkpoints:

```bash
uv run scripts/run_alignment_ablation.py \
  --baseline-data-path data/raw/reprogramming.h5ad \
  --perturbed-data-path artifacts/oskm_perturbation/oskm_overexpress.h5ad \
  --dataset-profile gse242423_human_fibroblast_oskm \
  --euclidean-config configs/colab_7b.toml \
  --euclidean-checkpoint artifacts/euclidean/final_model.pt \
  --projective-config configs/colab_7b.toml \
  --projective-checkpoint artifacts/projective/final_model.pt \
  --output-root artifacts/alignment_ablation
```

That writes `euclidean/`, `projective/`, and `ablation_manifest.json` under the output root. The report script can consume that manifest directly:

```bash
uv run scripts/generate_oskm_perturbation_report.py \
  --comparison-dir artifacts/alignment_ablation/projective \
  --ablation-manifest artifacts/alignment_ablation/ablation_manifest.json
```

For named study validation, use the validation-bundle runner on top of those paired ablations:

```bash
uv run scripts/run_validation_bundle.py \
  --track human_fibroblast_oskm \
  --baseline-data-path data/raw/GSE242423.h5ad \
  --perturbed-data-path artifacts/GSE242423_oskm_perturbed.h5ad \
  --euclidean-config configs/colab_7b.toml \
  --euclidean-checkpoint artifacts/euclidean/final_model.pt \
  --projective-config configs/colab_7b.toml \
  --projective-checkpoint artifacts/projective/final_model.pt \
  --output-root artifacts/validation_bundle
```

This writes a track-specific `validation_bundle.json` plus the paired ablation outputs for the named study. The initial validation-track registry lives in `configs/validation_tracks.toml`.

Summarize a finished validation bundle into a compact benchmark scorecard:

```bash
uv run scripts/summarize_validation_bundle.py \
  --validation-manifest artifacts/validation_bundle/human_fibroblast_oskm/validation_bundle.json
```

This produces `validation_benchmark_summary.json` and `VALIDATION_BENCHMARK.md` for the named study.

When the study track declares a `timepoint_column`, the validation benchmark also includes per-timepoint progression summaries so Euclidean and projective runs can be compared by stage, not just by aggregate shift/safety metrics.
The validation-track registry also carries recommendation thresholds, so the benchmark summary can emit a track-specific judgment such as `prefer_projective`, `mixed`, or `prefer_euclidean`.
Those recommendations now include supporting and concerning timepoints so it is easier to audit which stages of the trajectory are driving the conclusion.

Generate validation trajectory plots from the benchmark summary:

```bash
uv run scripts/plot_validation_bundle.py \
  --summary-path artifacts/validation_bundle/human_fibroblast_oskm/validation_benchmark_summary.json
```

This produces:

- `validation_timepoint_progress_delta.png`
- `validation_timepoint_safe_fraction.png`
- `validation_timepoint_safe_delta.png`

Export a structured explorer payload from the same benchmark summary:

```bash
uv run scripts/export_validation_explorer.py \
  --summary-path artifacts/validation_bundle/human_fibroblast_oskm/validation_benchmark_summary.json
```

This writes `validation_explorer_payload.json`, which packages the run table, per-timepoint summaries, chart-ready trajectory series, delta rows, and recommendation evidence for lightweight interactive views or notebook dashboards.

Render a self-contained HTML explorer from either the benchmark summary or the exported payload:

```bash
uv run scripts/render_validation_explorer.py \
  --summary-path artifacts/validation_bundle/human_fibroblast_oskm/validation_benchmark_summary.json
```

This writes `validation_explorer.html`, which turns the bundle into a directly viewable trajectory dashboard with overview cards, run tables, chart panels, and auditable recommendation evidence.

Export a cell-level trajectory dataset for notebook analysis or publication-style plots:

```bash
uv run scripts/export_validation_trajectory_dataset.py \
  --validation-manifest artifacts/validation_bundle/human_fibroblast_oskm/validation_bundle.json
```

This writes `validation_trajectory_dataset.json`, which includes per-run cell rows, timepoint/branch cohorts, and per-cell projective-vs-Euclidean deltas for richer downstream trajectory analyses.

Export 2D trajectory projections from the saved fused embeddings:

```bash
uv run scripts/export_validation_trajectory_projection.py \
  --validation-manifest artifacts/validation_bundle/human_fibroblast_oskm/validation_bundle.json
```

This writes `validation_trajectory_projection.json`, which adds scatter/arrow-ready baseline and perturbed coordinates for each cell in a shared 2D PCA space per run.

Render publication-style projection plots from that trajectory projection artifact:

```bash
uv run scripts/plot_validation_trajectory_projection.py \
  --projection-path artifacts/validation_bundle/human_fibroblast_oskm/validation_trajectory_projection.json
```

This produces branch-colored and safe-zone-colored scatter/arrow plots for each alignment run so the stage-wise reprogramming trajectories can be reviewed visually.

Render a lightweight browser viewer from the same projection artifact:

```bash
uv run scripts/render_validation_trajectory_projection.py \
  --projection-path artifacts/validation_bundle/human_fibroblast_oskm/validation_trajectory_projection.json
```

This writes `validation_trajectory_projection.html`, which lets you switch runs and recolor the projected trajectories by branch, safe zone, or timepoint in-browser.

Export the main validation summary, explorer, and trajectory artifacts in one pass:

```bash
uv run scripts/export_validation_bundle_artifacts.py \
  --validation-manifest artifacts/validation_bundle/human_fibroblast_oskm/validation_bundle.json
```

This writes a consolidated artifact bundle including the benchmark summary, explorer payload and HTML, trajectory dataset, trajectory projection, projection HTML, and a small manifest of generated files.
The benchmark summary and explorer artifacts include interpretation-limit notes so projection views, heuristic safe-window calls, and alignment recommendations are kept separate from biological or in vivo safety claims.
Run rows also include the geometry distance backend and graph embedding source, making it visible when a projective alignment run used UHG distance versus a Euclidean fallback and whether alignment consumed hyperbolic or projected graph embeddings.

Before a real validation run, generate a readiness report for the configured validation tracks and dataset profiles:

```bash
uv run scripts/report_validation_readiness.py \
  --output-path artifacts/validation_readiness.json
```

This flags each validation track as `ready`, `needs_data`, or `incomplete_metadata` based on profile metadata and local baseline/perturbed data availability.

Audit profile and recommendation thresholds before treating validation outputs as calibrated:

```bash
uv run scripts/report_validation_calibration.py \
  --output-path artifacts/validation_calibration.json
```

This checks heuristic window profiles and track recommendation thresholds for missing values, out-of-range values, and inconsistent ordering.

The bundle runner also performs preflight checks for the selected track, input datasets, configs, checkpoints, and dataset-profile registry. To run those checks without model execution:

```bash
uv run scripts/run_validation_bundle.py \
  --track human_fibroblast_oskm \
  --baseline-data-path data/raw/GSE242423.h5ad \
  --perturbed-data-path artifacts/GSE242423_oskm_perturbed.h5ad \
  --euclidean-config configs/colab_7b.toml \
  --euclidean-checkpoint artifacts/euclidean/final_model.pt \
  --projective-config configs/colab_7b.toml \
  --projective-checkpoint artifacts/projective/final_model.pt \
  --preflight-only
```

After exporting the consolidated artifacts, run QA checks against the generated artifact manifest:

```bash
uv run scripts/qa_validation_artifacts.py \
  --artifact-manifest artifacts/validation_bundle/human_fibroblast_oskm/validation_artifacts_manifest.json
```

To keep real-run review ordered and auditable, generate a validation review protocol:

```bash
uv run scripts/build_validation_review_protocol.py \
  --track human_fibroblast_oskm \
  --validation-manifest artifacts/validation_bundle/human_fibroblast_oskm/validation_bundle.json \
  --output-root artifacts/validation_bundle/human_fibroblast_oskm
```

Before refactoring the HGNN stack toward manifold-native operations, audit the current geometry path:

```bash
uv run scripts/report_manifold_readiness.py \
  --output-path artifacts/manifold_readiness.json
```

Convert the readiness findings into an ordered implementation plan:

```bash
uv run scripts/build_manifold_refactor_plan.py \
  --readiness-report artifacts/manifold_readiness.json \
  --output-path artifacts/manifold_refactor_plan.json
```

The UHG encoder projection path now uses explicit `TangentSpaceLinear` adapters for remaining Euclidean linear maps, making tangent-space boundaries visible before deeper manifold-native layer refactors.

This produces:

- `shift_histogram.png`
- `oskm_score_vs_shift.png`
- `shift_by_cell_type.png`
- `risk_by_branch.png`
- `progress_vs_risk.png`
- `zone_counts.png`
- `marker_panel_balance.png`
- `alignment_ablation.png`
- `alignment_ablation_safety.png`
- `OSKM_PERTURBATION_REPORT.md`

## 🧪 Testing

Targeted CPU-friendly tests are included for the compatibility and OKSM groundwork:

```bash
pytest tests
```

The current tests focus on:

- trainer and script-facing compatibility layers
- lightweight alignment-loss behavior
- visualization-prep helpers
- Phase B utilities such as OSKM-aware data and graph helpers
- OSKM perturbation and embedding-shift comparison helpers
- perturbation report helpers

### Colab Quick Start

```python
# Install dependencies
!pip install uhg torch transformers accelerate peft bitsandbytes scanpy

# Run the complete pipeline
# (See notebooks/c2s_scale_gemma_production.ipynb for full implementation)
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

[Get started with Production Colab](https://colab.research.google.com/github/zbovaird/C2S-Scale-Gemma/blob/main/notebooks/c2s_scale_gemma_production.ipynb) | [View on GitHub](https://github.com/zbovaird/C2S-Scale-Gemma)