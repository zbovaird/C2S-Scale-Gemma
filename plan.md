<!-- 3d03dcc2-d50b-4acf-ba21-8bf938554994 e2361f96-bebd-4109-9e13-085b08354f9b -->
# C2S-Scale-Gemma Hybrid Implementation Plan

## Phase 1: Repository & Environment Setup

### 1.1 Initialize Repository Structure

Create complete directory structure:

- `/src` with submodules: `uhg_adapters/`, `graphs/`, `hgnn/`, `text/`, `fusion/`, `data/`, `eval/`, `utils/`
- `/scripts` for executable workflows
- `/configs` for TOML configuration files
- `/notebooks` for Colab prototyping
- Include `.gitignore` to exclude secrets, data files, checkpoints, and `__pycache__`
- Include `.env.example` template (no actual secrets)

### 1.2 Dependency Management with uv

Create `pyproject.toml` with:

- Core ML: `torch>=2.3`, `transformers>=4.43`, `accelerate>=1.1`, `bitsandbytes>=0.43`, `peft>=0.11`
- UHG library: `uhg` (will investigate API after install)
- Single-cell: `scanpy>=1.9`, `anndata`, `umap-learn`, `pynndescent`
- Graph processing: `networkx`, `pandas`, `numpy`, `scipy`, `scikit-learn`
- Training infra: `mlflow`, `omegaconf`, `wandb`, `tqdm`, `pyyaml`
- Development: `pytest`, `black`, `ruff`

Run `uv lock` and `uv sync` to generate lock file.

### 1.3 Configuration Files

Create TOML configs:

- `configs/colab_7b.toml`: Gemma-7B, 4-bit quantization, LoRA r=16, batch_size=8
- `configs/vertex_27b.toml`: Gemma-27B, GCS data paths, larger hidden dims
- `configs/datasets.toml`: Dataset paths, graph parameters
- `configs/ablations.toml`: Ablation study configurations

## Phase 2: Data Pipeline & Graph Construction

### 2.1 Data Acquisition Script

Create `scripts/download_data.py`:

- Use HuggingFace `datasets` library to fetch Cell2Sentence dataset
- API call: `load_dataset("vandijklab/cell2sentence")` or similar available dataset
- Alternative: Clone Cell2Sentence GitHub repo and use their data loading utilities
- Store in `data/raw/` directory
- No manual downloads required for Colab integration

### 2.2 Data Preprocessing

Implement `src/data/dataset.py`:

- Load `.h5ad` files using `scanpy`/`anndata`
- Convert expression data to "cell sentences" (rank-ordered gene names by expression)
- Cache tokenized text inputs using HF datasets cache
- Create paired views mapping text ↔ graph neighborhoods

Implement `src/data/collate.py`:

- Neighborhood batching with size cap (64-128 nodes)
- UHG radial-band sampling to preserve hierarchical structure

### 2.3 Graph Construction

Implement graph builders in `src/graphs/`:

**`build_knn.py`**:

- Cell-cell kNN (k=15-50) on latent embeddings from scanpy/SCVI
- Output: `data/processed/graphs/knn.parquet`

**`build_lr_bipartite.py`**:

- Ligand-receptor edges from curated tables
- Output: `data/processed/graphs/lr.parquet`

**`build_grn.py`**:

- Gene regulatory network from SCENIC/GRNBoost outputs
- Output: `data/processed/graphs/grn.parquet`

**`utils.py`**:

- Graph utility functions: normalization, sampling, validation

Create executable: `scripts/build_graphs.py --cfg configs/datasets.toml`

## Phase 3: UHG-HGNN Encoder

### 3.1 Investigate UHG Library

After `uv sync`, programmatically explore the UHG library (custom library without extensive docs):

- Install via pip and import all modules/functions
- Inspect source code, docstrings, and function signatures
- Identify available primitives for:
- Hyperbolic distance computations
- Aggregation operations in hyperbolic space
- Normalization functions
- Projective barycenter for pooling
- Any existing GNN layer implementations
- Document findings in `docs/uhg_api.md` with function signatures and usage examples
- If primitives are missing, implement custom functions using base UHG mathematics
- **Add unit tests** for all UHG operations to verify correctness

### 3.2 UHG Adapters

Implement `src/uhg_adapters/radial_projector.py`:

- Monotone radial projector: UHG (ℍ^d) → Euclidean (ℝ^d)
- Preserve radial order (monotonic in UHG radius)
- Maintain angular information (normalized chordal map)
- Unit tests for monotonicity and angular preservation bounds

Implement `src/uhg_adapters/sampling.py`:

- Radial-band sampling curriculum for neighborhood selection
- Progressive sampling from inner to outer bands

Implement `src/uhg_adapters/euclid_to_uhg.py`:

- Inverse mapping (if needed for experiments)

### 3.3 HGNN Layers & Encoder

Implement `src/hgnn/layers.py`:

- UHG-GraphSAGE or UHG-GIN layers (configurable)
- All operations use UHG primitives (no manual exp/log maps)
- Hyperbolic distance, aggregation, normalization via UHG library

Implement `src/hgnn/encoder.py`:

- Multi-layer HGNN encoder (3 layers default)
- Node-level embeddings: z_h_node ∈ ℍ^d
- Pooled neighborhood embeddings: z_h ∈ ℍ^d (projective barycenter)
- Hidden dim: 256 (7B) / 384 (27B)

Implement `src/hgnn/losses.py`:

- UHG contrastive loss for self-supervision
- Positive pairs: stochastic augmentations (edge-drop, node-drop, feature-mask)
- Hard negatives: same tissue, different cells
- Temperature τ=0.07

Create script: `scripts/pretrain_hgnn.py --cfg configs/colab_7b.toml`

## Phase 4: Text Encoder (C2S-Scale-Gemma)

### 4.1 Gemma Model Loading

Implement `src/text/gemma_loader.py`:

- Load from HuggingFace: `vandijklab/C2S-Scale-Gemma-2-7B` (Colab) or `vandijklab/C2S-Scale-Gemma-2-27B` (Vertex)
- 4-bit quantization: `load_in_4bit=True`, `bnb_4bit_compute_dtype="bfloat16"`
- Enable gradient checkpointing
- Freeze base model (unfreeze only norms if needed)

### 4.2 LoRA Adapters

Implement `src/text/adapters.py`:

- PEFT/LoRA injection (rank 8-16, alpha 16)
- Target modules: mid-to-late attention blocks
- Dropout: 0.05

### 4.3 Pooling

Implement `src/text/pooling.py`:

- Extract final hidden state
- CLS-like token pooling or mean pooling over "cell sentence" tokens
- Output: z_t ∈ ℝ^d_t (match Gemma hidden dim: 2048 for 7B, 4096 for 27B)

## Phase 5: Alignment & Fusion

### 5.1 Alignment Loss

Implement `src/fusion/align_losses.py`:

- InfoNCE contrastive loss with temperature τ
- Hard negatives sampling: same tissue/adjacent lineages
- In-batch negatives for efficiency
- Project z_h → z_h_e (Euclidean) via radial projector before computing loss

### 5.2 Fusion Heads

Implement `src/fusion/heads.py`:

- Small MLP head: concatenate [z_t ; z_h_e] → adapter input
- Inject into Gemma via LoRA at mid-to-late blocks
- Optional: attention-based fusion mechanism

### 5.3 Fusion Training Loop

Implement `src/fusion/trainer.py`:

- Dual-encoder alignment phase: train HGNN + projector + text encoder LoRA
- Loss: InfoNCE + task-specific losses
- Temperature sweep experiments (τ ∈ [0.05, 0.07, 0.1])
- Optimizer: AdamW, lr=2e-4 (7B) / 1.5e-4 (27B)
- Mixed precision bf16, gradient accumulation

Create scripts:

- `scripts/align_dual_encoder.py --cfg configs/colab_7b.toml`
- `scripts/finetune_lora.py --cfg configs/colab_7b.toml`

## Phase 6: Evaluation Suite

### 6.1 Standard C2S Tasks

Implement `src/eval/tasks.py`:

- Cell type prediction (classification)
- Tissue prediction (classification)
- Cluster captioning (generation)
- Perturbation reasoning (QA)
- Biological QA (generation)

### 6.2 Graph-Sensitive Tasks

Implement `src/eval/graph_tasks.py`:

- Ligand-receptor link prediction (AUROC/AP)
- OOD generalization: leave-one-tissue-out
- Counterfactual perturbation: held-out drugs/targets

### 6.3 Metrics & Logging

Implement `src/eval/metrics.py`:

- Accuracy, F1, AUROC, AP
- Recall@K for alignment quality
- Text generation metrics (BLEU, ROUGE if applicable)

Implement `src/utils/logging.py`:

- MLflow experiment tracking
- Log metrics, hyperparameters, artifacts
- Vertex AI Model Monitoring hooks (for production)

Create script: `scripts/evaluate.py --cfg configs/colab_7b.toml`

## Phase 7: Colab Prototype (7B)

### 7.1 Colab Notebook

Create `notebooks/colab_prototype.ipynb`:

- Mount GDrive or GCS for data storage
- `uv sync` to install dependencies
- Run full pipeline:

1. Download & preprocess data
2. Build graphs
3. Pretrain HGNN (UHG contrastive)
4. Load Gemma-7B (4-bit) + LoRA
5. Align encoders (InfoNCE, τ sweep)
6. Finetune on tasks
7. Evaluate + ablations
8. Export artifacts

### 7.2 Exit Criteria

Verify before Vertex scale-up:

- Hybrid beats text-only on ≥2 graph-sensitive tasks
- No training instabilities across 3 seeds (13, 17, 23)
- OOD tissue gap ≤10% of in-domain performance

### 7.3 Artifact Export

Create `scripts/export_artifacts.py`:

- Save HGNN weights, LoRA adapters, radial projector
- Package for Vertex AI deployment
- Output to `gs://<bucket>/artifacts/` or local export

## Phase 8: Vertex AI Scale-Up (27B)

### 8.1 Container Setup

Create Vertex AI custom training image:

- Base: NVIDIA PyTorch image or Vertex PyTorch image
- Install: uv, UHG lib, HF/PEFT/bitsandbytes
- Entrypoint: `entrypoint.sh` calls `uv run` with config path
- Push to Artifact Registry

### 8.2 Configuration

Update `configs/vertex_27b.toml`:

- Model: `vandijklab/C2S-Scale-Gemma-2-27B`
- Data paths: `gs://<bucket>/cells.h5ad`, graphs, pairs
- Larger hidden dims: HGNN 384, projector 4096
- Batch size 8, accumulation 8

### 8.3 Secrets & Environment

- Store HF tokens in Vertex Secrets
- Set GCS bucket paths in environment variables
- MLflow tracking URI (Vertex AI Experiments)

### 8.4 Training Jobs

Create Vertex CustomJob specs:

- `vertex/customjob_pretrain_hgnn_27b.yaml`
- `vertex/customjob_align_27b.yaml`
- `vertex/customjob_finetune_27b.yaml`
- `vertex/customjob_evaluate_27b.yaml`

Compute: A3 H100-80GB (1-2 GPUs), or 2-4× H100 for faster training

### 8.5 Serving (Optional)

Package for inference:

- Pre-compute HGNN embeddings offline for common cells
- Serve HGNN as lightweight microservice
- Vertex Prediction endpoint for Gemma + LoRA + fusion heads

## Phase 9: Documentation & Testing

### 9.1 README

Create comprehensive `README.md`:

- Project overview and architecture
- Installation instructions (uv setup)
- Quick start guide (Colab link)
- Vertex AI deployment instructions
- Citation and license (CC BY-NC-ND 4.0)

### 9.2 Testing

Implement unit tests:

- `tests/test_uhg_adapters.py`: radial projector monotonicity, angular preservation
- `tests/test_hgnn.py`: layer forward passes, encoder output shapes
- `tests/test_text.py`: Gemma loading, LoRA injection, pooling
- `tests/test_fusion.py`: alignment loss, fusion heads
- `tests/test_graphs.py`: graph construction, sampling

### 9.3 Security & Compliance

- Add `.env.example` with placeholder keys
- `.gitignore`: exclude `.env`, data files, checkpoints, logs
- Document data privacy requirements in `SECURITY.md`
- Pin all library versions in `pyproject.toml`
- Record git SHA in MLflow tags for reproducibility

## Phase 10: Cancer Research & Real Data Integration

### 10.1 Cancer Dataset Acquisition

Implement cancer-specific data pipeline:

**TCGA Integration**:
- Download TCGA single-cell RNA-seq data
- Cancer type annotations (breast, lung, colon, etc.)
- Patient survival outcomes and clinical metadata
- Drug response data where available

**CellxGene Cancer Collections**:
- Curated cancer datasets from CellxGene
- Tumor microenvironment data
- Immune cell infiltration patterns
- Metastasis vs. primary tumor comparisons

**CCLE Integration**:
- Cancer cell line encyclopedia
- Drug sensitivity data (IC50 values)
- Genetic mutation profiles
- Drug-target interaction networks

Create `scripts/download_cancer_data.py`:
```bash
uv run scripts/download_cancer_data.py --dataset tcga
uv run scripts/download_cancer_data.py --dataset cellxgene_cancer
uv run scripts/download_cancer_data.py --dataset ccle
```

### 10.2 Cancer-Specific Graph Construction

Implement `src/graphs/build_cancer_graphs.py`:

**Drug-Target Networks**:
- Drug-target interaction graphs from DrugBank/ChEMBL
- Target protein-protein interaction networks
- Drug-drug interaction networks
- Output: `data/processed/graphs/drug_target.parquet`

**Cancer Signaling Pathways**:
- KEGG cancer pathways (PI3K-AKT, MAPK, etc.)
- Reactome cancer pathways
- Custom cancer-specific pathway annotations
- Output: `data/processed/graphs/cancer_pathways.parquet`

**Tumor Microenvironment**:
- Cell-cell interaction networks in tumors
- Immune cell-cancer cell interactions
- Stromal cell-cancer cell interactions
- Spatial proximity graphs from spatial transcriptomics
- Output: `data/processed/graphs/tumor_microenvironment.parquet`

**Metastasis Networks**:
- Cell migration and invasion pathways
- Epithelial-mesenchymal transition (EMT) networks
- Metastasis-specific gene regulatory networks
- Output: `data/processed/graphs/metastasis.parquet`

### 10.3 Cancer-Specific Cell Sentence Format

Enhance `src/data/dataset.py` for cancer data:

**Cancer Gene Signatures**:
- Include oncogenes (MYC, KRAS, TP53, etc.)
- Tumor suppressor genes (BRCA1, BRCA2, etc.)
- Cancer-specific markers (HER2, EGFR, etc.)
- Drug resistance markers

**Mutation Status Integration**:
- Add mutation status to cell sentences
- Include copy number variation data
- Chromosomal instability markers
- Epigenetic modification patterns

**Clinical Context**:
- Patient age, gender, stage
- Treatment history
- Survival outcomes
- Response to therapy

### 10.4 Cancer Research Tasks

Implement `src/eval/cancer_tasks.py`:

**Drug Response Prediction**:
- Predict IC50 values for cancer cell lines
- Binary drug sensitivity classification
- Multi-drug response prediction
- Drug combination synergy prediction

**Cancer Type Classification**:
- Multi-cancer type identification
- Cancer subtype classification
- Primary vs. metastatic classification
- Cancer grade prediction

**Prognosis Prediction**:
- Overall survival prediction
- Disease-free survival prediction
- Risk stratification
- Treatment response prediction

**Biomarker Discovery**:
- Cancer-specific gene signatures
- Drug resistance biomarkers
- Prognostic biomarkers
- Predictive biomarkers for therapy

**Drug Discovery**:
- Novel drug-target interaction prediction
- Drug repurposing for cancer
- Drug combination optimization
- Adverse effect prediction

### 10.5 Cancer-Specific UHG-HGNN Enhancements

Enhance `src/hgnn/cancer_encoder.py`:

**Hierarchical Cancer Taxonomy**:
- Tumor → Tissue → Cell type hierarchy
- Cancer progression stages
- Metastasis pathways
- Drug resistance evolution

**Temporal Graph Evolution**:
- Cancer progression over time
- Drug resistance development
- Treatment response evolution
- Metastasis formation

**Spatial-Temporal Graphs**:
- Tumor growth patterns
- Metastasis spread patterns
- Immune infiltration dynamics
- Drug penetration patterns

### 10.6 Cancer-Specific Text Encoder Enhancements

Enhance `src/text/cancer_gemma_loader.py`:

**Cancer-Specific Prompts**:
- Include clinical context in prompts
- Add cancer stage and grade information
- Include treatment history
- Add patient demographics

**Drug-Gene Relationships**:
- Incorporate drug databases (DrugBank, ChEMBL)
- Include drug mechanism of action
- Add drug side effects
- Include drug-drug interactions

**Pathway Information**:
- Include biological pathway context
- Add cancer signaling pathways
- Include metabolic pathways
- Add immune response pathways

### 10.7 Cancer Research Evaluation Metrics

Implement `src/eval/cancer_metrics.py`:

**Clinical Relevance**:
- Correlation with patient survival
- Concordance with clinical outcomes
- Biomarker validation metrics
- Clinical utility assessment

**Drug Response Metrics**:
- AUROC for drug sensitivity
- Concordance index for survival
- Drug response correlation
- Combination therapy synergy

**Biomarker Quality**:
- Gene set enrichment analysis (GSEA)
- Pathway enrichment analysis
- Biomarker stability across datasets
- Cross-validation performance

**Interpretability**:
- Attention visualization for drug targets
- Gene importance ranking
- Pathway contribution analysis
- Clinical decision support metrics

### 10.8 Cancer Research Configuration

Create `configs/cancer_research.toml`:

```toml
[data]
cancer_datasets = ["tcga", "cellxgene_cancer", "ccle"]
drug_databases = ["drugbank", "chembl", "kegg"]
pathway_databases = ["kegg", "reactome", "biocarta"]

[model.cancer_hgnn]
hierarchical_taxonomy = true
temporal_evolution = true
spatial_temporal = true
cancer_specific_genes = true

[model.cancer_text]
clinical_context = true
drug_gene_relationships = true
pathway_information = true
cancer_specific_prompts = true

[evaluation.cancer_tasks]
drug_response_prediction = true
cancer_type_classification = true
prognosis_prediction = true
biomarker_discovery = true
drug_discovery = true

[evaluation.metrics]
clinical_relevance = true
drug_response_metrics = true
biomarker_quality = true
interpretability = true
```

### 10.9 Cancer Research Workflow

Create `scripts/cancer_research_pipeline.py`:

```bash
# Complete cancer research workflow
uv run scripts/cancer_research_pipeline.py --cfg configs/cancer_research.toml

# Individual cancer tasks
uv run scripts/evaluate_cancer_tasks.py --task drug_response --cfg configs/cancer_research.toml
uv run scripts/evaluate_cancer_tasks.py --task cancer_classification --cfg configs/cancer_research.toml
uv run scripts/evaluate_cancer_tasks.py --task prognosis_prediction --cfg configs/cancer_research.toml
uv run scripts/evaluate_cancer_tasks.py --task biomarker_discovery --cfg configs/cancer_research.toml
```

### 10.10 Vertex AI Cancer Research Deployment

Update `configs/vertex_cancer_27b.toml`:

```toml
[model.text]
model_name = "vandijklab/C2S-Scale-Gemma-2-27B"
cancer_specific_training = true
clinical_context_integration = true

[data]
cancer_datasets = ["tcga", "cellxgene_cancer", "ccle"]
drug_databases = ["drugbank", "chembl"]
pathway_databases = ["kegg", "reactome"]

[training]
cancer_specific_losses = true
drug_response_weight = 0.3
prognosis_weight = 0.2
biomarker_weight = 0.2
classification_weight = 0.3

[deployment]
cancer_research_endpoint = true
drug_discovery_api = true
biomarker_discovery_api = true
clinical_decision_support = true
```

### 10.11 Expected Cancer Research Improvements

**Over Text-Only Models**:
- Better drug response prediction through graph-aware understanding
- Improved cancer classification with hierarchical taxonomy
- Enhanced biomarker discovery through attention mechanisms
- More accurate prognosis prediction with temporal evolution

**Over Google's C2S-Scale-Gemma**:
- Graph-aware drug-target interactions
- Hierarchical cancer taxonomy understanding
- Temporal cancer progression modeling
- Interpretable attention for clinical decision support

**Success Metrics**:
- Drug Response AUROC > 0.8
- Cancer Classification Accuracy > 0.9
- Biomarker Discovery: Significant enrichment in known cancer pathways
- Clinical Relevance: Correlation with patient survival outcomes
- Interpretability: Clinically meaningful attention patterns

## Key Files Reference

**Critical implementations:**

- `src/uhg_adapters/radial_projector.py` - UHG→Euclidean monotone projector
- `src/hgnn/encoder.py` - HGNN encoder with UHG ops
- `src/text/gemma_loader.py` - Load C2S-Scale-Gemma with 4-bit + LoRA
- `src/fusion/align_losses.py` - InfoNCE contrastive alignment
- `src/fusion/trainer.py` - Dual-encoder training loop
- `scripts/build_graphs.py` - Graph construction pipeline
- `scripts/align_dual_encoder.py` - Alignment training
- `notebooks/colab_prototype.ipynb` - End-to-end Colab workflow

**Cancer Research Extensions:**

- `src/graphs/build_cancer_graphs.py` - Cancer-specific graph construction
- `src/eval/cancer_tasks.py` - Cancer research evaluation tasks
- `src/hgnn/cancer_encoder.py` - Cancer-specific UHG-HGNN enhancements
- `src/text/cancer_gemma_loader.py` - Cancer-specific text encoder
- `scripts/cancer_research_pipeline.py` - Complete cancer research workflow
- `configs/cancer_research.toml` - Cancer research configuration

**Configuration:**

- `configs/colab_7b.toml` - Colab prototype settings
- `configs/vertex_27b.toml` - Vertex AI production settings
- `configs/cancer_research.toml` - Cancer research settings
- `configs/datasets.toml` - Data and graph parameters

**Entry points:**

- `scripts/download_data.py` - Fetch Cell2Sentence data from HF
- `scripts/download_cancer_data.py` - Fetch cancer datasets
- `scripts/build_graphs.py` - Construct kNN, L-R, GRN graphs
- `scripts/build_cancer_graphs.py` - Construct cancer-specific graphs
- `scripts/pretrain_hgnn.py` - HGNN self-supervised pretraining
- `scripts/align_dual_encoder.py` - Text-graph alignment
- `scripts/finetune_lora.py` - Task-specific fine-tuning
- `scripts/evaluate.py` - Full evaluation suite
- `scripts/cancer_research_pipeline.py` - Cancer research workflow
- `scripts/export_artifacts.py` - Package for deployment

### To-dos

- [x] Initialize repository structure with all directories, .gitignore, and .env.example
- [x] Create pyproject.toml with all dependencies, run uv lock and uv sync
- [x] Create TOML config files for colab_7b, vertex_27b, datasets, and ablations
- [x] Install and explore UHG library API, document available functions
- [x] Implement scripts/download_data.py to fetch Cell2Sentence dataset from HuggingFace
- [x] Implement src/data/dataset.py and src/data/collate.py for data loading and batching
- [x] Implement graph builders (kNN, L-R, GRN) in src/graphs/ and scripts/build_graphs.py
- [x] Implement radial projector, sampling, and optional inverse mapping in src/uhg_adapters/
- [x] Implement HGNN layers, encoder, and losses in src/hgnn/
- [x] Implement scripts/pretrain_hgnn.py for self-supervised HGNN training
- [x] Implement Gemma loader, LoRA adapters, and pooling in src/text/
- [x] Implement alignment losses, fusion heads, and trainer in src/fusion/
- [x] Implement scripts/align_dual_encoder.py and scripts/finetune_lora.py
- [x] Implement evaluation tasks, graph tasks, and metrics in src/eval/
- [x] Create notebooks/colab_prototype.ipynb with full end-to-end pipeline
- [x] Create comprehensive README.md, plan.md, and SECURITY.md
- [x] Push complete implementation to GitHub repository
- [ ] Implement unit tests for all core components in tests/
- [ ] Create Vertex AI container image, CustomJob specs, and deployment configs
- [ ] **NEW: Implement cancer dataset acquisition and integration**
- [ ] **NEW: Implement cancer-specific graph construction (drug-target, pathways, TME)**
- [ ] **NEW: Implement cancer research tasks (drug response, classification, prognosis)**
- [ ] **NEW: Enhance UHG-HGNN for cancer-specific hierarchical taxonomy**
- [ ] **NEW: Enhance text encoder for cancer-specific prompts and clinical context**
- [ ] **NEW: Implement cancer research evaluation metrics and clinical validation**
- [ ] **NEW: Deploy cancer research pipeline on Vertex AI with 27B model**
- [ ] **NEW: Compare performance with Google's C2S-Scale-Gemma on cancer tasks**
