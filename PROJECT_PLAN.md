# Project Plan: C2S-Scale-Gemma + Yamanaka Trajectory

This document is the **reprogramming / trajectory roadmap** for evolving the C2S-Scale-Gemma hybrid toward modeling cellular reprogramming with hyperbolic geometry. It complements the broader hybrid implementation notes in [`plan.md`](plan.md).

---

## Objective

Transform the C2S-Scale-Gemma hybrid architecture into a specialized tool for modeling **cellular reprogramming**. Use **Unified Hyperbolic Geometry (UHG)** to represent the non-Euclidean “climb” from differentiated somatic cells back toward **pluripotency**, with Yamanaka factors (OSKM) as primary structural signals.

---

## Phase 1: UHG Infrastructure and Manifold Setup

- **Lorentzian manifold:** Shift from purely Euclidean graph layers to UHG **Lorentz** or **Poincaré** manifold operations where the model stack allows it. These spaces are a good fit for exponential branching in differentiation hierarchies.
- **Hierarchical scaling:** Make hyperbolic curvature ($\kappa$) a **learnable** parameter so the space can stretch as trajectories move toward a stem-cell “root.”
- **Primary code touchpoint (intent):** evolve the HGNN stack to rely consistently on UHG manifold operations (e.g. hyperbolic linear / distance), not ad hoc Euclidean layers in the hyperbolic pathway.

---

## Phase 2: Yamanaka Factor Feature Engineering

- **Token prioritization:** Extend the Cell2Sentence / tokenizer path so **Yamanaka factors (OSKM)** behave as **anchor tokens** when building text views of cells.
- **Graph centrality:** In graph construction (kNN and/or GRN), assign **higher edge weights** to interactions involving **Oct4 (POU5F1), Sox2, Klf4, and c-Myc (MYC)**.
- **Synthetic perturbation:** Add a script that can **silence or overexpress** these factors in inputs to test whether hyperbolic embeddings move toward the expected “root” in embedding space.

---

## Phase 3: Trajectory Alignment (“Time Machine” Logic)

- **Loss function:** Implement **hyperbolic contrastive alignment** so LLM and graph branches align in **hyperbolic space**, not only with Euclidean cosine on projected vectors.
- **Curvature metrics:** Define a **pluripotency score** or **biological age proxy** from **hyperbolic distance** to a reference **embryonic stem cell (ESC)** cluster (or equivalent reference set), once defined in data.

---

## Phase 4: Validation and Benchmarking

- **Dataset:** Use public scRNA-seq of human fibroblasts undergoing **OKSM** reprogramming (e.g. **GSE103224** and related series) alongside current PBMC-style runs.
- **Visualization:** Use **hyperbolic UMAP** or the **uhg** visualization tooling to show trajectories. Target qualitative success: a **cone- or tree-like** structure with OSKM driving the main movement toward the root.

---

## Immediate Next Steps (Checklist)

- [ ] **HGNN / manifold layers:** Refactor the hyperbolic encoder path so Euclidean `torch.nn.Linear` (where it sits on the hyperbolic pathway) gives way to **`uhg` hyperbolic linear / manifold-native ops**, with **one** primary manifold (Lorentz vs Poincaré) end-to-end.
- [ ] **Alignment script / losses:** Update contrastive alignment to use **hyperbolic distance** (e.g. `uhg.manifolds.Lorentz.dist` if Lorentz is the chosen model) instead of relying solely on `F.cosine_similarity` on embeddings that are not guaranteed to live in the same geometric space.
- [ ] **Data prep (PBMC):** Isolate cells that **share regulatory pathways** with Yamanaka factors to stress-test “root-finding” before full reprogramming series are wired in.

## Progress So Far

- [x] Added compatibility layers so the dual-encoder training/evaluation stack works against a consistent API.
- [x] Centralized OSKM aliases and presence checks for human/mouse-style symbol resolution.
- [x] Added configurable OSKM anchor handling in the Cell2Sentence data path.
- [x] Added OSKM-aware graph reweighting for kNN builds.
- [x] Added in silico OSKM perturbation tooling and embedding-comparison workflows.
- [x] Added branch/risk overlays, partial reprogramming window heuristics, and longevity-safe-zone reporting.
- [x] Added config-driven reference labels, heuristic window profiles, and marker-panel scoring hooks for rejuvenation vs pluripotency risk.
- [x] Added a configurable geometry-aware alignment mode (`projective_distance`) alongside the Euclidean cosine baseline.
- [x] Added paired Euclidean-vs-projective ablation workflows, manifests, and safety/risk comparison plots for perturbation reports.

## Updated Remaining Build

1. Expand the dataset-backed validation bundle layer with track-specific manifests, expected timepoint contracts, and benchmark artifacts for fibroblast-to-iPSC and transient-partial-OSKM studies.
2. Harden and validate the curvature-aware alignment mode on real OKSM datasets using the paired ablation workflow and track-specific metrics.
3. Expand visuals from static QA plots into trajectory-centric hyperbolic views and a lightweight interactive explorer.
4. Tighten documentation around config profiles, benchmark datasets, and expected interpretation limits.
5. Refactor the HGNN stack toward more manifold-native operations once the validation loop is producing stable evidence.

---

## Repository Mapping (this codebase)

Names in earlier sketches (e.g. `src/hgnn/hgnn_encoder.py`) differ from this tree. Use these as the **current** integration points when implementing the phases above:

| Concept in plan | Likely files in this repo |
|-----------------|----------------------------|
| HGNN + UHG encoder | [`src/hgnn/uhg_hgnn_encoder.py`](src/hgnn/uhg_hgnn_encoder.py), [`src/hgnn/encoder.py`](src/hgnn/encoder.py), [`src/hgnn/layers.py`](src/hgnn/layers.py) |
| Dual-encoder alignment training | [`scripts/align_dual_encoder.py`](scripts/align_dual_encoder.py), [`src/fusion/trainer.py`](src/fusion/trainer.py) |
| Contrastive / alignment losses | [`src/fusion/align_losses.py`](src/fusion/align_losses.py) |
| Graph build entrypoint | [`scripts/build_graphs.py`](scripts/build_graphs.py); building blocks under [`src/graphs/`](src/graphs/) (e.g. [`build_knn.py`](src/graphs/build_knn.py), [`build_grn.py`](src/graphs/build_grn.py)) |
| Cell2Sentence / data | [`src/data/dataset.py`](src/data/dataset.py), [`src/data/collate.py`](src/data/collate.py) |
| UHG adapters / projection | [`src/uhg_adapters/`](src/uhg_adapters/), [`docs/uhg_api.md`](docs/uhg_api.md) |

During review, adjust this table if files move or split.

---

## Design Constraints (keep these explicit)

- **Manifold consistency:** Distance, linear maps, and the alignment loss should assume the **same** manifold and the **same** curvature handling (learnable $\kappa$ must be wired consistently).
- **Reference “root”:** ESC (or surrogate) cluster definition must be **dataset-specific** (cell IDs, batch, annotation column).
- **Naming:** Gene symbols vs Ensembl IDs for Oct4/Sox2/Klf4/MYC must match the **AnnData `var`** used in each run.

---

## Phase Overview (high level)

```mermaid
flowchart LR
  phase1[Phase1_UHG_Manifold]
  phase2[Phase2_OSKM_Features]
  phase3[Phase3_Hyp_Alignment]
  phase4[Phase4_Data_Viz]
  phase1 --> phase2 --> phase3 --> phase4
```
