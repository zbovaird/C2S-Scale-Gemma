"""
Dataset loading and preprocessing for C2S-Scale-Gemma Hybrid.

Handles loading of single-cell data, conversion to cell sentences, and graph construction.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
try:
    import scanpy as sc
except ImportError:  # pragma: no cover - optional runtime dependency
    sc = None
import torch
from torch.utils.data import Dataset, DataLoader
try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - optional runtime dependency
    AutoTokenizer = None
try:
    import anndata as ad
except ImportError:  # pragma: no cover - optional runtime dependency
    ad = None

from biology.oskm import resolve_oskm_genes
from biology.panels import score_expression_panels

logger = logging.getLogger(__name__)


class CellSentenceDataset(Dataset):
    """
    Dataset for cell sentences and associated metadata.
    
    Loads single-cell data and converts it to cell sentences format.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer=None,
        tokenizer_name: str = "google/gemma-2-7b",
        max_seq_length: int = 2048,
        cache_dir: Optional[str] = None,
        max_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        top_genes: int = 1000,
        oskm_anchor_mode: str = "none",
        oskm_species: str = "human",
        marker_panels: Optional[Dict[str, List[str]]] = None,
        metadata_columns: Optional[List[str]] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to data directory or specific file
            tokenizer_name: HuggingFace tokenizer name
            max_seq_length: Maximum sequence length for tokenization
            cache_dir: Cache directory for tokenizer
        """
        self.data_path = Path(data_path)
        self.max_seq_length = max_length or max_seq_length
        self.device = device
        self.top_genes = top_genes
        self.oskm_anchor_mode = oskm_anchor_mode
        self.oskm_species = oskm_species
        self.marker_panels = marker_panels or {}
        self.metadata_columns = metadata_columns or []
        
        # Reuse provided tokenizer when scripts already loaded one.
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            if AutoTokenizer is None:
                raise ImportError(
                    "transformers is required to construct a tokenizer automatically."
                )
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                cache_dir=cache_dir
            )
        
        # Load data
        self.cell_data = self._load_data()
        
        logger.info(f"Loaded {len(self.cell_data)} cells")
    
    def _load_data(self) -> pd.DataFrame:
        """Load cell data from various formats."""
        data_path = Path(self.data_path)
        
        # Try different file formats
        if data_path.is_file():
            if data_path.suffix == '.h5ad':
                return self._load_h5ad(data_path)
            elif data_path.suffix == '.csv':
                return self._load_csv(data_path)
            elif data_path.suffix == '.parquet':
                return self._load_parquet(data_path)
        elif data_path.is_dir():
            # Look for files in directory
            for pattern in ['*.h5ad', '*.csv', '*.parquet']:
                files = list(data_path.glob(pattern))
                if files:
                    return self._load_data_file(files[0])
        
        raise ValueError(f"Could not find valid data file in {data_path}")
    
    def _load_h5ad(self, file_path: Path) -> pd.DataFrame:
        """Load AnnData file and convert to cell sentences."""
        logger.info(f"Loading AnnData from {file_path}")
        if sc is None:
            raise ImportError("scanpy is required to load .h5ad datasets.")
        
        adata = sc.read_h5ad(file_path)
        
        # Convert to cell sentences using C2S-Scale-Gemma format
        cell_sentences = []
        gene_names = adata.var_names.tolist()
        gene_index_lookup = {gene_name: idx for idx, gene_name in enumerate(gene_names)}
        resolved_oskm_genes = resolve_oskm_genes(gene_names, species=self.oskm_species)
        anchor_genes = list(resolved_oskm_genes.values())
        oskm_scores = []
        oskm_present_symbols = []
        panel_scores = (
            score_expression_panels(
                adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X),
                gene_names,
                self.marker_panels,
            )
            if self.marker_panels
            else {}
        )
        
        for i in range(adata.n_obs):
            # Get expression values
            if hasattr(adata.X, 'toarray'):
                expr = adata.X[i].toarray().flatten()
            else:
                expr = adata.X[i]
            
            # Rank genes by expression (descending) - C2S-Scale-Gemma format
            ranked_indices = expr.argsort()[::-1]
            ranked_gene_names = [
                gene_names[idx]
                for idx in ranked_indices[: self.top_genes]
                if expr[idx] > 0
            ]
            ranked_gene_names = self._apply_anchor_policy(ranked_gene_names, anchor_genes)
            sentence = " ".join(ranked_gene_names)
            cell_sentences.append(sentence)

            present_symbols = [
                symbol for symbol in anchor_genes if symbol in ranked_gene_names
            ]
            oskm_present_symbols.append(" ".join(present_symbols))
            oskm_scores.append(
                float(sum(expr[gene_index_lookup[symbol]] for symbol in present_symbols))
            )
        
        # Create DataFrame
        df = pd.DataFrame({
            'cell_id': adata.obs.index,
            'cell_sentence': cell_sentences,
            'cell_type': adata.obs.get('cell_type', 'unknown'),
            'tissue': adata.obs.get('tissue', 'unknown'),
            'n_genes': adata.obs.get('n_genes', 0),
            'total_counts': adata.obs.get('total_counts', 0),
            'oskm_score': oskm_scores,
            'oskm_present_symbols': oskm_present_symbols,
        })

        for column_name in self.metadata_columns:
            if column_name in df.columns:
                continue
            if column_name in adata.obs.columns:
                df[column_name] = adata.obs[column_name].tolist()

        for panel_name, scores in panel_scores.items():
            df[f"{panel_name}_score"] = scores
        
        return df

    def _apply_anchor_policy(
        self,
        ranked_gene_names: List[str],
        anchor_genes: List[str],
    ) -> List[str]:
        """Optionally promote OKSM genes in the cell sentence."""
        if self.oskm_anchor_mode == "none":
            return ranked_gene_names

        if self.oskm_anchor_mode != "prepend_present":
            raise ValueError(f"Unsupported oskm_anchor_mode: {self.oskm_anchor_mode}")

        anchor_hits = [gene for gene in anchor_genes if gene in ranked_gene_names]
        non_anchor_genes = [gene for gene in ranked_gene_names if gene not in anchor_hits]
        return anchor_hits + non_anchor_genes
    
    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """Load CSV file."""
        return pd.read_csv(file_path)
    
    def _load_parquet(self, file_path: Path) -> pd.DataFrame:
        """Load Parquet file."""
        return pd.read_parquet(file_path)
    
    def _load_data_file(self, file_path: Path) -> pd.DataFrame:
        """Load data from specific file."""
        if file_path.suffix == '.h5ad':
            return self._load_h5ad(file_path)
        elif file_path.suffix == '.csv':
            return self._load_csv(file_path)
        elif file_path.suffix == '.parquet':
            return self._load_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def __len__(self) -> int:
        """Return number of cells."""
        return len(self.cell_data)
    
    def create_c2s_prompt(self, cell_sentence: str, task: str = "cell_type", num_genes: int = 1000, organism: str = "Homo sapiens") -> str:
        """
        Create C2S-Scale-Gemma formatted prompt.
        
        Args:
            cell_sentence: Space-separated gene names ordered by expression
            task: Task type (cell_type, tissue, etc.)
            num_genes: Number of genes in the sentence
            organism: Organism name
            
        Returns:
            Formatted prompt for C2S-Scale-Gemma model
        """
        if task == "cell_type":
            prompt = f"""The following is a list of {num_genes} gene names ordered by descending expression level in a {organism} cell. Your task is to give the cell type which this cell belongs to based on its gene expression.
Cell sentence: {cell_sentence}.
The cell type corresponding to these genes is:"""
        elif task == "tissue":
            prompt = f"""The following is a list of {num_genes} gene names ordered by descending expression level in a {organism} cell. Your task is to give the tissue which this cell belongs to based on its gene expression.
Cell sentence: {cell_sentence}.
The tissue corresponding to these genes is:"""
        else:
            # Generic prompt
            prompt = f"""The following is a list of {num_genes} gene names ordered by descending expression level in a {organism} cell.
Cell sentence: {cell_sentence}."""
            
        return prompt
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item by index.
        
        Returns:
            Dictionary containing tokenized cell sentence and metadata
        """
        row = self.cell_data.iloc[idx]
        
        # Get cell sentence
        cell_sentence = row['cell_sentence']
        
        # Create C2S-Scale-Gemma formatted prompt
        prompt = self.create_c2s_prompt(cell_sentence, task="cell_type")
        
        # Tokenize the prompt
        tokenized = self.tokenizer(
            prompt,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'cell_sentence': cell_sentence,
            'prompt': prompt,
            'cell_id': row['cell_id'],
            'cell_type': row['cell_type'],
            'tissue': row['tissue'],
            'n_genes': row['n_genes'],
            'total_counts': row['total_counts'],
            'oskm_score': row.get('oskm_score', 0.0),
            'oskm_present_symbols': row.get('oskm_present_symbols', ""),
        }
    
    def get_cell_types(self) -> List[str]:
        """Get list of unique cell types."""
        return self.cell_data['cell_type'].unique().tolist()
    
    def get_tissues(self) -> List[str]:
        """Get list of unique tissues."""
        return self.cell_data['tissue'].unique().tolist()
    
    def get_cell_type_counts(self) -> Dict[str, int]:
        """Get count of cells per cell type."""
        return self.cell_data['cell_type'].value_counts().to_dict()
    
    def get_tissue_counts(self) -> Dict[str, int]:
        """Get count of cells per tissue."""
        return self.cell_data['tissue'].value_counts().to_dict()


class Cell2SentenceDataset(CellSentenceDataset):
    """Compatibility alias for older script imports."""


class GraphDataset(Dataset):
    """
    Dataset for graph data (kNN, ligand-receptor, GRN).
    
    Loads pre-computed graph structures.
    """
    
    def __init__(
        self,
        graph_paths: Dict[str, Union[str, Path]],
        cell_data_path: Union[str, Path]
    ):
        """
        Initialize graph dataset.
        
        Args:
            graph_paths: Dictionary mapping graph type to file path
            cell_data_path: Path to cell data
        """
        self.graph_paths = {k: Path(v) for k, v in graph_paths.items()}
        self.cell_data_path = Path(cell_data_path)
        
        # Load graphs
        self.graphs = self._load_graphs()
        
        # Load cell data
        self.cell_data = self._load_cell_data()
        
        logger.info(f"Loaded graphs: {list(self.graphs.keys())}")
        logger.info(f"Loaded {len(self.cell_data)} cells")
    
    def _load_graphs(self) -> Dict[str, pd.DataFrame]:
        """Load graph data from files."""
        graphs = {}
        
        for graph_type, path in self.graph_paths.items():
            if path.exists():
                if path.suffix == '.parquet':
                    graphs[graph_type] = pd.read_parquet(path)
                elif path.suffix == '.csv':
                    graphs[graph_type] = pd.read_csv(path)
                else:
                    logger.warning(f"Unsupported graph file format: {path.suffix}")
            else:
                logger.warning(f"Graph file not found: {path}")
        
        return graphs
    
    def _load_cell_data(self) -> pd.DataFrame:
        """Load cell data."""
        if self.cell_data_path.suffix == '.h5ad':
            adata = sc.read_h5ad(self.cell_data_path)
            return pd.DataFrame({
                'cell_id': adata.obs.index,
                'cell_type': adata.obs.get('cell_type', 'unknown'),
                'tissue': adata.obs.get('tissue', 'unknown')
            })
        else:
            return pd.read_csv(self.cell_data_path)
    
    def __len__(self) -> int:
        """Return number of cells."""
        return len(self.cell_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item by index.
        
        Returns:
            Dictionary containing graph data for the cell
        """
        cell_id = self.cell_data.iloc[idx]['cell_id']
        
        result = {
            'cell_id': cell_id,
            'cell_type': self.cell_data.iloc[idx]['cell_type'],
            'tissue': self.cell_data.iloc[idx]['tissue']
        }
        
        # Add graph data
        for graph_type, graph_df in self.graphs.items():
            if graph_type == 'knn':
                # Get kNN neighbors
                neighbors = graph_df[graph_df['source'] == cell_id]['target'].tolist()
                result[f'{graph_type}_neighbors'] = torch.tensor(neighbors, dtype=torch.long)
            elif graph_type == 'lr':
                # Get ligand-receptor interactions
                lr_interactions = graph_df[
                    (graph_df['ligand_cell'] == cell_id) | 
                    (graph_df['receptor_cell'] == cell_id)
                ]
                result[f'{graph_type}_interactions'] = torch.tensor(
                    lr_interactions.values, dtype=torch.long
                )
            elif graph_type == 'grn':
                # Get gene regulatory network
                grn_edges = graph_df[graph_df['source'] == cell_id]['target'].tolist()
                result[f'{graph_type}_edges'] = torch.tensor(grn_edges, dtype=torch.long)
        
        return result


class PairedDataset(Dataset):
    """
    Dataset that pairs cell sentences with graph data.
    
    Combines CellSentenceDataset and GraphDataset.
    """
    
    def __init__(
        self,
        cell_sentence_dataset: CellSentenceDataset,
        graph_dataset: GraphDataset
    ):
        """
        Initialize paired dataset.
        
        Args:
            cell_sentence_dataset: Dataset for cell sentences
            graph_dataset: Dataset for graph data
        """
        self.cell_sentence_dataset = cell_sentence_dataset
        self.graph_dataset = graph_dataset
        
        # Ensure datasets have same length
        if len(cell_sentence_dataset) != len(graph_dataset):
            raise ValueError("Cell sentence and graph datasets must have same length")
    
    def __len__(self) -> int:
        """Return number of cells."""
        return len(self.cell_sentence_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get paired item by index.
        
        Returns:
            Dictionary containing both text and graph data
        """
        # Get text data
        text_data = self.cell_sentence_dataset[idx]
        
        # Get graph data
        graph_data = self.graph_dataset[idx]
        
        # Combine data
        combined_data = {**text_data, **graph_data}
        
        return combined_data


def create_data_loader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn: Optional[callable] = None
) -> DataLoader:
    """
    Create DataLoader for dataset.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        collate_fn: Custom collate function
    
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
