#!/usr/bin/env python3
"""
Cancer-Specific Graph Construction

Builds cancer-specific graphs from downloaded data:
- Drug-target networks
- Cancer signaling pathways
- Tumor microenvironment
- Metastasis networks
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import json
import scanpy as sc
import anndata

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CancerGraphBuilder:
    """Build cancer-specific graphs from downloaded data."""
    
    def __init__(self, data_dir: str = "data/cancer", output_dir: str = "data/processed/graphs"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Graph storage
        self.graphs = {}
    
    def build_drug_target_graph(self) -> nx.Graph:
        """Build drug-target interaction graph."""
        logger.info("Building drug-target interaction graph...")
        
        try:
            # Load drug-target data
            drug_targets_path = self.data_dir / "chembl_drug_targets.csv"
            if not drug_targets_path.exists():
                logger.error(f"Drug-target data not found at {drug_targets_path}")
                return nx.Graph()
            
            df = pd.read_csv(drug_targets_path)
            
            # Create graph
            G = nx.Graph()
            
            # Add nodes and edges
            for _, row in df.iterrows():
                drug_id = row['drug_id']
                target_id = row['target_id']
                
                # Add drug node
                G.add_node(drug_id, type='drug', 
                          mechanism=row.get('mechanism_of_action', ''),
                          action_type=row.get('action_type', ''))
                
                # Add target node
                G.add_node(target_id, type='target')
                
                # Add edge
                G.add_edge(drug_id, target_id, 
                          direct_interaction=row.get('direct_interaction', False),
                          disease_efficacy=row.get('disease_efficacy', ''))
            
            # Save graph
            output_path = self.output_dir / "drug_target_graph.parquet"
            self._save_graph(G, output_path)
            
            logger.info(f"Built drug-target graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            logger.error(f"Error building drug-target graph: {e}")
            return nx.Graph()
    
    def build_cancer_pathway_graph(self) -> nx.Graph:
        """Build cancer signaling pathway graph."""
        logger.info("Building cancer signaling pathway graph...")
        
        try:
            # Load pathway data
            pathways_path = self.data_dir / "kegg_cancer_pathways.csv"
            if not pathways_path.exists():
                logger.error(f"Pathway data not found at {pathways_path}")
                return nx.Graph()
            
            df = pd.read_csv(pathways_path)
            
            # Create graph
            G = nx.Graph()
            
            # Add pathway nodes
            pathways = df['pathway_id'].unique()
            for pathway_id in pathways:
                pathway_name = df[df['pathway_id'] == pathway_id]['pathway_name'].iloc[0]
                G.add_node(pathway_id, type='pathway', name=pathway_name)
            
            # Add gene nodes and pathway-gene edges
            for _, row in df.iterrows():
                gene_id = row['gene_id']
                pathway_id = row['pathway_id']
                
                # Add gene node
                G.add_node(gene_id, type='gene')
                
                # Add pathway-gene edge
                G.add_edge(pathway_id, gene_id, relationship='contains')
            
            # Add gene-gene edges based on pathway co-membership
            pathway_groups = df.groupby('pathway_id')['gene_id'].apply(list)
            for pathway_id, genes in pathway_groups.items():
                # Connect genes within the same pathway
                for i, gene1 in enumerate(genes):
                    for gene2 in genes[i+1:]:
                        if G.has_edge(gene1, gene2):
                            G[gene1][gene2]['weight'] += 1
                        else:
                            G.add_edge(gene1, gene2, weight=1, pathway=pathway_id)
            
            # Save graph
            output_path = self.output_dir / "cancer_pathway_graph.parquet"
            self._save_graph(G, output_path)
            
            logger.info(f"Built cancer pathway graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            logger.error(f"Error building cancer pathway graph: {e}")
            return nx.Graph()
    
    def build_protein_interaction_graph(self) -> nx.Graph:
        """Build protein-protein interaction graph."""
        logger.info("Building protein-protein interaction graph...")
        
        try:
            # Load protein interaction data
            interactions_path = self.data_dir / "string_protein_interactions.csv"
            if not interactions_path.exists():
                logger.error(f"Protein interaction data not found at {interactions_path}")
                return nx.Graph()
            
            df = pd.read_csv(interactions_path)
            
            # Create graph
            G = nx.Graph()
            
            # Add nodes and edges
            for _, row in df.iterrows():
                protein1 = row.get('preferredName_A', '')
                protein2 = row.get('preferredName_B', '')
                score = float(row.get('score', 0))
                
                if protein1 and protein2 and score > 0:
                    # Add protein nodes
                    G.add_node(protein1, type='protein')
                    G.add_node(protein2, type='protein')
                    
                    # Add edge with confidence score
                    G.add_edge(protein1, protein2, 
                              confidence=score,
                              interaction_type='functional')
            
            # Save graph
            output_path = self.output_dir / "protein_interaction_graph.parquet"
            self._save_graph(G, output_path)
            
            logger.info(f"Built protein interaction graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            logger.error(f"Error building protein interaction graph: {e}")
            return nx.Graph()
    
    def build_tumor_microenvironment_graph(self, cancer_data: anndata.AnnData) -> nx.Graph:
        """Build tumor microenvironment graph from single-cell data."""
        logger.info("Building tumor microenvironment graph...")
        
        try:
            # Create graph
            G = nx.Graph()
            
            # Add cell nodes
            for cell_id in cancer_data.obs.index:
                cell_type = cancer_data.obs.loc[cell_id, 'cancer_type']
                stage = cancer_data.obs.loc[cell_id, 'stage']
                grade = cancer_data.obs.loc[cell_id, 'grade']
                
                G.add_node(cell_id, 
                          type='cell',
                          cancer_type=cell_type,
                          stage=stage,
                          grade=grade)
            
            # Add cell-cell edges based on expression similarity
            # Use top variable genes for similarity calculation
            sc.pp.highly_variable_genes(cancer_data, min_mean=0.0125, max_mean=3, min_disp=0.5)
            cancer_data.raw = cancer_data
            cancer_data = cancer_data[:, cancer_data.var.highly_variable]
            
            # Calculate pairwise similarities
            from sklearn.metrics.pairwise import cosine_similarity
            expression_matrix = cancer_data.X.toarray() if hasattr(cancer_data.X, 'toarray') else cancer_data.X
            
            # Sample cells for computational efficiency
            n_cells = min(1000, cancer_data.n_obs)
            cell_indices = np.random.choice(cancer_data.n_obs, size=n_cells, replace=False)
            sampled_expression = expression_matrix[cell_indices]
            
            # Calculate cosine similarities
            similarities = cosine_similarity(sampled_expression)
            
            # Add edges for similar cells
            threshold = 0.7  # Similarity threshold
            for i in range(len(cell_indices)):
                for j in range(i+1, len(cell_indices)):
                    if similarities[i, j] > threshold:
                        cell1 = cancer_data.obs.index[cell_indices[i]]
                        cell2 = cancer_data.obs.index[cell_indices[j]]
                        G.add_edge(cell1, cell2, 
                                  similarity=similarities[i, j],
                                  relationship='similar_expression')
            
            # Save graph
            output_path = self.output_dir / "tumor_microenvironment_graph.parquet"
            self._save_graph(G, output_path)
            
            logger.info(f"Built tumor microenvironment graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            logger.error(f"Error building tumor microenvironment graph: {e}")
            return nx.Graph()
    
    def build_metastasis_graph(self) -> nx.Graph:
        """Build metastasis network graph."""
        logger.info("Building metastasis network graph...")
        
        try:
            # Create metastasis network based on known metastasis pathways
            G = nx.DiGraph()  # Directed graph for metastasis flow
            
            # Define metastasis pathways (simplified)
            metastasis_pathways = {
                'primary_tumor': ['invasion', 'angiogenesis', 'immune_escape'],
                'invasion': ['EMT', 'matrix_degradation', 'migration'],
                'EMT': ['mesenchymal_transition', 'stem_cell_activation'],
                'migration': ['circulation', 'arrest'],
                'circulation': ['survival', 'arrest'],
                'arrest': ['extravasation', 'colonization'],
                'colonization': ['metastatic_growth']
            }
            
            # Add nodes and edges
            for source, targets in metastasis_pathways.items():
                G.add_node(source, type='metastasis_step')
                for target in targets:
                    G.add_node(target, type='metastasis_step')
                    G.add_edge(source, target, relationship='leads_to')
            
            # Add cancer-specific metastasis patterns
            cancer_metastasis = {
                'breast_cancer': ['bone', 'lung', 'liver', 'brain'],
                'lung_cancer': ['bone', 'liver', 'brain', 'adrenal'],
                'colon_cancer': ['liver', 'lung', 'peritoneum'],
                'prostate_cancer': ['bone', 'liver', 'lung'],
                'melanoma': ['lung', 'liver', 'brain', 'bone']
            }
            
            for cancer_type, sites in cancer_metastasis.items():
                G.add_node(cancer_type, type='cancer_type')
                for site in sites:
                    G.add_node(site, type='metastatic_site')
                    G.add_edge(cancer_type, site, relationship='metastasizes_to')
            
            # Save graph
            output_path = self.output_dir / "metastasis_graph.parquet"
            self._save_graph(G, output_path)
            
            logger.info(f"Built metastasis graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            logger.error(f"Error building metastasis graph: {e}")
            return nx.DiGraph()
    
    def _save_graph(self, G: nx.Graph, output_path: Path):
        """Save graph to parquet format."""
        try:
            # Convert to DataFrame format
            edges = []
            for u, v, data in G.edges(data=True):
                edges.append({
                    'source': u,
                    'target': v,
                    **data
                })
            
            edges_df = pd.DataFrame(edges)
            edges_df.to_parquet(output_path, index=False)
            
            # Save node attributes separately
            nodes = []
            for node, data in G.nodes(data=True):
                nodes.append({
                    'node': node,
                    **data
                })
            
            nodes_df = pd.DataFrame(nodes)
            nodes_path = output_path.parent / f"{output_path.stem}_nodes.parquet"
            nodes_df.to_parquet(nodes_path, index=False)
            
        except Exception as e:
            logger.error(f"Error saving graph to {output_path}: {e}")
    
    def build_all_cancer_graphs(self) -> Dict[str, nx.Graph]:
        """Build all cancer-specific graphs."""
        logger.info("Building all cancer-specific graphs...")
        
        # Build individual graphs
        self.graphs['drug_target'] = self.build_drug_target_graph()
        self.graphs['cancer_pathway'] = self.build_cancer_pathway_graph()
        self.graphs['protein_interaction'] = self.build_protein_interaction_graph()
        self.graphs['metastasis'] = self.build_metastasis_graph()
        
        # Build tumor microenvironment graph if cancer data is available
        cancer_data_path = self.data_dir / "cellxgene_cancer_data.h5ad"
        if cancer_data_path.exists():
            cancer_data = sc.read_h5ad(cancer_data_path)
            self.graphs['tumor_microenvironment'] = self.build_tumor_microenvironment_graph(cancer_data)
        
        # Create summary
        summary = {}
        for name, graph in self.graphs.items():
            summary[name] = {
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'is_directed': graph.is_directed()
            }
        
        # Save summary
        summary_path = self.output_dir / "cancer_graphs_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Built {len(self.graphs)} cancer graphs: {summary}")
        return self.graphs

def main():
    """Main function to build cancer graphs."""
    builder = CancerGraphBuilder()
    graphs = builder.build_all_cancer_graphs()
    
    print("\n🎉 Cancer graph construction complete!")
    print(f"📊 Built {len(graphs)} graphs")
    print(f"💾 Graphs saved to: {builder.output_dir}")

if __name__ == "__main__":
    main()
