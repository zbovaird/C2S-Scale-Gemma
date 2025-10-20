#!/usr/bin/env python3
"""
Cancer-Specific Text Encoder Enhancements

Enhanced C2S-Scale-Gemma text encoder with cancer-specific capabilities:
- Clinical context integration (patient demographics, treatment history)
- Drug-gene relationship understanding
- Pathway information integration
- Cancer-specific prompt templates
- Survival outcome modeling
- Drug mechanism of action understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Import base text components
from ..text.gemma_loader import GemmaLoader
from ..text.adapters import LoRAAdapter
from ..text.pooling import TextPooler

logger = logging.getLogger(__name__)

class ClinicalContextEncoder(nn.Module):
    """Encode clinical context information."""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Patient demographics
        self.age_encoder = nn.Linear(1, hidden_dim // 4)
        self.gender_encoder = nn.Embedding(3, hidden_dim // 4)  # Male, Female, Other
        self.race_encoder = nn.Embedding(6, hidden_dim // 4)    # Common races
        
        # Cancer stage and grade
        self.stage_encoder = nn.Embedding(5, hidden_dim // 4)   # Stage I-IV, Unknown
        self.grade_encoder = nn.Embedding(4, hidden_dim // 4)   # Grade 1-3, Unknown
        
        # Treatment history
        self.treatment_encoder = nn.Linear(10, hidden_dim // 2)  # 10 common treatments
        
        # Combine all clinical features
        self.clinical_combiner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, clinical_data: Dict) -> torch.Tensor:
        """Encode clinical context."""
        
        batch_size = 1  # Assume single patient for now
        
        # Encode demographics
        age = torch.tensor([clinical_data.get('age', 65)], dtype=torch.float32)
        age_embedding = self.age_encoder(age.unsqueeze(0))
        
        gender = torch.tensor([clinical_data.get('gender', 0)])  # 0=Male, 1=Female, 2=Other
        gender_embedding = self.gender_encoder(gender)
        
        race = torch.tensor([clinical_data.get('race', 0)])
        race_embedding = self.race_encoder(race)
        
        # Encode cancer information
        stage = torch.tensor([clinical_data.get('stage', 0)])  # 0=I, 1=II, 2=III, 3=IV, 4=Unknown
        stage_embedding = self.stage_encoder(stage)
        
        grade = torch.tensor([clinical_data.get('grade', 0)])  # 0=1, 1=2, 2=3, 3=Unknown
        grade_embedding = self.grade_encoder(grade)
        
        # Encode treatment history
        treatments = clinical_data.get('treatments', [0] * 10)
        treatment_tensor = torch.tensor(treatments, dtype=torch.float32).unsqueeze(0)
        treatment_embedding = self.treatment_encoder(treatment_tensor)
        
        # Combine all embeddings
        clinical_embedding = torch.cat([
            age_embedding,
            gender_embedding,
            race_embedding,
            stage_embedding,
            grade_embedding,
            treatment_embedding
        ], dim=-1)
        
        # Apply combiner
        clinical_embedding = self.clinical_combiner(clinical_embedding)
        
        return clinical_embedding


class DrugGeneRelationshipEncoder(nn.Module):
    """Encode drug-gene relationships."""
    
    def __init__(self, hidden_dim: int = 768, max_drugs: int = 100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_drugs = max_drugs
        
        # Drug embeddings
        self.drug_encoder = nn.Embedding(max_drugs, hidden_dim // 2)
        
        # Gene embeddings
        self.gene_encoder = nn.Embedding(20000, hidden_dim // 2)  # ~20k human genes
        
        # Drug-gene interaction encoder
        self.interaction_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Drug mechanism of action
        self.mechanism_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 10),  # 10 common mechanisms
            nn.Softmax(dim=-1)
        )
        
    def forward(self, drug_gene_data: Dict) -> Dict[str, torch.Tensor]:
        """Encode drug-gene relationships."""
        
        # Extract drug and gene information
        drugs = drug_gene_data.get('drugs', [])
        genes = drug_gene_data.get('genes', [])
        interactions = drug_gene_data.get('interactions', [])
        
        # Encode drugs
        drug_embeddings = []
        for drug_id in drugs:
            if drug_id < self.max_drugs:
                drug_emb = self.drug_encoder(torch.tensor(drug_id))
                drug_embeddings.append(drug_emb)
        
        if drug_embeddings:
            drug_embedding = torch.stack(drug_embeddings).mean(dim=0)
        else:
            drug_embedding = torch.zeros(self.hidden_dim // 2)
        
        # Encode genes
        gene_embeddings = []
        for gene_id in genes:
            if gene_id < 20000:
                gene_emb = self.gene_encoder(torch.tensor(gene_id))
                gene_embeddings.append(gene_emb)
        
        if gene_embeddings:
            gene_embedding = torch.stack(gene_embeddings).mean(dim=0)
        else:
            gene_embedding = torch.zeros(self.hidden_dim // 2)
        
        # Combine drug and gene embeddings
        combined_embedding = torch.cat([drug_embedding, gene_embedding], dim=-1)
        
        # Encode interactions
        interaction_embedding = self.interaction_encoder(combined_embedding)
        
        # Predict mechanism of action
        mechanism_prediction = self.mechanism_encoder(combined_embedding)
        
        return {
            'drug_embedding': drug_embedding,
            'gene_embedding': gene_embedding,
            'interaction_embedding': interaction_embedding,
            'mechanism_prediction': mechanism_prediction
        }


class PathwayInformationEncoder(nn.Module):
    """Encode biological pathway information."""
    
    def __init__(self, hidden_dim: int = 768, max_pathways: int = 1000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_pathways = max_pathways
        
        # Pathway embeddings
        self.pathway_encoder = nn.Embedding(max_pathways, hidden_dim // 2)
        
        # Pathway type embeddings
        self.pathway_type_encoder = nn.Embedding(10, hidden_dim // 4)  # 10 pathway types
        
        # Pathway-gene relationship encoder
        self.pathway_gene_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Cancer pathway importance
        self.cancer_importance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, pathway_data: Dict) -> Dict[str, torch.Tensor]:
        """Encode pathway information."""
        
        # Extract pathway information
        pathways = pathway_data.get('pathways', [])
        pathway_types = pathway_data.get('pathway_types', [])
        pathway_genes = pathway_data.get('pathway_genes', [])
        
        # Encode pathways
        pathway_embeddings = []
        for pathway_id in pathways:
            if pathway_id < self.max_pathways:
                pathway_emb = self.pathway_encoder(torch.tensor(pathway_id))
                pathway_embeddings.append(pathway_emb)
        
        if pathway_embeddings:
            pathway_embedding = torch.stack(pathway_embeddings).mean(dim=0)
        else:
            pathway_embedding = torch.zeros(self.hidden_dim // 2)
        
        # Encode pathway types
        type_embeddings = []
        for pathway_type in pathway_types:
            if pathway_type < 10:
                type_emb = self.pathway_type_encoder(torch.tensor(pathway_type))
                type_embeddings.append(type_emb)
        
        if type_embeddings:
            type_embedding = torch.stack(type_embeddings).mean(dim=0)
        else:
            type_embedding = torch.zeros(self.hidden_dim // 4)
        
        # Combine pathway and type embeddings
        combined_embedding = torch.cat([pathway_embedding, type_embedding], dim=-1)
        
        # Encode pathway-gene relationships
        pathway_gene_embedding = self.pathway_gene_encoder(combined_embedding)
        
        # Predict cancer importance
        cancer_importance = self.cancer_importance(pathway_gene_embedding)
        
        return {
            'pathway_embedding': pathway_embedding,
            'type_embedding': type_embedding,
            'pathway_gene_embedding': pathway_gene_embedding,
            'cancer_importance': cancer_importance
        }


class CancerGemmaLoader(GemmaLoader):
    """Cancer-specific C2S-Scale-Gemma loader with enhanced capabilities."""
    
    def __init__(
        self,
        model_name: str = "vandijklab/C2S-Scale-Gemma-2-27B",
        device: torch.device = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        quantization_config: Dict = None,
        use_auth_token: str = None,
        cancer_specific: bool = True,
        clinical_context: bool = True,
        drug_gene_relationships: bool = True,
        pathway_information: bool = True
    ):
        super().__init__(
            model_name=model_name,
            device=device,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            use_auth_token=use_auth_token
        )
        
        self.cancer_specific = cancer_specific
        self.clinical_context = clinical_context
        self.drug_gene_relationships = drug_gene_relationships
        self.pathway_information = pathway_information
        
        # Cancer-specific encoders
        if cancer_specific:
            if clinical_context:
                self.clinical_encoder = ClinicalContextEncoder()
            
            if drug_gene_relationships:
                self.drug_gene_encoder = DrugGeneRelationshipEncoder()
            
            if pathway_information:
                self.pathway_encoder = PathwayInformationEncoder()
        
        # Cancer-specific prompt templates
        self.cancer_prompt_templates = {
            'drug_response': self._create_drug_response_prompt,
            'prognosis': self._create_prognosis_prompt,
            'biomarker_discovery': self._create_biomarker_prompt,
            'cancer_classification': self._create_classification_prompt,
            'treatment_recommendation': self._create_treatment_prompt
        }
        
    def _create_drug_response_prompt(
        self,
        cell_sentence: str,
        drug_name: str,
        clinical_data: Dict = None,
        num_genes: int = 1000,
        organism: str = "Homo sapiens"
    ) -> str:
        """Create drug response prediction prompt."""
        
        base_prompt = f"""The following is a list of {num_genes} gene names ordered by descending expression level in a {organism} cancer cell. Your task is to predict the response to {drug_name} treatment based on the cell's gene expression profile.

Cell sentence: {cell_sentence}

Clinical Context:"""
        
        if clinical_data and self.clinical_context:
            clinical_info = f"""
- Patient Age: {clinical_data.get('age', 'Unknown')}
- Cancer Stage: {clinical_data.get('stage', 'Unknown')}
- Cancer Grade: {clinical_data.get('grade', 'Unknown')}
- Previous Treatments: {', '.join(clinical_data.get('treatments', []))}
"""
            base_prompt += clinical_info
        
        base_prompt += f"""

Based on the gene expression profile and clinical context, predict the response to {drug_name}:
- Drug Response: [Sensitive/Resistant/Partial]
- Confidence: [High/Medium/Low]
- Rationale: [Explanation based on gene expression and clinical factors]

The predicted response to {drug_name} is:"""
        
        return base_prompt
    
    def _create_prognosis_prompt(
        self,
        cell_sentence: str,
        clinical_data: Dict = None,
        num_genes: int = 1000,
        organism: str = "Homo sapiens"
    ) -> str:
        """Create prognosis prediction prompt."""
        
        base_prompt = f"""The following is a list of {num_genes} gene names ordered by descending expression level in a {organism} cancer cell. Your task is to predict the patient's prognosis based on the cell's gene expression profile.

Cell sentence: {cell_sentence}

Clinical Context:"""
        
        if clinical_data and self.clinical_context:
            clinical_info = f"""
- Patient Age: {clinical_data.get('age', 'Unknown')}
- Cancer Stage: {clinical_data.get('stage', 'Unknown')}
- Cancer Grade: {clinical_data.get('grade', 'Unknown')}
- Cancer Type: {clinical_data.get('cancer_type', 'Unknown')}
"""
            base_prompt += clinical_info
        
        base_prompt += f"""

Based on the gene expression profile and clinical context, predict the patient's prognosis:
- Overall Survival: [Good/Moderate/Poor]
- Disease-Free Survival: [Good/Moderate/Poor]
- Risk of Metastasis: [Low/Medium/High]
- Recommended Follow-up: [Frequency and type]

The predicted prognosis is:"""
        
        return base_prompt
    
    def _create_biomarker_prompt(
        self,
        cell_sentence: str,
        clinical_data: Dict = None,
        num_genes: int = 1000,
        organism: str = "Homo sapiens"
    ) -> str:
        """Create biomarker discovery prompt."""
        
        base_prompt = f"""The following is a list of {num_genes} gene names ordered by descending expression level in a {organism} cancer cell. Your task is to identify potential biomarkers for cancer diagnosis, prognosis, or treatment response.

Cell sentence: {cell_sentence}

Clinical Context:"""
        
        if clinical_data and self.clinical_context:
            clinical_info = f"""
- Cancer Type: {clinical_data.get('cancer_type', 'Unknown')}
- Cancer Stage: {clinical_data.get('stage', 'Unknown')}
- Treatment Response: {clinical_data.get('treatment_response', 'Unknown')}
"""
            base_prompt += clinical_info
        
        base_prompt += f"""

Based on the gene expression profile, identify potential biomarkers:
- Diagnostic Biomarkers: [Genes for cancer detection]
- Prognostic Biomarkers: [Genes for outcome prediction]
- Predictive Biomarkers: [Genes for treatment response]
- Novel Biomarkers: [Previously unknown biomarkers]

The identified biomarkers are:"""
        
        return base_prompt
    
    def _create_classification_prompt(
        self,
        cell_sentence: str,
        clinical_data: Dict = None,
        num_genes: int = 1000,
        organism: str = "Homo sapiens"
    ) -> str:
        """Create cancer classification prompt."""
        
        base_prompt = f"""The following is a list of {num_genes} gene names ordered by descending expression level in a {organism} cancer cell. Your task is to classify the cancer type based on the cell's gene expression profile.

Cell sentence: {cell_sentence}

Clinical Context:"""
        
        if clinical_data and self.clinical_context:
            clinical_info = f"""
- Patient Demographics: Age {clinical_data.get('age', 'Unknown')}, {clinical_data.get('gender', 'Unknown')}
- Tissue of Origin: {clinical_data.get('tissue_type', 'Unknown')}
- Previous Diagnosis: {clinical_data.get('previous_diagnosis', 'None')}
"""
            base_prompt += clinical_info
        
        base_prompt += f"""

Based on the gene expression profile and clinical context, classify the cancer:
- Primary Cancer Type: [Breast/Lung/Colon/Prostate/Melanoma/Other]
- Cancer Subtype: [Specific subtype if applicable]
- Confidence: [High/Medium/Low]
- Key Diagnostic Genes: [Genes supporting the classification]

The cancer classification is:"""
        
        return base_prompt
    
    def _create_treatment_prompt(
        self,
        cell_sentence: str,
        clinical_data: Dict = None,
        num_genes: int = 1000,
        organism: str = "Homo sapiens"
    ) -> str:
        """Create treatment recommendation prompt."""
        
        base_prompt = f"""The following is a list of {num_genes} gene names ordered by descending expression level in a {organism} cancer cell. Your task is to recommend the best treatment strategy based on the cell's gene expression profile.

Cell sentence: {cell_sentence}

Clinical Context:"""
        
        if clinical_data and self.clinical_context:
            clinical_info = f"""
- Cancer Type: {clinical_data.get('cancer_type', 'Unknown')}
- Cancer Stage: {clinical_data.get('stage', 'Unknown')}
- Previous Treatments: {', '.join(clinical_data.get('treatments', []))}
- Treatment Response: {clinical_data.get('treatment_response', 'Unknown')}
- Patient Preferences: {clinical_data.get('patient_preferences', 'Standard care')}
"""
            base_prompt += clinical_info
        
        base_prompt += f"""

Based on the gene expression profile and clinical context, recommend treatment:
- Primary Treatment: [Recommended first-line treatment]
- Alternative Treatments: [Backup treatment options]
- Targeted Therapies: [Specific targeted drugs]
- Combination Therapies: [Drug combinations]
- Clinical Trial Options: [Available trials]
- Rationale: [Explanation for recommendations]

The recommended treatment strategy is:"""
        
        return base_prompt
    
    def predict_drug_response(
        self,
        cell_sentence: str,
        drug_name: str,
        clinical_data: Dict = None,
        max_new_tokens: int = 100,
        num_genes: int = 1000,
        organism: str = "Homo sapiens"
    ) -> str:
        """Predict drug response for cancer cells."""
        
        # Create cancer-specific prompt
        prompt = self._create_drug_response_prompt(
            cell_sentence=cell_sentence,
            drug_name=drug_name,
            clinical_data=clinical_data,
            num_genes=num_genes,
            organism=organism
        )
        
        # Generate response
        response = self.generate_response(prompt, max_new_tokens=max_new_tokens)
        
        return response
    
    def predict_prognosis(
        self,
        cell_sentence: str,
        clinical_data: Dict = None,
        max_new_tokens: int = 100,
        num_genes: int = 1000,
        organism: str = "Homo sapiens"
    ) -> str:
        """Predict cancer prognosis."""
        
        # Create prognosis prompt
        prompt = self._create_prognosis_prompt(
            cell_sentence=cell_sentence,
            clinical_data=clinical_data,
            num_genes=num_genes,
            organism=organism
        )
        
        # Generate response
        response = self.generate_response(prompt, max_new_tokens=max_new_tokens)
        
        return response
    
    def discover_biomarkers(
        self,
        cell_sentence: str,
        clinical_data: Dict = None,
        max_new_tokens: int = 100,
        num_genes: int = 1000,
        organism: str = "Homo sapiens"
    ) -> str:
        """Discover cancer biomarkers."""
        
        # Create biomarker discovery prompt
        prompt = self._create_biomarker_prompt(
            cell_sentence=cell_sentence,
            clinical_data=clinical_data,
            num_genes=num_genes,
            organism=organism
        )
        
        # Generate response
        response = self.generate_response(prompt, max_new_tokens=max_new_tokens)
        
        return response
    
    def classify_cancer(
        self,
        cell_sentence: str,
        clinical_data: Dict = None,
        max_new_tokens: int = 100,
        num_genes: int = 1000,
        organism: str = "Homo sapiens"
    ) -> str:
        """Classify cancer type."""
        
        # Create classification prompt
        prompt = self._create_classification_prompt(
            cell_sentence=cell_sentence,
            clinical_data=clinical_data,
            num_genes=num_genes,
            organism=organism
        )
        
        # Generate response
        response = self.generate_response(prompt, max_new_tokens=max_new_tokens)
        
        return response
    
    def recommend_treatment(
        self,
        cell_sentence: str,
        clinical_data: Dict = None,
        max_new_tokens: int = 100,
        num_genes: int = 1000,
        organism: str = "Homo sapiens"
    ) -> str:
        """Recommend cancer treatment strategy."""
        
        # Create treatment recommendation prompt
        prompt = self._create_treatment_prompt(
            cell_sentence=cell_sentence,
            clinical_data=clinical_data,
            num_genes=num_genes,
            organism=organism
        )
        
        # Generate response
        response = self.generate_response(prompt, max_new_tokens=max_new_tokens)
        
        return response
    
    def encode_with_clinical_context(
        self,
        cell_sentence: str,
        clinical_data: Dict = None,
        drug_gene_data: Dict = None,
        pathway_data: Dict = None
    ) -> Dict[str, torch.Tensor]:
        """Encode cell sentence with clinical context."""
        
        # Base text encoding
        base_embedding = self.encode_text(cell_sentence)
        
        # Cancer-specific enhancements
        cancer_embeddings = {}
        
        if self.clinical_context and clinical_data:
            clinical_embedding = self.clinical_encoder(clinical_data)
            cancer_embeddings['clinical'] = clinical_embedding
        
        if self.drug_gene_relationships and drug_gene_data:
            drug_gene_embeddings = self.drug_gene_encoder(drug_gene_data)
            cancer_embeddings.update(drug_gene_embeddings)
        
        if self.pathway_information and pathway_data:
            pathway_embeddings = self.pathway_encoder(pathway_data)
            cancer_embeddings.update(pathway_embeddings)
        
        # Combine all embeddings
        combined_embedding = base_embedding
        for embedding_name, embedding in cancer_embeddings.items():
            combined_embedding = combined_embedding + embedding
        
        return {
            'base_embedding': base_embedding,
            'combined_embedding': combined_embedding,
            'cancer_embeddings': cancer_embeddings
        }
