#!/usr/bin/env python3
"""
Cell Type Prediction Evaluation Script for C2S-Scale-Gemma Hybrid Model.

This script evaluates the cell type prediction performance using the actual
C2S-Scale-Gemma model with proper prompt formatting.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.dataset import CellSentenceDataset
from text.c2s_gemma_loader import C2SScaleGemmaLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_cell_type_prediction(
    model_loader: C2SScaleGemmaLoader,
    dataset: CellSentenceDataset,
    max_samples: int = 100,
    max_new_tokens: int = 20
) -> Dict[str, Any]:
    """
    Evaluate cell type prediction performance.
    
    Args:
        model_loader: C2S-Scale-Gemma model loader
        dataset: Cell sentence dataset
        max_samples: Maximum number of samples to evaluate
        max_new_tokens: Maximum new tokens for generation
        
    Returns:
        Dictionary containing evaluation results
    """
    logger.info(f"Evaluating cell type prediction on {min(max_samples, len(dataset))} samples")
    
    predictions = []
    ground_truth = []
    cell_sentences = []
    
    # Evaluate samples
    for i in range(min(max_samples, len(dataset))):
        if i % 10 == 0:
            logger.info(f"Evaluating sample {i}/{min(max_samples, len(dataset))}")
        
        # Get sample
        sample = dataset[i]
        cell_sentence = sample['cell_sentence']
        true_cell_type = sample['cell_type']
        
        try:
            # Predict cell type
            predicted_cell_type = model_loader.predict_cell_type(
                cell_sentence=cell_sentence,
                max_new_tokens=max_new_tokens
            )
            
            predictions.append(predicted_cell_type)
            ground_truth.append(true_cell_type)
            cell_sentences.append(cell_sentence)
            
        except Exception as e:
            logger.warning(f"Error predicting sample {i}: {e}")
            predictions.append("unknown")
            ground_truth.append(true_cell_type)
            cell_sentences.append(cell_sentence)
    
    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predictions)
    
    # Classification report
    unique_labels = list(set(ground_truth + predictions))
    classification_rep = classification_report(
        ground_truth, 
        predictions, 
        labels=unique_labels,
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(ground_truth, predictions, labels=unique_labels)
    
    results = {
        'accuracy': accuracy,
        'classification_report': classification_rep,
        'confusion_matrix': conf_matrix,
        'unique_labels': unique_labels,
        'predictions': predictions,
        'ground_truth': ground_truth,
        'cell_sentences': cell_sentences,
        'num_samples': len(predictions)
    }
    
    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}")
    
    return results


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    labels: List[str],
    save_path: str = "confusion_matrix.png"
) -> None:
    """Plot confusion matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title('Cell Type Prediction Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to {save_path}")


def plot_accuracy_by_cell_type(
    classification_report: Dict[str, Any],
    save_path: str = "accuracy_by_cell_type.png"
) -> None:
    """Plot accuracy by cell type."""
    cell_types = []
    accuracies = []
    
    for label, metrics in classification_report.items():
        if isinstance(metrics, dict) and 'precision' in metrics:
            cell_types.append(label)
            accuracies.append(metrics['precision'])
    
    plt.figure(figsize=(12, 6))
    plt.bar(cell_types, accuracies)
    plt.title('Cell Type Prediction Accuracy by Cell Type')
    plt.xlabel('Cell Type')
    plt.ylabel('Precision')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Accuracy by cell type plot saved to {save_path}")


def save_results(
    results: Dict[str, Any],
    output_dir: str = "evaluation_results"
) -> None:
    """Save evaluation results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save detailed results
    results_df = pd.DataFrame({
        'ground_truth': results['ground_truth'],
        'predictions': results['predictions'],
        'cell_sentence': results['cell_sentences']
    })
    results_df.to_csv(output_path / "detailed_results.csv", index=False)
    
    # Save classification report
    classification_df = pd.DataFrame(results['classification_report']).T
    classification_df.to_csv(output_path / "classification_report.csv")
    
    # Save summary
    summary = {
        'accuracy': results['accuracy'],
        'num_samples': results['num_samples'],
        'num_unique_labels': len(results['unique_labels']),
        'unique_labels': results['unique_labels']
    }
    
    with open(output_path / "summary.txt", 'w') as f:
        f.write("Cell Type Prediction Evaluation Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Accuracy: {summary['accuracy']:.4f}\n")
        f.write(f"Number of samples: {summary['num_samples']}\n")
        f.write(f"Number of unique labels: {summary['num_unique_labels']}\n")
        f.write(f"Unique labels: {', '.join(summary['unique_labels'])}\n")
    
    logger.info(f"Results saved to {output_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate C2S-Scale-Gemma cell type prediction")
    parser.add_argument(
        "--model_name",
        type=str,
        default="vandijklab/C2S-Scale-Gemma-2-27B",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to cell data (h5ad file or directory)"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        help="HuggingFace authentication token"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=20,
        help="Maximum new tokens for generation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--quantization",
        action="store_true",
        help="Use 4-bit quantization"
    )
    
    args = parser.parse_args()
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data_path}")
    dataset = CellSentenceDataset(
        data_path=args.data_path,
        tokenizer_name=args.model_name,
        max_seq_length=2048
    )
    
    # Load C2S-Scale-Gemma model
    logger.info(f"Loading C2S-Scale-Gemma model: {args.model_name}")
    
    quantization_config = None
    if args.quantization:
        quantization_config = {
            'load_in_4bit': True,
            'bnb_4bit_compute_dtype': torch.bfloat16,
            'bnb_4bit_use_double_quant': True,
            'bnb_4bit_quant_type': 'nf4'
        }
    
    model_loader = C2SScaleGemmaLoader(
        model_name=args.model_name,
        device=device,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        use_auth_token=args.hf_token
    )
    
    # Evaluate
    results = evaluate_cell_type_prediction(
        model_loader=model_loader,
        dataset=dataset,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens
    )
    
    # Save results
    save_results(results, args.output_dir)
    
    # Create plots
    plot_confusion_matrix(
        results['confusion_matrix'],
        results['unique_labels'],
        os.path.join(args.output_dir, "confusion_matrix.png")
    )
    
    plot_accuracy_by_cell_type(
        results['classification_report'],
        os.path.join(args.output_dir, "accuracy_by_cell_type.png")
    )
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
