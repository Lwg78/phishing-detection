"""
Model evaluation utilities: metrics, confusion matrix, ROC curves.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

from src.config import OUTPUT_DIR


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for ROC-AUC)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['roc_auc'] = 0.0
    
    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = "Model"):
    """Print metrics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"{model_name} Performance Metrics")
    print(f"{'='*60}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    
    if 'roc_auc' in metrics:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f} ({metrics['roc_auc']*100:.2f}%)")
    
    print(f"{'='*60}\n")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         model_name: str = "Model",
                         output_path: Optional[Path] = None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of model
        output_path: Path to save figure (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'],
                cbar_kws={'label': 'Count'})
    
    plt.title(f'{model_name} Confusion Matrix\nF1: {f1_score(y_true, y_pred):.4f}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Add FN/FP annotations
    tn, fp, fn, tp = cm.ravel()
    plt.text(0.5, -0.15, f'FN (Missed): {fn}', 
             ha='center', transform=plt.gca().transAxes, color='red', fontweight='bold')
    
    plt.tight_layout()
    
    # ✅ FIX: Use 'output_path' (the argument name), NOT 'save_path'
    if output_path:
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {output_path}")
    
    plt.show()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"✓ Confusion matrix saved to {save_path}")
    
    plt.close() # Close to free memory

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray,
                   model_name: str = "Model",
                   output_path: Optional[Path] = None):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        model_name: Name of model
        output_path: Path to save figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC={auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.5)')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{model_name} ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to {output_path}")
    
    plt.close()


def compare_models(results: Dict[str, Dict[str, float]],
                   output_path: Optional[Path] = None):
    """
    Create comparison visualization for multiple models.
    
    Args:
        results: Dictionary mapping model names to their metrics
        output_path: Path to save figure
    """
    df = pd.DataFrame(results).T
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar plot
    df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].plot(
        kind='bar', ax=axes[0], rot=45
    )
    axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_ylim([0.9, 1.0])
    axes[0].legend(loc='lower right')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Heatmap
    sns.heatmap(df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].T,
                annot=True, fmt='.4f', cmap='RdYlGn', ax=axes[1],
                vmin=0.9, vmax=1.0, cbar_kws={'label': 'Score'})
    axes[1].set_title('Performance Heatmap', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Model', fontsize=12)
    axes[1].set_ylabel('Metric', fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to {output_path}")
    
    plt.close()
    
    return df


def save_results(results: Dict, filename: str = "model_performance.csv"):
    """
    Save results to CSV.
    
    Args:
        results: Dictionary of model results
        output_path: Path to save CSV
    """
    df = pd.DataFrame([results])
    output_path = OUTPUT_DIR / filename
    
    # Append if exists, else write new
    if output_path.exists():
        df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        df.to_csv(output_path, index=False)
    
    print(f"✓ Metrics saved to {output_path}")
    
    return df
