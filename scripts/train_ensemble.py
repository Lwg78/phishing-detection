"""
Script to train ensemble models (Stacking, Voting).
Usage: python scripts/train_ensemble.py --ensemble stacking
"""
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE, OUTPUT_DIR
from src.data_loader import load_data
from src.preprocessing import clean_data, prepare_features
from src.feature_engineering import engineer_features
from src.models.ensemble import get_ensemble
from src.evaluation import (
    calculate_metrics, print_metrics, plot_confusion_matrix,
    plot_roc_curve, save_results
)
from src.utils import timer, get_timestamp


@timer
def train_ensemble_pipeline(ensemble_name: str, save_ensemble: bool = True):
    """
    Complete training pipeline for ensemble models.
    
    Args:
        ensemble_name: Name of ensemble (stacking, hard_voting, soft_voting, weighted_soft_voting)
        save_ensemble: Whether to save trained ensemble
    """
    print("\n" + "="*80)
    print(f"PHISHING DETECTION - ENSEMBLE TRAINING: {ensemble_name.upper()}")
    print("="*80 + "\n")
    
    # 1. Load processed data (assume it exists from individual training)
    print("STEP 1: Loading Processed Data")
    print("-" * 40)
    df = load_data(prefer_csv=True)
    
    # Clean if needed
    df = clean_data(df)
    
    # Engineer features if not already present
    if 'UTS' not in df.columns:
        print("  Engineering features...")
        df = engineer_features(df)
    
    # 2. Prepare features
    print("\nSTEP 2: Preparing Features")
    print("-" * 40)
    X, y = prepare_features(df)
    
    # 3. Train-test split
    print("\nSTEP 3: Train-Test Split")
    print("-" * 40)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Training set: {len(X_train):,} samples")
    print(f"  Test set: {len(X_test):,} samples")
    
    # 4. Train ensemble
    print(f"\nSTEP 4: Training {ensemble_name} Ensemble")
    print("-" * 40)
    print("  ⚠ This may take several minutes (training multiple models)...")
    
    ensemble = get_ensemble(ensemble_name)
    ensemble.train(X_train, y_train)
    
    # 5. Evaluate
    print(f"\nSTEP 5: Evaluating {ensemble_name}")
    print("-" * 40)
    
    y_pred = ensemble.predict(X_test)
    y_prob = ensemble.predict_proba(X_test)[:, 1]
    
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    print_metrics(metrics, model_name=ensemble_name.upper())
    
    # 6. Visualizations
    print(f"\nSTEP 6: Creating Visualizations")
    print("-" * 40)
    
    timestamp = get_timestamp()
    output_dir = OUTPUT_DIR / f"ensemble_{ensemble_name}" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cm_path = output_dir / f"{ensemble_name}_confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, model_name=ensemble_name.upper(),
                         output_path=cm_path)
    
    roc_path = output_dir / f"{ensemble_name}_roc_curve.png"
    plot_roc_curve(y_test, y_prob, model_name=ensemble_name.upper(),
                  output_path=roc_path)
    
    results = {ensemble_name: metrics}
    results_path = output_dir / f"{ensemble_name}_results.csv"
    save_results(results, output_path=results_path)
    
    # 7. Save ensemble
    if save_ensemble:
        print(f"\nSTEP 7: Saving Ensemble")
        print("-" * 40)
        ensemble.save()
    
    print("\n" + "="*80)
    print("ENSEMBLE TRAINING COMPLETE!")
    print("="*80 + "\n")
    
    return ensemble, metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train ensemble phishing detection model'
    )
    parser.add_argument(
        '--ensemble',
        type=str,
        required=True,
        choices=['stacking', 'hard_voting', 'soft_voting', 'weighted_soft_voting'],
        help='Ensemble method to train'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save trained ensemble'
    )
    
    args = parser.parse_args()
    
    train_ensemble_pipeline(args.ensemble, save_ensemble=not args.no_save)


if __name__ == "__main__":
    main()
