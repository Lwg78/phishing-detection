"""
Script to train individual models (XGBoost, Random Forest, LightGBM).
Usage: python scripts/train_individual.py --model xgboost
"""
import sys
import argparse
import pandas as pd
from pathlib import Path

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split

from src.config import MODEL_DIR, RANDOM_STATE, TEST_SIZE, OUTPUT_DIR
from src.data.data_loader import load_data, save_processed_data
from src.data.preprocessing import clean_data, prepare_features, check_class_balance
from src.models.base_model import get_model
from src.models.evaluation import (
    calculate_metrics, print_metrics, plot_confusion_matrix,
    plot_roc_curve, save_results
)
from src.utils import timer, setup_logging, get_timestamp


def train_pipeline(model_name: str, save_model: bool = True):
    """
    Complete training pipeline for individual models.
    
    Args:
        model_name: Name of model to train (xgboost, random_forest, lightgbm, etc.)
        save_model: Whether to save trained model
    """
    print("\n" + "="*80)
    print(f"PHISHING DETECTION - 🚀 STARTING TRAINING: {model_name.upper()}")
    print("="*80 + "\n")
    
    # Setup logging
    logger = setup_logging()
    
    # 1. Load data
    print("STEP 1: Loading Data")
    print("-" * 40)
    df = load_data(prefer_csv=True)
    
    # 2. Clean data
    print("\nSTEP 2: Cleaning Data")
    print("-" * 40)
    df = clean_data(df)
    
    # Save processed data
    save_processed_data(df)
    
    # 3. Prepare features and target
    print("\nSTEP 3: Preparing Features")
    print("-" * 40)
    X, y = prepare_features(df)
    check_class_balance(y)
    
    # 4. Train-test split
    print("\nSTEP 4: Train-Test Split")
    print("-" * 40)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Training set: {len(X_train):,} samples")
    print(f"  Test set: {len(X_test):,} samples")
    
    # 5. Initialize Model
    print(f"\nSTEP 5: Initializing {model_name.upper()} model")
    print("-" * 40)
    detector = get_model(model_name)
    
    # 6. Train
    print(f"\nSTEP 6: 🧠 Training {model_name.upper()} on {len(X_train):,} samples...")
    print("-" * 40)
    detector.fit(X_train, y_train)
    
    # 7. Evaluate
    print("\nSTEP 7: 📊 Evaluating...")
    print("-" * 40)
    y_pred = detector.predict(X_test)
    y_prob = detector.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    print_metrics(metrics, model_name=model_name.upper())
    
    # 8. Visualizations
    print(f"\nSTEP 8: Creating Visualizations")
    print("-" * 40)
    
    timestamp = get_timestamp()
    output_dir = OUTPUT_DIR / model_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Confusion matrix
    cm_path = output_dir / f"{model_name}_confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, model_name=model_name.upper(), 
                         output_path=cm_path)
    
    # ROC curve
    roc_path = output_dir / f"{model_name}_roc_curve.png"
    plot_roc_curve(y_test, y_prob, model_name=model_name.upper(), 
                  output_path=roc_path)
    
    # Save metrics
    results = {model_name: metrics}
    results_path = output_dir / f"{model_name}_results.csv"
    save_results(results, output_path=results_path)
    
    # 9. Save model
    if save_model:
        print(f"\nSTEP 9: Saving Model")
        print("-" * 40)
        detector.save()
    
    print("\n✅ TRAINING COMPLETE.")
    
    return detector, metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train individual phishing detection model'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['xgboost', 'random_forest', 'lightgbm', 'gradient_boosting',
                'logistic_regression', 'decision_tree'],
        help='Model to train'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save trained model'
    )
    
    args = parser.parse_args()
    
    train_pipeline(args.model, save_model=not args.no_save)


if __name__ == "__main__":
    main()
