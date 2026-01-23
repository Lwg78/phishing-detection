"""
Train models using ONLY URL features (no webpage-specific features).
This enables real-time prediction without visiting the webpage.
"""
import sys
import pandas as pd
import numpy as np
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm  

# Fix path to find 'src'
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from sklearn.model_selection import train_test_split

from src.config import MODEL_DIR, RAW_DATA_DIR, RANDOM_STATE, TEST_SIZE, OUTPUT_DIR
from src.data.data_loader import load_data
from src.data.preprocessing import clean_data, check_class_balance
from src.feature_engineering import extract_url_features
from src.models.base_model import get_model
from src.models.evaluation import (
    calculate_metrics, print_metrics, plot_confusion_matrix,
    plot_roc_curve, save_results
)
from src.utils import timer, get_timestamp


def train_url_only_pipeline(model_name: str, save_model: bool = True):
    """
    Train model using ONLY URL-extractable features.
    """
    print("\n" + "="*80)
    print(f"\n🚀 TRAINING WITH URL FEATURES ONLY: {model_name.upper()}")
    print("="*80 + "\n")
    
    # 1. Load data
    print("STEP 1: Loading Data")
    print("-" * 40)
    df = load_data(prefer_csv=True)

    if df.empty:
        print("❌ No data found. Please run 'make_dataset.py' or check database.")
        return
    
    # 2. Clean data
    print("\nSTEP 2: Cleaning Data")
    print("-" * 40)
    df = clean_data(df)
    
    # 3. Feature engineering (URL features only)
    print("\nSTEP 3: Feature Engineering (URL Features Only)")
    print("-" * 40)
    
    # Check if features already extracted
    # if 'UTS' not in df.columns:
    #   df = extract_url_features(df)
    # ❌ WRONG: df = extract_url_features(df)
    # ✅ RIGHT: Apply the function to the 'URL' column row-by-row
    
    tqdm.pandas(desc="Extracting Features")
    
    try:
        # This creates a new DataFrame of features (length, entropy, etc.)
        feature_df = df['URL'].progress_apply(extract_url_features)
        
        # Combine the original Label (0/1) with the new Features
        # We DROP the original 'URL' string because models can't read text
        final_df = pd.concat([df['label'], feature_df], axis=1)
        
        # Sanitize (Fill NaNs)
        final_df = final_df.fillna(0)
        
    except Exception as e:
        print(f"❌ Error during feature extraction: {e}")
        return

    print(f"✓ Extracted {final_df.shape[1]-1} features for {len(final_df)} URLs")

    # 4. Select ONLY URL Features (Drop content-based features)
    # This list must match what 'extract_url_features' produces in feature_engineering.py
    print("\nSTEP 4: Selecting URL Features")
    print("-" * 40)
    
    url_features = [
        'url_length', 'domain_length', 'num_dots', 'num_hyphens', 'num_at', 
        'num_slashes', 'num_questionmarks', 'num_equals', 'num_ampersand', 
        'num_underscores', 'num_hash', 'num_percent', 'has_ip', 'has_https', 
        'char_probability', 'digit_letter_ratio', 'typo_similarity', 'is_known_brand'
    ]
    
    # Keep only URL features + label
    available_features = [f for f in url_features if f in df.columns]
    X = df[available_features]
    y = df['label'] if 'label' in df.columns else df['Label']
    
    print(f"⚙️ Filtering for {len(available_feats)} URL-specific features...")
    X = df[available_feats]
    y = df['label']

    check_class_balance(y)
    
    # 5. Train-test split
    print("\nSTEP 5: Train-Test Split")
    print("-" * 40)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Training set: {len(X_train):,} samples")
    print(f"  Test set: {len(X_test):,} samples")
    
    # 6. Train model
    print(f"\nSTEP 6: Training {model_name} (URL Features Only)")
    print("-" * 40)
    detector = get_model(model_name)
    detector.fit(X_train, y_train)
    
    # 7. Evaluate
    print(f"\nSTEP 7: Evaluating {model_name}")
    print("-" * 40)
    
    y_pred = detector.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    print(f"   [URL-Only] Accuracy: {metrics['accuracy']:.2%}")
    
    # 8. Visualizations
    print(f"\nSTEP 8: Creating Visualizations")
    print("-" * 40)
    
    timestamp = get_timestamp()
    output_dir = OUTPUT_DIR / f"{model_name}_url_only" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cm_path = output_dir / f"{model_name}_url_only_confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, 
                         model_name=f"{model_name.upper()} (URL Only)",
                         output_path=cm_path)
    
    roc_path = output_dir / f"{model_name}_url_only_roc_curve.png"
    plot_roc_curve(y_test, y_prob,
                  model_name=f"{model_name.upper()} (URL Only)",
                  output_path=roc_path)
    
    results = {f"{model_name}_url_only": metrics}
    results_path = output_dir / f"{model_name}_url_only_results.csv"
    save_results(results, output_path=results_path)
    
    # 9. Save model with special name
    if save_model:
        print(f"\nSTEP 9: Saving Model")
        print("-" * 40)
        output_path = MODEL_DIR / f"{model_name}_url_only.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(detector.model, f)
        
    print(f"\n✅ SUCCESS: Model saved to {output_path}")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n✓ Model trained on {len(available_features)} URL features only")
    print(f"✓ Can now predict without visiting webpages")
    print(f"✓ Use: python scripts/predict.py --url <url> --model {model_name}_url_only\n")
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser(
        description='Train phishing detection model using URL features only'
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
    
    train_url_only_pipeline(args.model, save_model=not args.no_save)


if __name__ == "__main__":
    train_url_only_pipeline('xgboost')
    
