"""Validate model accuracy against labeled test URLs."""
"""
Validation Script.
Tests the FULL pipeline (Whitelist -> AI) against a list of known URLs.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix

from src.config import MODEL_DIR
from src.feature_engineering import extract_url_features
from src.models.predict_model import PhishingPredictor

# Read validation data
def run_validation():
    validation_file = PROJECT_ROOT / "validation_urls.txt"
    if not validation_file.exists():
        print("❌ No validation_urls.txt found. Please create one.")
        return
    
    # Load Predictor (It loads Whitelist & Model automatically)
    # We use the 'url_only' model for fast validation
    predictor = PhishingPredictor(model_name='xgboost_url_only')
    
    print(f"\n🔍 Starting Validation using: {predictor.model_name}")
    print("="*60)
    
    correct = 0
    total = 0

    with open(validation_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
            continue
        # Format: "http://google.com, legitimate"
            try:
                url, expected = line.split(',')
                expected = expected.strip().lower() # 'legitimate' or 'phishing'
            except ValueError:
                continue
            
        parts = line.strip().split(',')
        if len(parts) == 2:
            url, expected = parts
            data.append({'url': url.strip(), 'expected': expected.strip()})

df = pd.DataFrame(data)

print(f"Loaded {len(df)} validation URLs")
print(f"  Legitimate: {sum(df['expected'] == 'legitimate')}")
print(f"  Phishing: {sum(df['expected'] == 'phishing')}")

# Load model directly
model_name = 'xgboost_balanced'
model_path = MODEL_DIR / f"{model_name}.pkl"

if not model_path.exists():
    print(f"\n❌ Model not found: {model_path}")
    print("\nAvailable models:")
    for pkl_file in MODEL_DIR.glob("*.pkl"):
        print(f"  - {pkl_file.stem}")
    sys.exit(1)

print(f"\nLoading model: {model_path}")
with open(model_path, 'rb') as f:
    model = pickle.load(f)

print(f"✓ Model loaded: {type(model)}")

# Get expected features
if hasattr(model, 'feature_names_in_'):
    expected_features = list(model.feature_names_in_)
elif hasattr(model, 'get_booster'):
    expected_features = model.get_booster().feature_names
else:
    expected_features = None

print(f"  Expects {len(expected_features)} features" if expected_features else "  Feature names unknown")

# Get predictions
print("\nMaking predictions...")
predictions = []
probabilities = []

for idx, url in enumerate(df['url'], 1):
    print(f"  {idx}/{len(df)}: {url[:50]}...", end='')
    
    try:
        features = extract_url_features(url)
        df_feat = pd.DataFrame([features])
        
        # Ensure all expected features present
        if expected_features:
            # Add missing features with default 0
            for feat in expected_features:
                if feat not in df_feat.columns:
                    df_feat[feat] = 0
            
            # Reorder to match training
            df_feat = df_feat[expected_features]
            
# Run Prediction
            result = predictor.predict(url)
            verdict = result['status'].lower() # 'safe' or 'phishing'
            
            # Map 'safe' -> 'legitimate' for comparison
            normalized_verdict = 'legitimate' if verdict == 'safe' else 'phishing'
            
            is_correct = (normalized_verdict == expected)
            if is_correct: correct += 1
            total += 1
            
            # Print Result
            icon = "✅" if is_correct else "❌"
            print(f"{icon} Expect: {expected.upper():<10} | Got: {verdict.upper():<10} | {url}")
            if not is_correct:
                print(f"   Reason: {result['reason']}")

    print("="*60)
    print(f"🎯 Score: {correct}/{total} ({(correct/total)*100:.1f}%)")

if __name__ == "__main__":
    run_validation()
