"""
System Health Check & Data Integrity Test.
Runs the full pipeline against your actual text files to ensure compatibility.
"""
import sys
import pandas as pd
from pathlib import Path
import random

# Setup Project Root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import modules
from src.config import PROJECT_ROOT, WHITELIST_PATH
from src.utils import load_whitelist
from src.feature_engineering import extract_url_features
from src.models.predict_model import PhishingPredictor

def print_pass(msg):
    print(f"✅ PASS: {msg}")

def print_fail(msg):
    print(f"❌ FAIL: {msg}")

def print_info(msg):
    print(f"ℹ️  INFO: {msg}")

def test_file_inputs(predictor):
    """
    Reads your specific text files and tests a random sample from them.
    """
    print("\n" + "="*60)
    print("📂 TESTING USER DATA FILES")
    print("="*60)

    # --- FILE 1: my_test_urls.txt (Raw List) ---
    test_file = PROJECT_ROOT / "my_test_urls.txt"
    if test_file.exists():
        print_info(f"Found {test_file.name}")
        with open(test_file, 'r') as f:
            # Read non-empty lines
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        # Pick 3 random URLs to test (so we don't spam the console)
        sample = urls[:3] if len(urls) < 3 else random.sample(urls, 3)
        
        print(f"   Running predictions on {len(sample)} random URLs from file...")
        for url in sample:
            try:
                result = predictor.predict(url)
                print(f"   🔹 {url[:40]:<45} -> {result['status']} ({result['reason']})")
            except Exception as e:
                print_fail(f"Crash on URL {url}: {e}")
        print_pass("my_test_urls.txt processed successfully")
    else:
        print_info(f"Skipping {test_file.name} (File not found)")

    # --- FILE 2: validation_urls.txt (Labeled Data) ---
    val_file = PROJECT_ROOT / "validation_urls.txt"
    if val_file.exists():
        print_info(f"Found {val_file.name}")
        
        print("   Verifying accuracy on first 5 entries...")
        with open(val_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
        # Test first 5 lines
        for line in lines[:5]:
            try:
                # Parse "url, label" format
                parts = line.split(',')
                if len(parts) < 2: continue
                
                url = parts[0].strip()
                label = parts[1].strip().lower() # 'legitimate' or 'phishing'
                
                # Predict
                result = predictor.predict(url)
                predicted_status = result['status'].lower() # 'safe' or 'phishing'
                
                # Map 'safe' -> 'legitimate' for comparison
                mapped_pred = 'legitimate' if predicted_status == 'safe' else 'phishing'
                
                match = (mapped_pred == label)
                icon = "✅" if match else "⚠️"
                
                print(f"   {icon} {url[:30]:<35} | Exp: {label[:4]} | Got: {mapped_pred[:4]}")
                
            except Exception as e:
                print(f"   ⚠️ Could not parse line: {line}")

        print_pass("validation_urls.txt format check complete")
    else:
        print_info(f"Skipping {val_file.name} (File not found)")


def test_system():
    print("="*60)
    print("🚀 STARTING SYSTEM HEALTH CHECK")
    print("="*60)

    # TEST 1: Config
    if PROJECT_ROOT.exists():
        print_pass(f"Root: {PROJECT_ROOT}")
    else:
        print_fail("Project Root not found")

    # TEST 2: Whitelist
    whitelist = load_whitelist()
    if whitelist and 'exact_matches' in whitelist:
        print_pass(f"Whitelist Loaded: {len(whitelist['exact_matches'])} categories")
    else:
        print_fail("Whitelist failed to load")

    # TEST 3: Feature Engineering
    test_url = "http://dbs-login-secure.com"
    features = extract_url_features(test_url)
    if features['typo_similarity'] > 0:
        print_pass(f"Logic Check: Typosquatting detected for 'dbs'")
    else:
        print_fail("Logic Check: Typosquatting failed")
        
    if 'char_probability' in features:
         print_pass(f"Logic Check: Entropy Calculated ({features['char_probability']:.4f})")
    else:
         print_fail("Logic Check: Entropy missing")

    # TEST 4: Model Loading
    # We try to load 'xgboost_url_only' because it's fast. 
    # If not found, we fallback to 'xgboost' (full model)
    predictor = PhishingPredictor(model_name='xgboost_url_only')
    
    # If url_only failed, try the main model
    if not predictor.model:
        print("   'xgboost_url_only' not found, trying 'xgboost'...")
        predictor = PhishingPredictor(model_name='xgboost')

    if predictor.model:
        print_pass(f"Model Engine Online: {type(predictor.model)}")
        
        # --- RUN NEW FILE TESTS ---
        test_file_inputs(predictor)
        
    else:
        print("\n⚠️  WARNING: No trained model found (pkl file).")
        print("   The script checked for 'xgboost_url_only.pkl' and 'xgboost.pkl'.")
        print("   Please run: python scripts/train_url_only.py --model xgboost")

    print("\n" + "="*60)
    print("SYSTEM CHECK COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_system()
