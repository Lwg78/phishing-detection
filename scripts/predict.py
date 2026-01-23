"""
Real-time URL prediction script - FINAL FIXED VERSION.
Usage: 
    python scripts/predict.py --url "http://example.com" --model xgboost
    python scripts/predict.py --file urls.txt --model xgboost --output results.csv
"""
import joblib
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Setup project root to allow imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the "Smart Brain" we just built
from src.models.predict_model import PhishingPredictor

def predict_single_url(url: str, model_name: str = 'xgboost'):
    """Predicts status for a single URL using the central engine."""
    
    # Initialize the Predictor (Loads Whitelist + Rules + Model)
    # We use 'xgboost_url_only' by default if running from CLI for speed
    if model_name == 'xgboost':
        # Check if url_only exists, otherwise default to full model
        # This is a user convenience helper
        predictor = PhishingPredictor('xgboost_url_only')
        if not predictor.model:
            predictor = PhishingPredictor('xgboost')
    else:
        predictor = PhishingPredictor(model_name)

    print(f"\n🔍 Analyzing: {url}...")
    
    # Get Result
    result = predictor.predict(url)
    
    # Display
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    
    status = result['status']
    icon = "✅" if status == "SAFE" else "🚨"
    
    print(f"  URL: {url}")
    print(f"  Verdict: {icon} {status}")
    print(f"  Confidence: {result.get('confidence', 0):.1f}%")
    print(f"  Reason: {result['reason']}")
    
    if 'features' in result:
        risk = result['features'].get('typo_similarity', 0)
        print(f"  Typo Risk Score: {risk:.2f}")

    print("="*60 + "\n")

def predict_from_file(file_path: str, model_name: str = 'xgboost', output_file: str = None):
    """Batch prediction from text file."""
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return

    # Load URLs
    with open(file_path, 'r') as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"\n📂 Loaded {len(urls)} URLs from {file_path.name}")
    
    # Initialize Predictor ONCE (faster)
    predictor = PhishingPredictor('xgboost_url_only')
    if not predictor.model:
        print("   (Switching to full xgboost model)")
        predictor = PhishingPredictor('xgboost')

    results = []
    
    print("-" * 60)
    print(f"{'URL':<50} | {'VERDICT':<10} | {'REASON'}")
    print("-" * 60)

    for url in urls:
        res = predictor.predict(url)
        results.append({
            'url': url,
            'status': res['status'],
            'confidence': res.get('confidence', 0),
            'reason': res['reason'],
            'risk_score': res.get('features', {}).get('typo_similarity', 0)
        })
        
        # Print short summary
        short_url = (url[:47] + '..') if len(url) > 47 else url
        print(f"{short_url:<50} | {res['status']:<10} | {res['reason']}")

    # Save to CSV
    if output_file:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, help='Single URL to check')
    parser.add_argument('--file', type=str, help='File with list of URLs')
    parser.add_argument('--model', type=str, default='xgboost', help='Model name')
    parser.add_argument('--output', type=str, help='Output CSV file path')
    
    args = parser.parse_args()
    
    if args.url:
        predict_single_url(args.url, args.model)
    elif args.file:
        predict_from_file(args.file, args.model, args.output)
    else:
        print("Please provide --url or --file")

if __name__ == "__main__":
    main()
