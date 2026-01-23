"""
Main script to process raw data into training data.
Orchestrates: Loader -> Cleaner -> Feature Engineering -> Saver
"""
import pandas as pd
import numpy as np
import sqlite3
import os
from pathlib import Path
from tqdm import tqdm

# Import your custom feature extractor (which uses the whitelist!)
# Note: We use 'src' because this script is run from the project root
from src.config import PROCESSED_CSV, RAW_DATA_DIR
from src.data.data_loader import load_data
from src.data.preprocessing import clean_data
from src.feature_engineering import extract_url_features

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parents[2] # Go up: src/data/ -> src/ -> root
DB_PATH = PROJECT_ROOT / "data" / "raw" / "phishing_url.db"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "training_dataset.csv"

def main():
    print("🚀 Starting Data Processing Pipeline...")
    
    # 1. LOAD (Uses data_loader.py)
    # This automatically handles SQL vs CSV fallback logic
    df = load_data(prefer_csv=False)
    
    if df.empty:
        print("❌ No data found. Exiting.")
        return

    # 2. CLEAN (Uses preprocessing.py)
    # Removes duplicates, negative values, etc.
    df = clean_data(df)

    # 3. FEATURE ENGINEERING (Uses feature_engineering.py)
    print(f"⚙️ Extracting features for {len(df):,} URLs...")
    tqdm.pandas()
    
    # Apply your smart feature extractor
    feature_df = df['URL'].progress_apply(extract_url_features)
    
    # Combine Labels + Features
    # We drop 'URL' here because the model doesn't need the string, just the math
    final_df = pd.concat([df['label'], feature_df], axis=1)
    
    # 4. SANITIZE (Final Safety Check)
    # Replace Infinity or NaN that might occur during division
    final_df = final_df.replace([np.inf, -np.inf], 0).fillna(0)

    # 5. SAVE
    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(PROCESSED_CSV, index=False)
    print(f"✅ Saved processed dataset to: {PROCESSED_CSV}")
    print(f"   Shape: {final_df.shape}")

if __name__ == "__main__":
    main()
    
    
