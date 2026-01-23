import sys
import pandas as pd
import sqlite3
import numpy as np
from tqdm import tqdm
from pathlib import Path
import random

# Setup Project Root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROJECT_ROOT, WHITELIST_PATH
from src.utils import load_whitelist
from src.feature_engineering import extract_url_features
from src.models.predict_model import PhishingPredictor


logger = get_logger("Trainer")

def load_data():
    logger.info(f"Connecting to database at {PROJECT_ROOT}...")
    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT URL as url, label FROM phishing_url" 
        df = pd.read_sql_query(query, conn)
        conn.close()
        logger.info(f"Successfully loaded {len(df)} URLs.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    logger.info("Extracting features (This uses the NEW ML logic)...")
    tqdm.pandas()
    
    feature_df = df['url'].progress_apply(extract_features)
    feature_df['label'] = df['label']
    feature_df = calculate_uts(feature_df)
    feature_df = feature_df.fillna(0)
    
    return feature_df

def main():
    raw_df = load_data()
    if raw_df is None: return

    processed_df = preprocess_data(raw_df)
    
    y = processed_df['label']
    X = processed_df.drop('label', axis=1)
    
    # NOTE: We keep 'has_https' now. 
    # With the new 'typo_similarity' feature, the model is smart enough
    # to know that HTTPS doesn't automatically mean Safe.
        
    logger.info(f"Training on {X.shape[1]} features.")
    
    ensemble = PhishingEnsemble()
    ensemble.train(X, y)
    ensemble.save_model()
    
    logger.info("🎉 Machine Learning Model Updated!")

if __name__ == "__main__":
    main()
