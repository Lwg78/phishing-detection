"""
Configuration file with platform-independent paths.
Works on Windows, Linux, macOS, and CI/CD environments.
"""
import os
from pathlib import Path

# Auto-detect project root (works anywhere)
# Project Root (Calculated dynamically)
# If this file is in src/config.py, we go up one level to 'src', then one to 'root'
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# 1. Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"  # Standardized Name
OUTPUT_DIR = DATA_DIR / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"

# 2. Database and files
DB_FILE = RAW_DATA_DIR / "phishing_url.db"
PROCESSED_CSV = DATA_PROCESSED_DIR / "phishing_clean.csv"

# --- NEW ADDITION ---
WHITELIST_PATH = CONFIG_DIR / "whitelist.json"

# 3. Create directories if they don't exist
for directory in [RAW_DATA_DIR, DATA_PROCESSED_DIR, OUTPUT_DIR, MODEL_DIR, CONFIG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# 4. Model hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# 5. Feature engineering weights (For UTS Calculation)
# Derived from EDA importance analysis
UTS_WEIGHTS = {
    'url_length': 0.12,
    'num_dots': 0.08,
    'num_hyphens': 0.07,
    'num_at': 0.10,
    'has_ip': 0.13,
    'num_slashes': 0.05,
    'num_questionmarks': 0.07,
    'num_equals': 0.06,
    'NoOfURLRedirect': 0.10,
    'NoOfPopup': 0.10,
    'char_probability': 0.12
}

# Model configurations (from best results)
XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss',
    'n_jobs': -1
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

LIGHTGBM_PARAMS = {
    'n_estimators': 200,
    'max_depth': 10,
    'learning_rate': 0.1,
    'num_leaves': 50,
    'random_state': RANDOM_STATE,
    'verbose': -1,
    'n_jobs': -1
}
