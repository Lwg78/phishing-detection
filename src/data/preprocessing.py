"""
Data preprocessing and cleaning utilities.
Based on your EDA notebook quality analysis.
"""
import pandas as pd
import numpy as np
from typing import Tuple


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset based on EDA findings:
    - Remove rows with impossible negative values
    - Handle missing values
    - Remove duplicates
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    initial_rows = len(df)
    print(f"Initial dataset: {initial_rows:,} rows")
    
    # 1. Remove duplicates
    if 'URL' in df.columns:
        df = df.drop_duplicates(subset=['URL'])
    else:
        df = df.drop_duplicates()
        print(f"  After removing duplicates: {len(df):,} rows")
    
    # Identify numeric columns (exclude URL and label)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Remove rows with negative values (impossible for count features)
    count_features = [col for col in numeric_cols if any(x in col.lower() for x in 
                     ['num_', 'count', 'length', 'noofurl', 'noofpopup', 'noofiframe'])]
    
    if count_features:
        for col in count_features:
            if (df[col] < 0).any():
                negative_count = (df[col] < 0).sum()
                df = df[df[col] >= 0]
                print(f"  Removed {negative_count} rows with negative {col}")
    
    # 2. Handle infinite values
    df = df.replace([np.inf, -np.inf], 0)

    # 3. Fill NaNs
    df = df.fillna(0)
    
    # Fill missing values (strategy from your EDA)
    # Numeric: fill with 0 (safer for count features)
    # Categorical: fill with mode or 'unknown'
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna('unknown')
    
    removed = initial_rows - len(df)
    print(f"✓ Cleaning complete: removed {removed:,} contaminated rows ({removed/initial_rows*100:.2f}%)")
    print(f"  Final dataset: {len(df):,} rows")
    
    return df


def prepare_features(df: pd.DataFrame, target_col: str = 'label') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for modeling.
    
    Args:
        df: DataFrame with all features
        target_col: Name of target column
        
    Returns:
        Tuple of (X, y) where X is features and y is target
        
    Prepares X (Features) and y (Target) for training.
    Removes non-training columns like 'URL'.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
        
    # 1. Separate Target
    y = df[target_col]
    
    # 2. Drop Metadata Columns (We don't train on the URL string itself)
    # We also drop 'source' or 'id' if they exist
    drop_cols = ['URL', 'url', target_col, 'id', 'source']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # 3. Ensure Numeric
    # If any non-numeric columns remain, we try to convert or drop them
    X = X.select_dtypes(include=[np.number])
    
    print(f"✓ Features prepared: {X.shape[1]} features, {len(y):,} samples")
    print(f"  Class distribution: Phishing={sum(y==1):,}, Legitimate={sum(y==0):,}")
    
    return X, y


def check_class_balance(y: pd.Series, threshold: float = 0.4) -> bool:
    """
    Check if classes are reasonably balanced.
    
    Args:
        y: Target series
        threshold: Minimum acceptable ratio of minority class
        
    Returns:
        True if balanced, False otherwise
    """
    class_counts = y.value_counts()
    minority_ratio = class_counts.min() / class_counts.sum()
    
    if minority_ratio < threshold:
        print(f"⚠ WARNING: Imbalanced dataset! Minority class: {minority_ratio*100:.1f}%")
        return False
    
    print(f"✓ Dataset is balanced: {minority_ratio*100:.1f}% minority class")
    return True
