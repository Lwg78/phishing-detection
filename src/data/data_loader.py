"""
Data loading utilities that work with SQLite and CSV.
Platform-independent and CI/CD friendly.
"""
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Union
from src.config import DB_FILE, PROCESSED_CSV


def load_from_sqlite(db_path: Union[str, Path] = DB_FILE) -> pd.DataFrame:
    """
    Load data from SQLite database.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        DataFrame with phishing URL data
    """
    db_path = Path(db_path)
    
    if not db_path.exists():
        raise FileNotFoundError(f"❌ Database not found: {db_path}")
    
    try:
        conn = sqlite3.connect(str(db_path))
        df = pd.read_sql_query("SELECT URL, label FROM phishing_url", conn)
        conn.close()
        
        print(f"✓ Loaded {len(df):,} rows from database")
        return df
        
    except Exception as e:
        raise RuntimeError(f"❌ Error loading database: {e}")


def load_from_csv(csv_path: Union[str, Path] = PROCESSED_CSV) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with phishing URL data
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"❌ CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df):,} rows from CSV")
    return df


def load_data(prefer_csv: bool = False) -> pd.DataFrame:
    """
    Smart data loader: tries CSV first (if prefer_csv=True), falls back to SQLite.
    
    Args:
        prefer_csv: If True, try CSV first; otherwise try SQLite first
        
    Returns:
        DataFrame with phishing URL data
    """
    if prefer_csv:
        try:
            return load_from_csv()
        except FileNotFoundError:
            print("⚠️ CSV not found, loading from database...")
            return load_from_sqlite()
    else:
        try:
            return load_from_sqlite()
        except FileNotFoundError:
            print("⚠️ Database not found, loading from CSV...")
            return load_from_csv()


def save_processed_data(df: pd.DataFrame, output_path: Union[str, Path] = PROCESSED_CSV):
    """
    Save processed data to CSV.
    
    Args:
        df: DataFrame to save
        output_path: Path to save CSV
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"✓ Saved {len(df):,} rows to {output_path}")
