"""
Utility functions for logging, timing, and file operations.
"""
import time
import logging
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, Any
from functools import wraps
from src.config import WHITELIST_PATH

def setup_logging(log_name='phishing_detection'):
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory to save log files (optional)
        level: Logging level
    """
    # Create logger
    logger = logging.getLogger('phishing_detection')
    logger.setLevel(eval)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(eval)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_dir provided)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'phishing_detection_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
        
        print(f"✓ Logging to: {log_file}")
    
    return logger


def timer(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Usage:
        @timer
        def my_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        print(f"⏱  {func.__name__} took {elapsed:.2f} seconds")
        return result
    
    return wrapper

def load_whitelist():
    """
    Loads the whitelist JSON configuration.
    Returns a dictionary. Returns empty dict if file not found.
    """
    if not os.path.exists(WHITELIST_PATH):
        print(f"⚠️ WARNING: Whitelist not found at {WHITELIST_PATH}")
        return {}

    try:
        with open(WHITELIST_PATH, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ ERROR: Failed to parse whitelist.json: {e}")
        return {}
        

def format_number(num: int) -> str:
    """Format large numbers with commas."""
    return f"{num:,}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage."""
    return f"{value * 100:.{decimals}f}%"


def check_gpu_availability():
    """Check if GPU is available for XGBoost/LightGBM."""
    try:
        import xgboost as xgb
        gpu_available = xgb.XGBClassifier(tree_method='gpu_hist', n_estimators=1)
        print("✓ GPU available for XGBoost")
        return True
    except Exception:
        print("ℹ GPU not available, using CPU")
        return False


def memory_usage_mb():
    """Get current memory usage in MB."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    return mem_mb


class ProgressTracker:
    """Simple progress tracker for long operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, increment: int = 1):
        """Update progress."""
        self.current += increment
        progress = self.current / self.total
        elapsed = time.time() - self.start_time
        
        if self.current == self.total:
            print(f"\r{self.description}: 100% ({self.total}/{self.total}) - "
                  f"Completed in {elapsed:.2f}s")
        else:
            eta = elapsed / progress - elapsed if progress > 0 else 0
            print(f"\r{self.description}: {progress*100:.1f}% "
                  f"({self.current}/{self.total}) - ETA: {eta:.1f}s", end='')
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        if self.current < self.total:
            print()  # New line if not completed


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if not."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp() -> str:
    """Get formatted timestamp string."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')
