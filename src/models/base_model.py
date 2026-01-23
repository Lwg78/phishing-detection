"""
Individual ML models: XGBoost, Random Forest, LightGBM.
Based on your ML Model Selection notebook results.
"""
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from src.config import MODEL_DIR, RANDOM_STATE

try:
    from xgboost import XGBClassifier
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠ XGBoost not installed. Install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠ LightGBM not installed. Install with: pip install lightgbm")

from ..config import XGBOOST_PARAMS, RANDOM_FOREST_PARAMS, LIGHTGBM_PARAMS, MODEL_DIR


class PhishingDetector:
    """Base class for phishing detection models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              scale_features: bool = False) -> None:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            scale_features: Whether to scale features
        """
        self.feature_names = X_train.columns.tolist()
        
        if scale_features:
            X_train = self.scaler.fit_transform(X_train)
        
        print(f"Training {self.model_name}...")
        self.model.fit(X_train, y_train)
        print(f"✓ {self.model_name} trained successfully")

    def fit(self, X, y):
        """Train the model."""
        if hasattr(X, "columns"):
            self.feature_names = list(X.columns)
        self.model.fit(X, y)
        
    def predict(self, X: pd.DataFrame, scale_features: bool = False) -> np.ndarray:
        """Predict class labels."""
        if scale_features:
            X = self.scaler.transform(X)
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame, scale_features: bool = False) -> np.ndarray:
        """Predict class probabilities."""
        if scale_features:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)
    
    def save(self, output_dir: Path = MODEL_DIR) -> None:
        """Save model to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / f"{self.model_name}.pkl"
        scaler_path = output_dir / f"{self.model_name}_scaler.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"✓ Model saved to {model_path}")
    
    def load(self, model_dir: Path = MODEL_DIR) -> None:
        """Load model from disk."""
        model_dir = Path(model_dir)
        model_path = model_dir / f"{self.model_name}.pkl"
        scaler_path = model_dir / f"{self.model_name}_scaler.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        print(f"✓ Model loaded from {model_path}")
        
# --- IMPLEMENTATIONS ---

class XGBoostDetector(PhishingDetector):
    """XGBoost model - Best performer (F1=0.9932)"""
    
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("xgboost")
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed")
        
        if params is None:
            params = XGBOOST_PARAMS
        
        self.model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=RANDOM_STATE,
            n_jobs=2  # Limit cores to prevent Mac crash
        )


class RandomForestDetector(PhishingDetector):
    """Random Forest model - Third best (F1=0.9914)"""
    
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("random_forest")
        
        if params is None:
            params = RANDOM_FOREST_PARAMS
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=RANDOM_STATE,
            n_jobs=2
        )


class LightGBMDetector(PhishingDetector):
    """LightGBM model - Second best (F1=0.9917)"""
    
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("lightgbm")
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed")
        
        if params is None:
            params = LIGHTGBM_PARAMS
        
        self.model = LGBMClassifier(
            random_state=RANDOM_STATE,
            n_jobs=2,
            force_col_wise=True
        )


class GradientBoostingDetector(PhishingDetector):
    """Gradient Boosting model"""
    
    def __init__(self):
        super().__init__("gradient_boosting")
        
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )


class LogisticRegressionDetector(PhishingDetector):
    """Logistic Regression - Baseline model"""
    
    def __init__(self):
        super().__init__("logistic_regression")
        
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )


class DecisionTreeDetector(PhishingDetector):
    """Decision Tree model"""
    
    def __init__(self):
        super().__init__("decision_tree")
        
        self.model = DecisionTreeClassifier(
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )


def get_model(model_name: str) -> PhishingDetector:
    """
    Factory function to get model by name.
    
    Args:
        model_name: Name of model (xgboost, random_forest, lightgbm, etc.)
        
    Returns:
        PhishingDetector instance
    """
    models = {
        'xgboost': XGBoostDetector,
        'random_forest': RandomForestDetector,
        'lightgbm': LightGBMDetector,
        'gradient_boosting': GradientBoostingDetector,
        'logistic_regression': LogisticRegressionDetector,
        'decision_tree': DecisionTreeDetector
    }
    
    model_name_lower = model_name.lower().replace(' ', '_')
    
    if model_name_lower not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name_lower]()
