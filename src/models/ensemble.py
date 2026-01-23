"""
Ensemble methods: Stacking, Soft/Hard Voting, Weighted Voting.
Based on your Ensemble Analysis notebook results.
"""
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from src.models.base_model import get_model, PhishingDetector
from src.config import MODEL_DIR, RANDOM_STATE

from .base_model import (
    PhishingDetector, XGBoostDetector, RandomForestDetector,
    LightGBMDetector, GradientBoostingDetector, XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE
)
from ..config import MODEL_DIR


class EnsembleDetector(PhishingDetector:
    """Base class for ensemble phishing detectors."""
    
    def __init__(self, name, method='soft_voting'):
        super().__init__(name)
        self.method = method
        self.estimators = self._get_default_estimators()
        self._build_ensemble()
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train ensemble model."""
        self.feature_names = X_train.columns.tolist()
        
        print(f"Training {self.ensemble_name}...")
        self.ensemble.fit(X_train, y_train)
        print(f"✓ {self.ensemble_name} trained successfully")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        return self.ensemble.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        return self.ensemble.predict_proba(X)

    def _get_default_estimators(self):
        """Creates the 'Dream Team' of models."""
        # We instantiate the classes to get their internal sklearn objects
        xgb = get_model('xgboost').model
        rf = get_model('random_forest').model
        lgb = get_model('lightgbm').model
        
        return [
            ('xgb', xgb),
            ('rf', rf),
            ('lgb', lgb)
        ]

    def _build_ensemble(self):
        if self.method == 'soft_voting':
            self.model = VotingClassifier(
                estimators=self.estimators,
                voting='soft',
                n_jobs=2
            )
        elif self.method == 'hard_voting':
            self.model = VotingClassifier(
                estimators=self.estimators,
                voting='hard',
                n_jobs=2
            )
        elif self.method == 'stacking':
            # Stacking uses a 'Final Estimator' (Logistic Regression) to judge the others
            self.model = StackingClassifier(
                estimators=self.estimators,
                final_estimator=LogisticRegression(),
                cv=5,
                n_jobs=2
            )
    
    def save(self, output_dir: Path = MODEL_DIR) -> None:
        """Save ensemble to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ensemble_path = output_dir / f"{self.ensemble_name}.pkl"
        
        with open(ensemble_path, 'wb') as f:
            pickle.dump(self.ensemble, f)
        
        print(f"✓ Ensemble saved to {ensemble_path}")
    
    def load(self, model_dir: Path = MODEL_DIR) -> None:
        """Load ensemble from disk."""
        model_dir = Path(model_dir)
        ensemble_path = model_dir / f"{self.ensemble_name}.pkl"
        
        if not ensemble_path.exists():
            raise FileNotFoundError(f"Ensemble not found: {ensemble_path}")
        
        with open(ensemble_path, 'rb') as f:
            self.ensemble = pickle.load(f)
        
        print(f"✓ Ensemble loaded from {ensemble_path}")

     # Factory for Ensembles
    def get_ensemble(method='soft_voting'):
        return EnsembleDetector(name=f"ensemble_{method}", method=method)

class HardVotingEnsemble(EnsembleDetector):
    """
    Hard Voting Ensemble - Simple majority vote.
    From your results: F1=0.9905
    """
    
    def __init__(self):
        super().__init__("hard_voting")
        
        # Build base estimators
        estimators = []
        
        if XGBOOST_AVAILABLE:
            estimators.append(('xgboost', XGBoostDetector().model))
        if LIGHTGBM_AVAILABLE:
            estimators.append(('lightgbm', LightGBMDetector().model))
        
        estimators.append(('rf', RandomForestDetector().model))
        estimators.append(('gb', GradientBoostingDetector().model))
        
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting='hard'
        )


class SoftVotingEnsemble(EnsembleDetector):
    """
    Soft Voting Ensemble - Average probabilities.
    From your results: F1=0.9930
    """
    
    def __init__(self):
        super().__init__("soft_voting")
        
        # Build base estimators
        estimators = []
        
        if XGBOOST_AVAILABLE:
            estimators.append(('xgboost', XGBoostDetector().model))
        if LIGHTGBM_AVAILABLE:
            estimators.append(('lightgbm', LightGBMDetector().model))
        
        estimators.append(('rf', RandomForestDetector().model))
        estimators.append(('gb', GradientBoostingDetector().model))
        
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft'
        )


class WeightedSoftVotingEnsemble(EnsembleDetector):
    """
    Weighted Soft Voting - Weighted average based on performance.
    From your results: F1=0.9931
    
    Weights based on F1 scores:
    - XGBoost: 0.9932
    - LightGBM: 0.9917
    - Random Forest: 0.9914
    - Gradient Boosting: 0.9827
    """
    
    def __init__(self, weights: Optional[List[float]] = None):
        super().__init__("weighted_soft_voting")
        
        # Default weights from your results (normalized)
        if weights is None:
            weights = [0.9932, 0.9917, 0.9914, 0.9827]  # XGB, LGB, RF, GB
            weights = [w / sum(weights) for w in weights]
        
        # Build base estimators
        estimators = []
        actual_weights = []
        
        if XGBOOST_AVAILABLE:
            estimators.append(('xgboost', XGBoostDetector().model))
            actual_weights.append(weights[0])
        if LIGHTGBM_AVAILABLE:
            estimators.append(('lightgbm', LightGBMDetector().model))
            actual_weights.append(weights[1])
        
        estimators.append(('rf', RandomForestDetector().model))
        actual_weights.append(weights[2])
        
        estimators.append(('gb', GradientBoostingDetector().model))
        actual_weights.append(weights[3])
        
        # Renormalize weights based on available models
        actual_weights = [w / sum(actual_weights) for w in actual_weights]
        
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=actual_weights
        )


class StackingEnsemble(EnsembleDetector):
    """
    Stacking Ensemble - Best performer (F1=0.9934)
    Uses Logistic Regression as meta-learner.
    """
    
    def __init__(self):
        super().__init__("stacking")
        
        # Build base estimators
        estimators = []
        
        if XGBOOST_AVAILABLE:
            estimators.append(('xgboost', XGBoostDetector().model))
        if LIGHTGBM_AVAILABLE:
            estimators.append(('lightgbm', LightGBMDetector().model))
        
        estimators.append(('rf', RandomForestDetector().model))
        estimators.append(('gb', GradientBoostingDetector().model))
        
        # Meta-learner (final estimator)
        meta_learner = LogisticRegression(max_iter=1000, random_state=42)
        
        self.ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,  # 5-fold cross-validation for meta-features
            n_jobs=-1
        )


def get_ensemble(ensemble_name: str) -> EnsembleDetector:
    """
    Factory function to get ensemble by name.
    
    Args:
        ensemble_name: Name of ensemble (stacking, hard_voting, soft_voting, weighted_soft_voting)
        
    Returns:
        EnsembleDetector instance
    """
    ensembles = {
        'stacking': StackingEnsemble,
        'hard_voting': HardVotingEnsemble,
        'soft_voting': SoftVotingEnsemble,
        'weighted_soft_voting': WeightedSoftVotingEnsemble
    }
    
    ensemble_name_lower = ensemble_name.lower().replace(' ', '_')
    
    if ensemble_name_lower not in ensembles:
        raise ValueError(f"Unknown ensemble: {ensemble_name}. Available: {list(ensembles.keys())}")
    
    return ensembles[ensemble_name_lower]()
