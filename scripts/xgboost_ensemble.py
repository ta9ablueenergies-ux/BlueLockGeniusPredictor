"""
XGBoost ensemble model for football prediction.

This module implements an XGBoost model that can be used in ensemble with the neural model
to improve prediction accuracy as specified in the CODEX roadmap.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
import joblib
import os

class XGBoostEnsemble:
    def __init__(self, model_dir="model/v11"):
        self.model_dir = model_dir
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None

    def prepare_features(self, df):
        """Prepare features for XGBoost training."""
        # Select numerical features from the dataframe
        feature_cols = [
            'B365H', 'B365D', 'B365A',  # Bookmaker odds
            'HY', 'AY', 'HR', 'AR', 'HC', 'AC',  # Match statistics
            'HS', 'AS', 'HST', 'AST', 'HF', 'AF',  # More match statistics
            'HPoss', 'APoss', 'HXG', 'AXG',  # Advanced metrics
            'h_yc_avg5', 'a_yc_avg5', 'h_cw_avg5', 'a_cw_avg5'  # Averages
        ]

        # Filter to only existing columns
        available_cols = [col for col in feature_cols if col in df.columns]

        # Create feature matrix
        X = df[available_cols].copy()

        # Add engineered features
        if 'B365H' in X.columns and 'B365D' in X.columns and 'B365A' in X.columns:
            # Implied probabilities from odds
            X['impl_prob_h'] = 1.0 / X['B365H']
            X['impl_prob_d'] = 1.0 / X['B365D']
            X['impl_prob_a'] = 1.0 / X['B365A']
            X['overround'] = X['impl_prob_h'] + X['impl_prob_d'] + X['impl_prob_a']

        return X

    def fit(self, X_train, y_train):
        """Train the XGBoost model."""
        # Prepare features
        X = self.prepare_features(X_train)

        # Handle missing values
        X = X.fillna(0)

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Train XGBoost model
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        self.model.fit(X_scaled, y_train)

        # Save the trained model
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(self.model_dir, 'xgboost_ensemble_model.pkl'))
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'xgboost_scaler.pkl'))

        return self

    def predict_proba(self, X):
        """Make probability predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")

        # Prepare features
        X_features = self.prepare_features(X)
        X_features = X_features.fillna(0)

        # Standardize features
        X_scaled = self.scaler.transform(X_features)

        # Get predictions
        return self.model.predict_proba(X_scaled)

    def save_model(self, filepath):
        """Save the trained model and scaler."""
        joblib.dump(self.model, f"{filepath}_model.pkl")
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")

    def load_model(self, filepath):
        """Load a pre-trained model and scaler."""
        self.model = joblib.load(f"{filepath}_model.pkl")
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        return self

def create_ensemble_features(neural_probs, xgboost_probs, y_true):
    """
    Create ensemble features by combining neural and XGBoost predictions.

    Args:
        neural_probs: probabilities from neural model (N, 3)
        xgboost_probs: probabilities from XGBoost model (N, 3)
        y_true: true labels (N,)

    Returns:
        features: combined features for meta-learner
    """
    # Combine the probability predictions from both models
    features = np.hstack([neural_probs, xgboost_probs])
    return features

def train_meta_learner(ensemble_features, y_true):
    """
    Train a meta-learner to combine predictions from neural and XGBoost models.

    Args:
        ensemble_features: combined features from both models
        y_true: true labels

    Returns:
        meta_model: trained meta-learner
    """
    # Use logistic regression as meta-learner
    from sklearn.linear_model import LogisticRegression

    meta_model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        random_state=42
    )

    meta_model.fit(ensemble_features, y_true)
    return meta_model

# Example usage:
# xgb_model = XGBoostEnsemble()
# xgb_model.fit(X_train, y_train)
# predictions = xgb_model.predict_proba(X_test)