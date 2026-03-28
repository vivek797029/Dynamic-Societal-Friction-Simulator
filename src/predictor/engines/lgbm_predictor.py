"""
LGBM Risk Predictor Engine — Layer 1
Uses LightGBM (same algorithm as ACLED CAST conflict alert system)
for numerical risk score prediction from socioeconomic indicators.

This is NOT generative AI — it's a gradient-boosted decision tree
that outputs exact probability scores.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from ..data.historical_cases import INDICATOR_METADATA


class LGBMRiskPredictor:
    """
    Gradient-Boosted Decision Tree model for friction risk prediction.
    Uses LightGBM if available, falls back to sklearn GradientBoosting.

    Input:  14 socioeconomic indicators (numerical features)
    Output: Risk score (0-100) + Risk level (LOW/MEDIUM/HIGH/CRITICAL)
    """

    def __init__(self, model_dir: str = "models/lgbm"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.risk_regressor = None       # Predicts risk score 0-100
        self.severity_classifier = None  # Predicts severity 1-5
        self.feature_names = list(INDICATOR_METADATA.keys())
        self.is_trained = False

        # LGBM hyperparameters (tuned for conflict prediction)
        self.lgbm_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "n_estimators": 500,
            "max_depth": 8,
            "min_child_samples": 10,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbose": -1
        }

    def _prepare_features(self, indicators: Dict) -> np.ndarray:
        """Convert indicator dict to feature vector."""
        feature_vec = []
        for name in self.feature_names:
            feature_vec.append(float(indicators.get(name, 0.0)))
        return np.array(feature_vec, dtype=np.float32).reshape(1, -1)

    def train(self, X: np.ndarray, y_risk: np.ndarray, y_severity: np.ndarray):
        """Train both regressor and classifier.

        Args:
            X: Feature matrix (n_samples, 14)
            y_risk: Risk scores 0-100
            y_severity: Severity labels 1-5
        """
        print(f"Training LGBM Risk Predictor on {X.shape[0]} samples...")
        print(f"Features: {len(self.feature_names)}")

        if HAS_LGBM:
            self._train_lgbm(X, y_risk, y_severity)
        elif HAS_SKLEARN:
            self._train_sklearn(X, y_risk, y_severity)
        else:
            raise RuntimeError("Neither lightgbm nor sklearn available")

        self.is_trained = True
        print("✅ LGBM Risk Predictor trained successfully!")

    def _train_lgbm(self, X, y_risk, y_severity):
        """Train using LightGBM."""
        # Risk score regressor
        self.risk_regressor = lgb.LGBMRegressor(**self.lgbm_params)
        self.risk_regressor.fit(
            X, y_risk,
            feature_name=self.feature_names
        )

        # Severity classifier
        clf_params = {**self.lgbm_params}
        clf_params["objective"] = "multiclass"
        clf_params["num_class"] = 6
        clf_params["metric"] = "multi_logloss"

        self.severity_classifier = lgb.LGBMClassifier(**clf_params)
        self.severity_classifier.fit(X, y_severity)

        # Cross-validation score
        cv_scores = cross_val_score(
            lgb.LGBMRegressor(**self.lgbm_params),
            X, y_risk, cv=5, scoring="neg_mean_squared_error"
        )
        rmse = np.sqrt(-cv_scores.mean())
        print(f"  Risk Predictor RMSE (5-fold CV): {rmse:.2f}")

    def _train_sklearn(self, X, y_risk, y_severity):
        """Fallback training using sklearn."""
        self.risk_regressor = GradientBoostingRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        self.risk_regressor.fit(X, y_risk)

        self.severity_classifier = GradientBoostingClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        self.severity_classifier.fit(X, y_severity)

        # Evaluate
        cv_scores = cross_val_score(
            GradientBoostingRegressor(n_estimators=300, max_depth=6),
            X, y_risk, cv=5, scoring="neg_mean_squared_error"
        )
        rmse = np.sqrt(-cv_scores.mean())
        print(f"  Risk Predictor RMSE (5-fold CV): {rmse:.2f}")

    def predict_risk(self, indicators: Dict) -> Dict:
        """Predict friction risk from socioeconomic indicators.

        Args:
            indicators: Dict of 14 socioeconomic indicators

        Returns:
            Dict with risk_score, risk_level, confidence, feature_importance
        """
        if not self.is_trained:
            # Use formula-based fallback
            return self._formula_predict(indicators)

        X = self._prepare_features(indicators)

        # Predict risk score
        risk_score = float(np.clip(self.risk_regressor.predict(X)[0], 0, 100))

        # Predict severity
        severity_probs = None
        severity_class = None
        if self.severity_classifier is not None:
            severity_probs = self.severity_classifier.predict_proba(X)[0]
            severity_class = int(self.severity_classifier.predict(X)[0])

        # Risk level
        risk_level = self._score_to_level(risk_score)

        # Feature importance
        importance = self._get_feature_importance(indicators)

        return {
            "risk_score": round(risk_score, 1),
            "risk_level": risk_level,
            "severity_class": severity_class,
            "severity_probabilities": {
                "low": float(severity_probs[1]) if severity_probs is not None else 0,
                "medium": float(severity_probs[2]) if severity_probs is not None else 0,
                "high": float(severity_probs[3]) if severity_probs is not None else 0,
                "very_high": float(severity_probs[4]) if severity_probs is not None else 0,
                "critical": float(severity_probs[5]) if severity_probs is not None and len(severity_probs) > 5 else 0
            },
            "top_risk_factors": importance[:5],
            "model_type": "LGBM" if HAS_LGBM else "GradientBoosting"
        }

    def _formula_predict(self, indicators: Dict) -> Dict:
        """GCRI-style weighted formula prediction (fallback when not trained)."""
        score = 0.0
        factors = []

        for key, meta in INDICATOR_METADATA.items():
            value = indicators.get(key, 0)
            weight = meta["weight"]
            threshold = meta.get("critical_threshold")
            direction = meta.get("direction", "higher_worse")

            if threshold is None:
                continue

            if direction == "higher_worse":
                # Higher = worse. ratio>1 means past threshold
                ratio = (value / threshold) if threshold > 0 else 0
                risk_contribution = np.clip(ratio, 0, 2.5)
            elif direction == "lower_worse":
                # Lower = worse. If value < threshold, risk increases
                if threshold < 0:
                    # Handle negative thresholds (like gdp_growth = -2)
                    risk_contribution = np.clip(abs(value) / abs(threshold), 0, 2.5) if value < 0 else 0.2
                else:
                    risk_contribution = np.clip(threshold / max(value, 0.01), 0, 2.5) if value < threshold else 0.3
            else:
                continue

            contribution = weight * risk_contribution * 100
            score += contribution
            factors.append({"factor": key, "value": round(float(value), 2),
                            "threshold": threshold, "contribution": round(float(contribution), 1)})

        # Normalize to 0-100
        total_weight = sum(m["weight"] for m in INDICATOR_METADATA.values() if m.get("critical_threshold"))
        max_score = total_weight * 100  # Score when everything at threshold
        risk_score = np.clip((score / max_score) * 100, 2, 98)
        factors.sort(key=lambda x: x["contribution"], reverse=True)

        return {
            "risk_score": round(float(risk_score), 1),
            "risk_level": self._score_to_level(risk_score),
            "severity_class": None,
            "severity_probabilities": {},
            "top_risk_factors": factors[:5],
            "model_type": "GCRI_Formula"
        }

    def _score_to_level(self, score: float) -> str:
        if score >= 80:
            return "CRITICAL"
        elif score >= 60:
            return "HIGH"
        elif score >= 40:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_feature_importance(self, indicators: Dict) -> List[Dict]:
        """Get feature importance ranking."""
        if HAS_LGBM and hasattr(self.risk_regressor, 'feature_importances_'):
            importances = self.risk_regressor.feature_importances_
        elif hasattr(self.risk_regressor, 'feature_importances_'):
            importances = self.risk_regressor.feature_importances_
        else:
            return self._formula_predict(indicators).get("top_risk_factors", [])

        # Pair with feature names
        pairs = list(zip(self.feature_names, importances))
        pairs.sort(key=lambda x: x[1], reverse=True)

        result = []
        for name, imp in pairs[:5]:
            meta = INDICATOR_METADATA.get(name, {})
            result.append({
                "factor": name,
                "importance": round(float(imp), 4),
                "value": indicators.get(name, 0),
                "threshold": meta.get("critical_threshold"),
                "contribution": round(float(imp) * 100, 1)
            })
        return result

    def save(self, filepath: str = None):
        """Save trained model."""
        if filepath is None:
            filepath = str(self.model_dir / "lgbm_risk_predictor.pkl")
        with open(filepath, "wb") as f:
            pickle.dump({
                "risk_regressor": self.risk_regressor,
                "severity_classifier": self.severity_classifier,
                "feature_names": self.feature_names,
                "is_trained": self.is_trained,
                "params": self.lgbm_params
            }, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str = None):
        """Load trained model."""
        if filepath is None:
            filepath = str(self.model_dir / "lgbm_risk_predictor.pkl")
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.risk_regressor = data["risk_regressor"]
        self.severity_classifier = data["severity_classifier"]
        self.feature_names = data["feature_names"]
        self.is_trained = data["is_trained"]
        print(f"Model loaded from {filepath}")
