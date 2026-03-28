"""
DSFS Core Predictor — Main Orchestrator
Connects all 6 engine layers into a unified prediction pipeline.

Architecture:
  Input Conditions → LGBM Risk Predictor → CNN-LSTM Escalation
  → Historical Matcher → Confidence Scorer → What-If Engine
  → Policy Engine → Final Output

This is NOT generative AI. The core prediction uses gradient-boosted
decision trees (LGBM), momentum-based escalation formulas, cosine
similarity matching, and weighted risk scoring.
"""

import json
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

from .engines.lgbm_predictor import LGBMRiskPredictor
from .engines.cnn_lstm_engine import EscalationPredictor
from .engines.historical_matcher import HistoricalCaseMatcher
from .engines.confidence_scorer import ConfidenceScorer
from .engines.whatif_engine import WhatIfEngine
from .engines.policy_engine import PolicyEngine
from .data.data_generator import FrictionDataGenerator


class DSFSPredictor:
    """
    Dynamic Society Friction Simulator — Core Prediction Engine

    A hybrid ML system that predicts societal friction risk by combining:
    1. LGBM (gradient-boosted trees) for risk scoring
    2. CNN-LSTM for temporal escalation prediction
    3. SBERT + FAISS for historical case matching
    4. GCRI-weighted confidence scoring
    5. Counterfactual what-if analysis
    6. Rule-based policy recommendations

    Usage:
        predictor = DSFSPredictor()
        predictor.train()
        result = predictor.predict({
            "unemployment_rate": 12.0,
            "inflation_rate": 8.5,
            ...
        })
    """

    VERSION = "2.0.0"

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Initialize all engines
        print("Initializing DSFS Predictor v2.0...")
        self.lgbm = LGBMRiskPredictor(str(self.model_dir / "lgbm"))
        self.escalation = EscalationPredictor(str(self.model_dir / "cnn_lstm"))
        self.matcher = HistoricalCaseMatcher()
        self.confidence = ConfidenceScorer()
        self.whatif = WhatIfEngine()
        self.policy = PolicyEngine()
        self.data_gen = FrictionDataGenerator()

        self.is_trained = False
        print("✅ All engines initialized")

    def train(self, n_samples: int = 10000):
        """Train the LGBM risk predictor on synthetic data."""
        print(f"\n{'='*60}")
        print("  DSFS TRAINING PIPELINE")
        print(f"{'='*60}")

        # Generate training data
        print(f"\n[1/3] Generating {n_samples} training samples...")
        X, y_risk, y_severity = self.data_gen.generate_feature_matrix(n_samples)
        print(f"  Features shape: {X.shape}")
        print(f"  Risk scores range: {y_risk.min():.1f} - {y_risk.max():.1f}")
        print(f"  Severity classes: {np.unique(y_severity)}")

        # Train LGBM
        print(f"\n[2/3] Training LGBM Risk Predictor...")
        self.lgbm.train(X, y_risk, y_severity)

        # Save model
        print(f"\n[3/3] Saving model...")
        self.lgbm.save()
        self.is_trained = True

        print(f"\n{'='*60}")
        print("  ✅ TRAINING COMPLETE")
        print(f"{'='*60}")

    def predict(self, indicators: Dict, description: str = "",
                include_whatif: bool = True,
                include_policy: bool = True) -> Dict:
        """Run full prediction pipeline.

        Args:
            indicators: Dict of 14 socioeconomic indicators
            description: Optional text description of situation
            include_whatif: Include what-if intervention analysis
            include_policy: Include policy recommendations

        Returns:
            Comprehensive prediction result with all layers
        """
        timestamp = datetime.now().isoformat()

        # ─── Layer 1: LGBM Risk Prediction ───────────────────
        risk_result = self.lgbm.predict_risk(indicators)
        risk_score = risk_result["risk_score"]

        # ─── Layer 2: Escalation Prediction ───────────────────
        escalation_result = self.escalation.predict_escalation(indicators, risk_score)
        cascade_timeline = self.escalation.generate_cascade_timeline(indicators, risk_score)

        # ─── Layer 3: Historical Matching ─────────────────────
        historical_matches = self.matcher.find_similar(indicators, description, top_k=3)

        # ─── Layer 4: Confidence Scoring ──────────────────────
        confidence_result = self.confidence.calculate_confidence(
            indicators, risk_score, historical_matches, escalation_result
        )

        # ─── Layer 5: What-If Analysis ────────────────────────
        whatif_results = None
        if include_whatif:
            recommended = self.whatif.recommend_interventions(indicators, risk_score)
            top_interventions = []
            for rec in recommended[:3]:
                analysis = self.whatif.analyze_intervention(
                    indicators, rec["intervention_id"], risk_score
                )
                top_interventions.append(analysis)
            whatif_results = {
                "recommended_interventions": recommended[:5],
                "top_3_analysis": top_interventions,
                "available_interventions": self.whatif.get_available_interventions()
            }

        # ─── Layer 6: Policy Recommendations ──────────────────
        policy_results = None
        if include_policy:
            policy_results = self.policy.generate_recommendations(
                indicators, risk_score,
                risk_result.get("top_risk_factors", []),
                historical_matches
            )

        # ─── Compile Final Output ─────────────────────────────
        output = {
            "metadata": {
                "version": self.VERSION,
                "timestamp": timestamp,
                "model_type": "Hybrid ML (LGBM + CNN-LSTM + SBERT + GCRI)",
                "is_generative_ai": False,
                "engines_used": [
                    "LightGBM Risk Predictor",
                    "CNN-LSTM Escalation Engine",
                    "SBERT Historical Matcher",
                    "GCRI Confidence Scorer",
                    "Counterfactual What-If Engine",
                    "Policy Recommendation Engine"
                ]
            },
            "input": {
                "indicators": indicators,
                "description": description
            },
            "prediction": {
                "risk_score": risk_score,
                "risk_level": risk_result["risk_level"],
                "top_risk_factors": risk_result["top_risk_factors"],
                "model_used": risk_result["model_type"]
            },
            "escalation": escalation_result,
            "cascade_timeline": cascade_timeline,
            "historical_parallels": historical_matches,
            "confidence": confidence_result,
            "interventions": whatif_results,
            "policy_recommendations": policy_results
        }

        return output

    def predict_quick(self, indicators: Dict) -> Dict:
        """Quick prediction — risk score + level only (fastest)."""
        result = self.lgbm.predict_risk(indicators)
        return {
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
            "top_factors": result["top_risk_factors"][:3]
        }

    def what_if(self, indicators: Dict, intervention_id: str,
                current_risk: float = None, magnitude: float = 1.0) -> Dict:
        """Run what-if analysis for a specific intervention."""
        if current_risk is None:
            current_risk = self.lgbm.predict_risk(indicators)["risk_score"]

        return self.whatif.analyze_intervention(indicators, intervention_id, current_risk, magnitude)

    def compare_interventions(self, indicators: Dict, intervention_ids: List[str],
                               current_risk: float = None) -> Dict:
        """Compare multiple interventions side by side."""
        if current_risk is None:
            current_risk = self.lgbm.predict_risk(indicators)["risk_score"]

        results = []
        for int_id in intervention_ids:
            result = self.whatif.analyze_intervention(indicators, int_id, current_risk)
            results.append(result)

        # Sort by effectiveness
        results.sort(key=lambda x: x.get("risk_after", 100))

        return {
            "current_risk": current_risk,
            "comparisons": results,
            "best_intervention": results[0]["intervention"]["name"] if results else None
        }

    def save_report(self, result: Dict, filepath: str):
        """Save prediction result as JSON report."""
        with open(filepath, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Report saved to {filepath}")

    def get_system_info(self) -> Dict:
        """Get system information and capabilities."""
        return {
            "name": "Dynamic Society Friction Simulator (DSFS)",
            "version": self.VERSION,
            "architecture": "Hybrid ML System",
            "engines": {
                "risk_predictor": "LightGBM / Gradient Boosted Decision Trees",
                "escalation_engine": "CNN-LSTM Temporal Forecaster",
                "historical_matcher": "SBERT Embeddings + FAISS Vector Search",
                "confidence_scorer": "GCRI-Weighted Multi-Factor Assessment",
                "whatif_engine": "Counterfactual Causal Analysis",
                "policy_engine": "Rule-Based + Historical Pattern Engine"
            },
            "indicators": list(self.lgbm.feature_names),
            "capabilities": [
                "Risk Score Prediction (0-100%)",
                "Risk Level Classification (LOW/MEDIUM/HIGH/CRITICAL)",
                "Multi-Horizon Escalation Forecasting (1m/3m/6m)",
                "Historical Case Matching with Similarity Scores",
                "Confidence Assessment with Explainable Reasoning",
                "What-If Intervention Analysis",
                "Cascade Timeline Generation",
                "Policy Recommendation Engine"
            ],
            "is_trained": self.is_trained,
            "historical_cases": len(self.matcher.cases)
        }
