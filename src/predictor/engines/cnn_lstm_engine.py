"""
CNN-LSTM Escalation Prediction Engine — Layer 2
Predicts how risk evolves over time using hybrid CNN-LSTM architecture.

CNN: Extracts short-term patterns from indicator sequences
LSTM: Captures long-range temporal dependencies for escalation trends

This is a time-series forecasting model — predicts risk trajectory
over 1-month, 3-month, and 6-month horizons.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Create a dummy base class when torch is not available
    class _DummyModule:
        pass
    class _nn:
        Module = _DummyModule
    nn = _nn()


class CNNLSTMModel(nn.Module if HAS_TORCH else object):
    """Hybrid CNN-LSTM for temporal risk escalation prediction."""

    def __init__(self, n_features: int = 14, seq_length: int = 12,
                 cnn_filters: int = 64, lstm_hidden: int = 128,
                 n_horizons: int = 3):
        super().__init__()

        # CNN layers — extract local temporal patterns
        self.conv1 = nn.Conv1d(n_features, cnn_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        self.bn2 = nn.BatchNorm1d(cnn_filters * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # LSTM layers — capture long-range dependencies
        self.lstm = nn.LSTM(
            input_size=cnn_filters * 2,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.Linear(lstm_hidden * 2, 1)

        # Output heads for multi-horizon prediction
        self.risk_head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_horizons)  # 1-month, 3-month, 6-month
        )

        self.escalation_head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4),  # 4 escalation stages
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_length, n_features)
        Returns:
            risk_trajectory: (batch, n_horizons) — risk at 1m, 3m, 6m
            escalation_probs: (batch, 4) — probability of each stage
        """
        # CNN feature extraction
        x_cnn = x.permute(0, 2, 1)  # (batch, features, seq)
        x_cnn = self.dropout(self.relu(self.bn1(self.conv1(x_cnn))))
        x_cnn = self.dropout(self.relu(self.bn2(self.conv2(x_cnn))))
        x_cnn = x_cnn.permute(0, 2, 1)  # (batch, seq, cnn_features)

        # LSTM temporal modeling
        lstm_out, _ = self.lstm(x_cnn)  # (batch, seq, hidden*2)

        # Attention pooling
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*2)

        # Multi-horizon risk prediction
        risk_trajectory = self.risk_head(context)
        risk_trajectory = torch.sigmoid(risk_trajectory) * 100  # 0-100 scale

        # Escalation stage probabilities
        escalation_probs = self.escalation_head(context)

        return risk_trajectory, escalation_probs


class EscalationPredictor:
    """
    High-level interface for the CNN-LSTM escalation engine.
    Handles training, prediction, and trajectory generation.
    """

    ESCALATION_STAGES = [
        "tension_building",
        "initial_outbreak",
        "active_escalation",
        "crisis_peak"
    ]

    HORIZONS = ["1_month", "3_months", "6_months"]

    def __init__(self, model_dir: str = "models/cnn_lstm"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.is_trained = False

        if HAS_TORCH:
            self.model = CNNLSTMModel()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

    def _generate_temporal_sequence(self, indicators: Dict, months: int = 12) -> np.ndarray:
        """Generate a synthetic temporal sequence from current indicators.
        Simulates how indicators might have evolved over the past N months.
        """
        feature_names = list(indicators.keys())
        n_features = len(feature_names)
        sequence = np.zeros((months, n_features))

        for i, name in enumerate(feature_names):
            current_val = float(indicators[name])
            # Generate plausible historical trajectory leading to current value
            trend = np.linspace(current_val * 0.85, current_val, months)
            noise = np.random.normal(0, abs(current_val) * 0.05, months)
            sequence[:, i] = trend + noise

        return sequence.astype(np.float32)

    def predict_escalation(self, indicators: Dict, current_risk: float = 50.0) -> Dict:
        """Predict risk escalation trajectory.

        Args:
            indicators: Current socioeconomic indicators
            current_risk: Current risk score from LGBM

        Returns:
            Dict with risk trajectory, escalation stage, cascade timeline
        """
        if HAS_TORCH and self.is_trained and self.model is not None:
            return self._model_predict(indicators)

        # Analytical fallback (no torch needed)
        return self._analytical_predict(indicators, current_risk)

    def _model_predict(self, indicators: Dict) -> Dict:
        """Predict using trained CNN-LSTM model."""
        sequence = self._generate_temporal_sequence(indicators)
        x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            risk_traj, esc_probs = self.model(x)

        risk_trajectory = risk_traj[0].cpu().numpy()
        escalation_probs = esc_probs[0].cpu().numpy()

        return {
            "risk_trajectory": {
                self.HORIZONS[i]: round(float(risk_trajectory[i]), 1)
                for i in range(len(self.HORIZONS))
            },
            "escalation_stage": {
                self.ESCALATION_STAGES[i]: round(float(escalation_probs[i]) * 100, 1)
                for i in range(4)
            },
            "current_stage": self.ESCALATION_STAGES[np.argmax(escalation_probs)],
            "trend": self._calculate_trend(risk_trajectory),
            "model_type": "CNN-LSTM"
        }

    def _analytical_predict(self, indicators: Dict, current_risk: float) -> Dict:
        """Analytical escalation prediction without neural network.
        Uses momentum-based risk propagation formula.
        """
        # Risk momentum factors
        unemployment = float(indicators.get("unemployment_rate", 5))
        inflation = float(indicators.get("inflation_rate", 3))
        stability = float(indicators.get("political_stability", 0.5))
        youth_bulge = float(indicators.get("youth_bulge_pct", 40))
        gini = float(indicators.get("gini_coefficient", 0.35))
        food_index = float(indicators.get("food_price_index", 100))

        # Momentum calculation (how fast risk is growing)
        economic_momentum = (unemployment / 10) * 0.3 + (inflation / 8) * 0.3 + (food_index / 120) * 0.2
        social_momentum = (youth_bulge / 50) * 0.3 + (gini / 0.4) * 0.3
        political_momentum = ((1 - stability) / 0.5) * 0.4

        total_momentum = (economic_momentum * 0.4 + social_momentum * 0.3 + political_momentum * 0.3)
        total_momentum = np.clip(total_momentum, 0.3, 2.5)

        # Project risk over horizons
        risk_1m = np.clip(current_risk * (1 + total_momentum * 0.08), 0, 100)
        risk_3m = np.clip(current_risk * (1 + total_momentum * 0.22), 0, 100)
        risk_6m = np.clip(current_risk * (1 + total_momentum * 0.40), 0, 100)

        # Escalation stage probabilities
        if current_risk >= 75:
            stage_probs = [0.05, 0.15, 0.35, 0.45]
        elif current_risk >= 55:
            stage_probs = [0.10, 0.25, 0.45, 0.20]
        elif current_risk >= 35:
            stage_probs = [0.25, 0.45, 0.25, 0.05]
        else:
            stage_probs = [0.60, 0.30, 0.08, 0.02]

        trend_val = risk_6m - current_risk
        if trend_val > 15:
            trend = "rapidly_escalating"
        elif trend_val > 5:
            trend = "escalating"
        elif trend_val > -5:
            trend = "stable"
        else:
            trend = "de_escalating"

        return {
            "risk_trajectory": {
                "1_month": round(float(risk_1m), 1),
                "3_months": round(float(risk_3m), 1),
                "6_months": round(float(risk_6m), 1)
            },
            "escalation_stage": {
                self.ESCALATION_STAGES[i]: round(float(stage_probs[i]) * 100, 1)
                for i in range(4)
            },
            "current_stage": self.ESCALATION_STAGES[np.argmax(stage_probs)],
            "trend": trend,
            "momentum": round(float(total_momentum), 3),
            "model_type": "Analytical_Momentum"
        }

    def _calculate_trend(self, trajectory: np.ndarray) -> str:
        """Determine trend from trajectory."""
        diff = trajectory[-1] - trajectory[0]
        if diff > 15:
            return "rapidly_escalating"
        elif diff > 5:
            return "escalating"
        elif diff > -5:
            return "stable"
        else:
            return "de_escalating"

    def generate_cascade_timeline(self, indicators: Dict, current_risk: float) -> List[Dict]:
        """Generate step-by-step cascade timeline.
        Shows: Protests → Riots → Violence → Crisis with dates.
        """
        escalation = self.predict_escalation(indicators, current_risk)
        momentum = escalation.get("momentum", 1.0)
        risk = current_risk

        stages = [
            {"stage": "Tension Building", "description": "Grievances accumulate, social media activity increases",
             "risk_change": 5},
            {"stage": "Initial Outbreak", "description": "First protests or demonstrations emerge",
             "risk_change": 15},
            {"stage": "Active Escalation", "description": "Protests spread, potential for violence increases",
             "risk_change": 20},
            {"stage": "Crisis Peak", "description": "Maximum instability, government intervention required",
             "risk_change": 10}
        ]

        timeline = []
        days_elapsed = 0
        for stage in stages:
            # Time to next stage depends on momentum and current risk
            if momentum > 1.5:
                days_to_next = np.random.randint(7, 21)
            elif momentum > 1.0:
                days_to_next = np.random.randint(21, 60)
            else:
                days_to_next = np.random.randint(60, 180)

            days_elapsed += days_to_next
            risk = min(100, risk + stage["risk_change"])

            timeline.append({
                "stage": stage["stage"],
                "description": stage["description"],
                "days_from_now": days_elapsed,
                "timeframe": f"{days_elapsed // 30} months, {days_elapsed % 30} days",
                "projected_risk": round(float(risk), 1),
                "probability": round(max(20, min(95, risk)), 0)
            })

        return timeline

    def save(self, filepath: str = None):
        """Save model weights."""
        if filepath is None:
            filepath = str(self.model_dir / "cnn_lstm_model.pt")
        if HAS_TORCH and self.model is not None:
            torch.save(self.model.state_dict(), filepath)
            print(f"CNN-LSTM model saved to {filepath}")

    def load(self, filepath: str = None):
        """Load model weights."""
        if filepath is None:
            filepath = str(self.model_dir / "cnn_lstm_model.pt")
        if HAS_TORCH and Path(filepath).exists():
            self.model.load_state_dict(torch.load(filepath, map_location="cpu"))
            self.is_trained = True
            print(f"CNN-LSTM model loaded from {filepath}")
