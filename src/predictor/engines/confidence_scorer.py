"""
Confidence Scoring Engine — Layer 4
Calculates WHY the model is confident in its prediction.

Uses multi-factor confidence assessment:
1. Historical similarity strength
2. Economic indicator alignment
3. Political factor alignment
4. Social factor alignment
5. Model agreement (ensemble consensus)

Output: Confidence percentage + reasoning factors
"""

import numpy as np
from typing import Dict, List


class ConfidenceScorer:
    """
    Calculates prediction confidence with explainable reasoning.
    Based on GCRI methodology and ensemble uncertainty estimation.
    """

    # Factor weights for confidence calculation
    FACTOR_WEIGHTS = {
        "historical_match": 0.25,
        "economic_alignment": 0.25,
        "political_alignment": 0.20,
        "social_alignment": 0.15,
        "model_consensus": 0.15
    }

    # Thresholds for each indicator being "alarming"
    ALARM_THRESHOLDS = {
        "unemployment_rate": {"threshold": 10, "direction": "above"},
        "inflation_rate": {"threshold": 8, "direction": "above"},
        "gini_coefficient": {"threshold": 0.40, "direction": "above"},
        "youth_bulge_pct": {"threshold": 55, "direction": "above"},
        "political_stability": {"threshold": 0.35, "direction": "below"},
        "press_freedom_index": {"threshold": 0.30, "direction": "below"},
        "gdp_growth": {"threshold": 0, "direction": "below"},
        "poverty_rate": {"threshold": 25, "direction": "above"},
        "corruption_index": {"threshold": 35, "direction": "below"},
        "food_price_index": {"threshold": 125, "direction": "above"}
    }

    def calculate_confidence(self, indicators: Dict, risk_score: float,
                             historical_matches: List[Dict],
                             escalation_result: Dict = None) -> Dict:
        """Calculate comprehensive confidence score with reasoning.

        Args:
            indicators: Current socioeconomic indicators
            risk_score: Predicted risk score from LGBM
            historical_matches: Matched historical cases
            escalation_result: CNN-LSTM escalation prediction

        Returns:
            Dict with confidence score, factors, and reasoning
        """
        factors = {}
        reasoning = []

        # 1. Historical match confidence
        hist_conf = self._historical_confidence(historical_matches)
        factors["historical_match"] = hist_conf
        if hist_conf > 70:
            best = historical_matches[0] if historical_matches else {}
            reasoning.append({
                "factor": "Historical Parallel",
                "strength": "strong",
                "detail": f"Current conditions are {best.get('similarity_pct', 0):.0f}% similar to "
                          f"{best.get('name', 'unknown')} ({best.get('year', '')}) — "
                          f"severity was {best.get('severity', 'unknown')}"
            })

        # 2. Economic alignment
        econ_conf = self._economic_confidence(indicators)
        factors["economic_alignment"] = econ_conf
        alarming_econ = self._get_alarming_indicators(indicators, ["unemployment_rate", "inflation_rate",
                                                                     "gdp_growth", "food_price_index"])
        if alarming_econ:
            reasoning.append({
                "factor": "Economic Indicators",
                "strength": "strong" if econ_conf > 70 else "moderate",
                "detail": f"{len(alarming_econ)} economic indicators past critical threshold: "
                          f"{', '.join(alarming_econ)}"
            })

        # 3. Political alignment
        pol_conf = self._political_confidence(indicators)
        factors["political_alignment"] = pol_conf
        alarming_pol = self._get_alarming_indicators(indicators, ["political_stability", "corruption_index",
                                                                    "press_freedom_index"])
        if alarming_pol:
            reasoning.append({
                "factor": "Political Indicators",
                "strength": "strong" if pol_conf > 70 else "moderate",
                "detail": f"Political risk factors active: {', '.join(alarming_pol)}"
            })

        # 4. Social alignment
        soc_conf = self._social_confidence(indicators)
        factors["social_alignment"] = soc_conf
        alarming_soc = self._get_alarming_indicators(indicators, ["gini_coefficient", "youth_bulge_pct",
                                                                    "poverty_rate"])
        if alarming_soc:
            reasoning.append({
                "factor": "Social Indicators",
                "strength": "strong" if soc_conf > 70 else "moderate",
                "detail": f"Social risk factors: {', '.join(alarming_soc)}"
            })

        # 5. Model consensus
        model_conf = self._model_consensus_confidence(risk_score, escalation_result)
        factors["model_consensus"] = model_conf
        if model_conf > 70:
            reasoning.append({
                "factor": "Model Agreement",
                "strength": "strong",
                "detail": "Multiple prediction models agree on risk assessment"
            })

        # Overall confidence (weighted average)
        overall = sum(
            factors[k] * self.FACTOR_WEIGHTS[k]
            for k in self.FACTOR_WEIGHTS
        )
        overall = np.clip(overall, 10, 98)

        # Confidence level label
        if overall >= 85:
            confidence_level = "VERY HIGH"
        elif overall >= 70:
            confidence_level = "HIGH"
        elif overall >= 55:
            confidence_level = "MODERATE"
        elif overall >= 40:
            confidence_level = "LOW"
        else:
            confidence_level = "VERY LOW"

        return {
            "confidence_score": round(float(overall), 1),
            "confidence_level": confidence_level,
            "factor_scores": {k: round(float(v), 1) for k, v in factors.items()},
            "factor_weights": self.FACTOR_WEIGHTS,
            "reasoning": reasoning,
            "total_alarming_indicators": len(alarming_econ) + len(alarming_pol) + len(alarming_soc),
            "alarming_indicators": alarming_econ + alarming_pol + alarming_soc
        }

    def _historical_confidence(self, matches: List[Dict]) -> float:
        """Confidence from historical case matching."""
        if not matches:
            return 30.0

        best_sim = matches[0].get("similarity_pct", 0)
        # Top match similarity directly translates to confidence
        return np.clip(best_sim, 20, 95)

    def _economic_confidence(self, indicators: Dict) -> float:
        """Confidence from economic indicators alignment."""
        econ_indicators = ["unemployment_rate", "inflation_rate", "gdp_growth",
                          "food_price_index", "poverty_rate"]
        return self._calculate_factor_confidence(indicators, econ_indicators)

    def _political_confidence(self, indicators: Dict) -> float:
        """Confidence from political indicators alignment."""
        pol_indicators = ["political_stability", "corruption_index", "press_freedom_index"]
        return self._calculate_factor_confidence(indicators, pol_indicators)

    def _social_confidence(self, indicators: Dict) -> float:
        """Confidence from social indicators alignment."""
        soc_indicators = ["gini_coefficient", "youth_bulge_pct", "poverty_rate"]
        return self._calculate_factor_confidence(indicators, soc_indicators)

    def _calculate_factor_confidence(self, indicators: Dict, factor_names: List[str]) -> float:
        """Calculate confidence for a group of indicators."""
        alarming_count = 0
        total = 0

        for name in factor_names:
            if name not in self.ALARM_THRESHOLDS:
                continue
            total += 1
            value = indicators.get(name, 0)
            alarm = self.ALARM_THRESHOLDS[name]

            if alarm["direction"] == "above" and value >= alarm["threshold"]:
                alarming_count += 1
            elif alarm["direction"] == "below" and value <= alarm["threshold"]:
                alarming_count += 1

        if total == 0:
            return 50.0

        ratio = alarming_count / total
        return np.clip(ratio * 100 + 20, 20, 95)

    def _model_consensus_confidence(self, risk_score: float, escalation: Dict = None) -> float:
        """Confidence from model agreement."""
        if escalation is None:
            return 60.0

        # Check if LGBM risk and escalation trend agree
        trend = escalation.get("trend", "stable")
        if risk_score >= 60 and trend in ["escalating", "rapidly_escalating"]:
            return 90.0
        elif risk_score >= 60 and trend == "stable":
            return 65.0
        elif risk_score < 40 and trend in ["stable", "de_escalating"]:
            return 85.0
        else:
            return 55.0

    def _get_alarming_indicators(self, indicators: Dict, subset: List[str]) -> List[str]:
        """Get list of indicators past their critical threshold."""
        alarming = []
        for name in subset:
            if name not in self.ALARM_THRESHOLDS:
                continue
            value = indicators.get(name, 0)
            alarm = self.ALARM_THRESHOLDS[name]

            if alarm["direction"] == "above" and value >= alarm["threshold"]:
                alarming.append(name)
            elif alarm["direction"] == "below" and value <= alarm["threshold"]:
                alarming.append(name)
        return alarming
