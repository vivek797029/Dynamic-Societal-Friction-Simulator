"""
Synthetic Training Data Generator for DSFS Friction Predictor.
Generates realistic socioeconomic condition → friction outcome pairs
by augmenting historical cases with controlled perturbation.
"""

import random
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
from .historical_cases import (
    HISTORICAL_CASES, INDICATOR_METADATA, SEVERITY_MAP, FRICTION_TYPES
)


class FrictionDataGenerator:
    """Generates training data for ML models by augmenting historical cases."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.cases = HISTORICAL_CASES
        self.indicator_names = list(INDICATOR_METADATA.keys())

    def _perturb_indicators(self, indicators: Dict, noise_level: float = 0.15) -> Dict:
        """Add realistic noise to indicators to create variations."""
        perturbed = {}
        for key, value in indicators.items():
            meta = INDICATOR_METADATA.get(key, {})
            # Add gaussian noise proportional to value
            noise = self.rng.normal(0, abs(value) * noise_level) if value != 0 else self.rng.normal(0, 1)
            new_val = value + noise
            # Clamp to reasonable ranges
            if meta.get("unit") == "0-1":
                new_val = np.clip(new_val, 0.0, 1.0)
            elif meta.get("unit") == "%":
                new_val = max(0, new_val)
            perturbed[key] = round(float(new_val), 2)
        return perturbed

    def _calculate_risk_score(self, indicators: Dict) -> float:
        """Calculate risk score using weighted GCRI-style formula."""
        score = 0.0
        for key, meta in INDICATOR_METADATA.items():
            value = indicators.get(key, 0)
            weight = meta["weight"]
            threshold = meta.get("critical_threshold")
            direction = meta.get("direction", "higher_worse")

            if threshold is None:
                continue

            if direction == "higher_worse":
                # Higher value = more risk
                normalized = min(value / threshold, 2.0) if threshold > 0 else 0
            elif direction == "lower_worse":
                # Lower value = more risk
                normalized = min(threshold / max(value, 0.01), 2.0)
            else:
                continue

            score += weight * normalized * 50  # Scale to 0-100 range

        return np.clip(score, 0, 100)

    def _assign_risk_level(self, score: float) -> str:
        if score >= 80:
            return "CRITICAL"
        elif score >= 60:
            return "HIGH"
        elif score >= 40:
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_timeline(self, risk_score: float) -> Dict:
        """Generate predicted timeline based on risk level."""
        if risk_score >= 80:
            onset = self.rng.randint(7, 30)
            peak = self.rng.randint(30, 90)
        elif risk_score >= 60:
            onset = self.rng.randint(30, 90)
            peak = self.rng.randint(90, 180)
        elif risk_score >= 40:
            onset = self.rng.randint(60, 180)
            peak = self.rng.randint(180, 365)
        else:
            onset = self.rng.randint(180, 365)
            peak = self.rng.randint(365, 730)

        return {
            "onset_days": int(onset),
            "peak_days": int(peak),
            "onset_text": f"{onset // 30} months" if onset >= 30 else f"{onset} days",
            "peak_text": f"{peak // 30} months" if peak >= 30 else f"{peak} days"
        }

    def _pick_escalation_path(self, friction_type: str, severity: str) -> List[str]:
        """Generate a realistic escalation path."""
        paths = {
            "economic_protest": [
                ["price_hike", "local_protest", "nationwide_protest", "government_response"],
                ["job_losses", "union_strike", "mass_protest", "policy_change"],
                ["shortages", "panic_buying", "protests", "looting", "state_response"]
            ],
            "identity_conflict": [
                ["discrimination", "community_tension", "violent_clashes", "curfew"],
                ["political_rhetoric", "community_divide", "targeted_violence", "military_deployment"]
            ],
            "political_protest": [
                ["policy_announcement", "opposition_rally", "mass_protest", "crackdown"],
                ["election_dispute", "protest", "clashes", "international_pressure"]
            ],
            "regime_change": [
                ["trigger_incident", "local_protest", "national_uprising", "regime_collapse"],
                ["economic_crisis", "mass_protests", "military_split", "transition"]
            ]
        }
        options = paths.get(friction_type, paths["economic_protest"])
        return random.choice(options)

    def generate_single_sample(self, base_case: Dict = None) -> Dict:
        """Generate a single training sample."""
        if base_case is None:
            base_case = random.choice(self.cases)

        # Perturb indicators
        indicators = self._perturb_indicators(base_case["indicators"])

        # Calculate risk
        risk_score = self._calculate_risk_score(indicators)
        risk_level = self._assign_risk_level(risk_score)

        # Generate timeline
        timeline = self._generate_timeline(risk_score)

        # Pick friction type and escalation
        friction_type = base_case["type"]
        escalation = self._pick_escalation_path(friction_type, risk_level)

        sample = {
            "input_conditions": indicators,
            "country": base_case["country"],
            "region": base_case["region"],
            "prediction": {
                "risk_score": round(float(risk_score), 1),
                "risk_level": risk_level,
                "friction_type": friction_type,
                "timeline": timeline,
                "escalation_path": escalation,
                "affected_groups": base_case["affected_groups"],
                "triggers": base_case["triggers"],
                "media_role": base_case["media_role"]
            },
            "similar_historical_case": base_case["name"],
            "severity_label": SEVERITY_MAP.get(base_case["severity"], 2)
        }
        return sample

    def generate_dataset(self, n_samples: int = 10000, train_ratio: float = 0.8) -> Dict:
        """Generate full train/eval dataset."""
        samples = []
        for i in range(n_samples):
            base_case = self.cases[i % len(self.cases)]
            sample = self.generate_single_sample(base_case)
            samples.append(sample)

        # Shuffle
        self.rng.shuffle(samples)

        # Split
        split_idx = int(len(samples) * train_ratio)
        train_data = samples[:split_idx]
        eval_data = samples[split_idx:]

        return {"train": train_data, "eval": eval_data}

    def generate_feature_matrix(self, n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate numpy feature matrix for ML training.
        Returns: (features, risk_scores, severity_labels)
        """
        dataset = self.generate_dataset(n_samples, train_ratio=1.0)
        all_samples = dataset["train"]

        features = []
        risk_scores = []
        severity_labels = []

        for sample in all_samples:
            feat_vec = [sample["input_conditions"].get(k, 0) for k in self.indicator_names]
            features.append(feat_vec)
            risk_scores.append(sample["prediction"]["risk_score"])
            severity_labels.append(sample["severity_label"])

        return (
            np.array(features, dtype=np.float32),
            np.array(risk_scores, dtype=np.float32),
            np.array(severity_labels, dtype=np.int32)
        )

    def save_dataset(self, output_dir: str, n_samples: int = 10000):
        """Save generated dataset to JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        dataset = self.generate_dataset(n_samples)

        with open(output_path / "train.json", "w") as f:
            json.dump(dataset["train"], f, indent=2)

        with open(output_path / "eval.json", "w") as f:
            json.dump(dataset["eval"], f, indent=2)

        print(f"Generated {len(dataset['train'])} training samples")
        print(f"Generated {len(dataset['eval'])} evaluation samples")
        print(f"Saved to {output_path}")

        return dataset
