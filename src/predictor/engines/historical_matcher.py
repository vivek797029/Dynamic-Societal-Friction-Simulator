"""
Historical Case Matcher — Layer 3
Uses SBERT embeddings + FAISS vector search to find the most
similar historical conflict to current conditions.

Shows: "This situation is 92% similar to 2016 Jat Protests"
with duration, severity, outcome for context.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from ..data.historical_cases import HistoricalCaseDB, INDICATOR_METADATA

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class HistoricalCaseMatcher:
    """
    Finds historically similar friction events using dual matching:
    1. Numerical similarity: Cosine distance on indicator vectors
    2. Semantic similarity: SBERT embeddings on textual descriptions

    Combined score provides robust matching.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.db = HistoricalCaseDB()
        self.cases = self.db.get_all_cases()
        self.indicator_names = list(INDICATOR_METADATA.keys())

        # SBERT model for text similarity
        self.sbert_model = None
        if HAS_SBERT:
            try:
                self.sbert_model = SentenceTransformer(model_name)
            except Exception:
                pass

        # Pre-compute embeddings and index
        self._build_index()

    def _build_index(self):
        """Build FAISS index and numerical feature matrix."""
        # Numerical features
        self.feature_matrix = np.zeros((len(self.cases), len(self.indicator_names)), dtype=np.float32)
        for i, case in enumerate(self.cases):
            for j, name in enumerate(self.indicator_names):
                self.feature_matrix[i, j] = case["indicators"].get(name, 0)

        # Normalize for cosine similarity
        norms = np.linalg.norm(self.feature_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.norm_features = self.feature_matrix / norms

        # FAISS index for fast search
        if HAS_FAISS:
            d = self.norm_features.shape[1]
            self.faiss_index = faiss.IndexFlatIP(d)  # Inner product = cosine on normalized
            self.faiss_index.add(self.norm_features)

        # SBERT embeddings for text similarity
        self.text_embeddings = None
        if self.sbert_model is not None:
            descriptions = [
                f"{c['name']}: {c['description']}. Triggers: {', '.join(c['triggers'])}. "
                f"Type: {c['type']}. Country: {c['country']}."
                for c in self.cases
            ]
            self.text_embeddings = self.sbert_model.encode(descriptions, normalize_embeddings=True)

    def find_similar(self, indicators: Dict, description: str = "",
                     top_k: int = 3, weights: Tuple[float, float] = (0.6, 0.4)) -> List[Dict]:
        """Find most similar historical cases.

        Args:
            indicators: Current socioeconomic indicators
            description: Optional text description of current situation
            top_k: Number of results to return
            weights: (numerical_weight, text_weight) for combined scoring

        Returns:
            List of matched cases with similarity scores
        """
        # Numerical similarity
        num_scores = self._numerical_similarity(indicators)

        # Text similarity
        text_scores = np.zeros(len(self.cases))
        if description and self.sbert_model is not None and self.text_embeddings is not None:
            text_scores = self._text_similarity(description)

        # Combined score
        w_num, w_text = weights
        if description and self.sbert_model is not None:
            combined = w_num * num_scores + w_text * text_scores
        else:
            combined = num_scores

        # Get top-k indices
        top_indices = np.argsort(combined)[::-1][:top_k]

        results = []
        for idx in top_indices:
            case = self.cases[idx]
            similarity = float(combined[idx]) * 100  # Convert to percentage

            results.append({
                "case_id": case["id"],
                "name": case["name"],
                "country": case["country"],
                "year": case["year"],
                "similarity_pct": round(min(similarity, 99.9), 1),
                "type": case["type"],
                "severity": case["severity"],
                "duration_days": case["duration_days"],
                "fatalities": case["fatalities"],
                "triggers": case["triggers"],
                "escalation_path": case["escalation_path"],
                "resolution": case["resolution"],
                "outcome": case["outcome"],
                "media_role": case["media_role"],
                "numerical_similarity": round(float(num_scores[idx]) * 100, 1),
                "text_similarity": round(float(text_scores[idx]) * 100, 1) if description else None
            })

        return results

    def _numerical_similarity(self, indicators: Dict) -> np.ndarray:
        """Compute cosine similarity between input and all historical cases."""
        query = np.array([indicators.get(name, 0) for name in self.indicator_names], dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return np.zeros(len(self.cases))
        query_normalized = query / query_norm

        if HAS_FAISS:
            scores, _ = self.faiss_index.search(query_normalized.reshape(1, -1), len(self.cases))
            return scores[0]
        else:
            # Manual cosine similarity
            return np.dot(self.norm_features, query_normalized)

    def _text_similarity(self, description: str) -> np.ndarray:
        """Compute text similarity using SBERT embeddings."""
        if self.sbert_model is None or self.text_embeddings is None:
            return np.zeros(len(self.cases))

        query_emb = self.sbert_model.encode([description], normalize_embeddings=True)
        scores = np.dot(self.text_embeddings, query_emb.T).flatten()
        return scores

    def get_case_comparison(self, indicators: Dict, case_id: str) -> Dict:
        """Get detailed comparison between current conditions and a specific case."""
        case = self.db.get_case_by_id(case_id)
        if case is None:
            return {"error": f"Case {case_id} not found"}

        comparison = {
            "case_name": case["name"],
            "indicator_comparison": [],
            "risk_factors_in_common": [],
            "key_differences": []
        }

        for name in self.indicator_names:
            current_val = indicators.get(name, 0)
            historical_val = case["indicators"].get(name, 0)
            diff = current_val - historical_val
            pct_diff = (diff / historical_val * 100) if historical_val != 0 else 0

            entry = {
                "indicator": name,
                "current": round(float(current_val), 2),
                "historical": round(float(historical_val), 2),
                "difference": round(float(diff), 2),
                "pct_change": round(float(pct_diff), 1),
                "worse": self._is_worse(name, diff)
            }
            comparison["indicator_comparison"].append(entry)

            meta = INDICATOR_METADATA.get(name, {})
            threshold = meta.get("critical_threshold")
            if threshold:
                direction = meta.get("direction", "higher_worse")
                if direction == "higher_worse" and current_val >= threshold and historical_val >= threshold:
                    comparison["risk_factors_in_common"].append(name)
                elif direction == "lower_worse" and current_val <= threshold and historical_val <= threshold:
                    comparison["risk_factors_in_common"].append(name)

        return comparison

    def _is_worse(self, indicator: str, diff: float) -> bool:
        """Check if the difference indicates worsening conditions."""
        meta = INDICATOR_METADATA.get(indicator, {})
        direction = meta.get("direction", "higher_worse")
        if direction == "higher_worse":
            return diff > 0
        elif direction == "lower_worse":
            return diff < 0
        return False
