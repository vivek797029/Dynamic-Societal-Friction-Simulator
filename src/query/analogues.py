"""Historical analogue retrieval.

Given a Scenario (or its cleavage-activation signature), find the top-k
closest (state, week) windows from the historical friction tensor.

Signature:
    sig = severity * 1-hot(cleavages) * 1-hot(states-marginalized) + base rate

We compare it against every (state, week) slice of the normalized friction
tensor using cosine similarity over the cleavage dimension. Results include:
    - state, week
    - similarity
    - top historical articles (optional, if an article index is supplied)

This is a light-weight retrieval pass -- fast, no embedding store needed.
A FAISS-backed variant is the natural upgrade once corpora are large.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..data.india_geo import STATES
from ..models.cleavage_classifier import CLEAVAGES
from .intervention import PipelineContext
from .scenario import Scenario


@dataclass
class Analogue:
    state: str
    iso_week: int
    similarity: float
    cleavage_profile: list[float] = field(default_factory=list)
    articles: list[dict] = field(default_factory=list)       # {article_id, source, title}

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "iso_week": int(self.iso_week),
            "similarity": float(self.similarity),
            "cleavage_profile": list(self.cleavage_profile),
            "articles": list(self.articles),
        }


def scenario_signature(scenario: Scenario, K: int = len(CLEAVAGES)) -> np.ndarray:
    """A cleavage-space signature vector, shape [K]. Independent of states."""
    sig = np.full(K, 0.05, dtype=np.float32)                 # small base rate
    if scenario.cleavages:
        for c in scenario.cleavages:
            sig[CLEAVAGES.index(c)] += scenario.severity
    else:
        sig += 0.3 * scenario.severity                       # nationwide-ish
    return sig


def _friction_per_week(ctx: PipelineContext) -> np.ndarray:
    """A summary (state, week, cleavage) tensor used for similarity.

    We use the normalized sum of E + T as the composite signal. This is
    consistent with how the Stage B aggregator mixes them before adding
    memory, so nearest neighbours here reflect the same features the
    forecaster sees.
    """
    X = ctx.E.astype(np.float32) + ctx.T.astype(np.float32)
    # z-score per state so states with high baseline noise don't dominate
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-6
    return (X - mu) / sd


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def find_analogues(ctx: PipelineContext, scenario: Scenario,
                   k: int = 5,
                   state_restrict: list[str] | None = None,
                   exclude_window_end: int | None = None,
                   article_index: dict[tuple[int, int], list[dict]] | None = None,
                   ) -> list[Analogue]:
    """Top-k (state, week) analogues by cosine similarity in cleavage space.

    Parameters
    ----------
    k : how many analogues to return.
    state_restrict : if given, only compute over these states.
    exclude_window_end : if given, drop weeks >= this index (avoid leaking
                         the future when the user asks about a hypothetical).
    article_index : optional dict {(state_idx, week_rel) -> [article dicts]}
                    to attach evidence citations. When None, Analogue.articles
                    is left empty.
    """
    sig = scenario_signature(scenario, K=ctx.K)
    X = _friction_per_week(ctx)                              # [S, T, K]

    if state_restrict:
        state_idxs = [STATES.index(s) for s in state_restrict if s in STATES]
    else:
        state_idxs = list(range(ctx.S))

    T_max = ctx.T_len if exclude_window_end is None else int(exclude_window_end)
    T_max = max(0, min(T_max, ctx.T_len))

    # Vectorized cosine over the (cleavage) axis.
    sig_n = sig / (np.linalg.norm(sig) + 1e-8)
    # [len(states), T_max, K]
    slice_ = X[state_idxs, :T_max, :]
    slice_n = slice_ / (np.linalg.norm(slice_, axis=-1, keepdims=True) + 1e-8)
    sims = (slice_n * sig_n).sum(axis=-1)                     # [S', T_max]

    flat = sims.reshape(-1)
    if flat.size == 0:
        return []
    top_idx = np.argpartition(-flat, min(k, flat.size - 1))[:k]
    # Sort the chosen subset descending for stable output.
    top_idx = top_idx[np.argsort(-flat[top_idx])]
    out: list[Analogue] = []
    for rank_pos, fi in enumerate(top_idx):
        s_pos, t = divmod(int(fi), T_max)
        s_idx = state_idxs[s_pos]
        profile = slice_[s_pos, t].tolist()
        arts = []
        if article_index is not None:
            arts = list(article_index.get((s_idx, int(t)), []))[:5]
        out.append(Analogue(
            state=STATES[s_idx],
            iso_week=int(t) + ctx.min_week,
            similarity=float(flat[fi]),
            cleavage_profile=profile,
            articles=arts,
        ))
    return out
