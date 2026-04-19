"""Build the trust-weighted discourse tensor T_k(r, t) from Stage-A artifacts.

Inputs:
  articles.parquet with columns: article_id, source_domain, state, iso_week, text
  source_trust.parquet from Stage A with columns: source_domain, tau
  cleavage probs per article  (predicted by cleavage head, stored separately)
  hostility per article × cleavage   (predicted by hostility head, stored separately)

Output: T[S, T, K] saved to .npy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data.india_geo import NUM_STATES, STATE_TO_IDX
from ..data.preprocessing import CLEAVAGES


def build_T_tensor(articles: pd.DataFrame,
                   source_trust: pd.DataFrame,
                   cleavage_probs: np.ndarray,   # [N, K]
                   hostility: np.ndarray,        # [N, K]
                   min_week: int, max_week: int) -> np.ndarray:
    S, T, K = NUM_STATES, max_week - min_week + 1, len(CLEAVAGES)
    T_num = np.zeros((S, T, K), dtype=np.float64)
    T_den = np.zeros((S, T, K), dtype=np.float64) + 1e-8

    tau_map = dict(zip(source_trust["source_domain"], source_trust["tau"]))
    assert len(articles) == cleavage_probs.shape[0] == hostility.shape[0]

    for i, row in enumerate(articles.itertuples(index=False)):
        state = row.state; week = int(row.iso_week)
        si = STATE_TO_IDX.get(state)
        ti = week - min_week
        if si is None or ti < 0 or ti >= T:
            continue
        tau = float(tau_map.get(row.source_domain, 0.5))
        for ki in range(K):
            p = float(cleavage_probs[i, ki])
            h = float(hostility[i, ki])
            T_num[si, ti, ki] += tau * p * h
            T_den[si, ti, ki] += tau * p
    T = T_num / T_den
    return T.astype(np.float32)
