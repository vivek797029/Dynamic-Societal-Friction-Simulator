"""Baselines the trust-weighted model must outperform.

1. **EventCount**       — GDELT negative-event count alone (no Goldstein, no text).
2. **GoldsteinMean**    — Mean Goldstein of negative events per (state, week).
3. **SentimentOnly**    — Mean hostility of articles (NO trust weighting).
4. **ARIMA-ACLED**      — Univariate ARIMA on ACLED counts (forecast-only baseline).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.india_geo import NUM_STATES, STATE_TO_IDX
from ..data.preprocessing import CLEAVAGES


def event_count_baseline(events: pd.DataFrame,
                         min_week: int, max_week: int) -> np.ndarray:
    S, T = NUM_STATES, max_week - min_week + 1
    out = np.zeros((S, T), dtype=np.float32)
    for _, row in events.iterrows():
        si = STATE_TO_IDX.get(row["state"])
        if si is None:
            continue
        ti = int(row["iso_week"]) - min_week
        if 0 <= ti < T and row.get("is_negative", False):
            out[si, ti] += 1.0
    return out


def goldstein_mean_baseline(events: pd.DataFrame,
                            min_week: int, max_week: int) -> np.ndarray:
    S, T = NUM_STATES, max_week - min_week + 1
    num = np.zeros((S, T), dtype=np.float64)
    den = np.zeros((S, T), dtype=np.float64) + 1e-8
    for _, row in events.iterrows():
        si = STATE_TO_IDX.get(row["state"]); ti = int(row["iso_week"]) - min_week
        if si is None or not (0 <= ti < T):
            continue
        g = float(row.get("GoldsteinScale", 0.0))
        num[si, ti] += abs(g); den[si, ti] += 1.0
    return (num / den).astype(np.float32)


def sentiment_only_baseline(articles: pd.DataFrame,
                            hostility: np.ndarray,
                            min_week: int, max_week: int) -> np.ndarray:
    """Mean hostility per (state, week), no trust weighting."""
    S, T = NUM_STATES, max_week - min_week + 1
    num = np.zeros((S, T), dtype=np.float64)
    den = np.zeros((S, T), dtype=np.float64) + 1e-8
    # Use mean over cleavages for each article.
    h_mean = hostility.mean(axis=1)  # [N]
    for i, row in enumerate(articles.itertuples(index=False)):
        si = STATE_TO_IDX.get(row.state); ti = int(row.iso_week) - min_week
        if si is None or not (0 <= ti < T):
            continue
        num[si, ti] += h_mean[i]; den[si, ti] += 1.0
    return (num / den).astype(np.float32)


def arima_baseline(y_history: np.ndarray, horizon: int = 1) -> np.ndarray:
    """Per-state ARIMA(1,0,1) forecast of y at t+h using data up to t."""
    from statsmodels.tsa.arima.model import ARIMA
    S, T = y_history.shape
    out = np.zeros_like(y_history)
    for s in range(S):
        try:
            model = ARIMA(y_history[s], order=(1, 0, 1)).fit(method_kwargs={"warn_convergence": False})
            pred = model.predict(start=0, end=T - 1)
            out[s] = np.clip(pred, 0, None)
        except Exception:
            out[s] = y_history[s].mean()
    return out
