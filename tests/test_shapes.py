"""Shape / contract tests that do not require GPU or network.

Run with:   python -m pytest tests/  -q
"""
from __future__ import annotations

import numpy as np
import torch

from src.data.india_geo import NUM_STATES, STATES, iso_week_index, resolve_state
from src.data.preprocessing import CLEAVAGES
from src.models.forecasting_head import EscalationHead, ForecastConfig, build_windows
from src.models.friction_aggregator import AggregatorConfig, FrictionAggregator
from src.models.temporal_kernel import ExponentialMemory


def test_india_geo_is_36():
    assert NUM_STATES == 36
    assert len(STATES) == 36
    assert resolve_state("IN16") == "Maharashtra"
    assert resolve_state("XX99") is None


def test_iso_week_monotone():
    a = iso_week_index("2015-01-01")
    b = iso_week_index("2016-01-01")
    assert b - a >= 51


def test_memory_kernel_shapes():
    K = 6
    mem = ExponentialMemory(num_cleavages=K, window=8)
    x = torch.randn(4, 20, K)
    out = mem(x)
    assert out.shape == x.shape


def test_aggregator_shapes():
    S, T, K = 36, 30, len(CLEAVAGES)
    E = torch.randn(S, T, K); Tt = torch.randn(S, T, K)
    agg = FrictionAggregator(AggregatorConfig(num_cleavages=K, window_weeks=8))
    F_k, F_agg = agg(E, Tt)
    assert F_k.shape == (S, T, K)
    assert F_agg.shape == (S, T)
    assert (F_k >= 0).all()


def test_forecasting_windows():
    S, T, K = 36, 30, len(CLEAVAGES)
    F_k = torch.randn(S, T, K).abs()
    F_agg = torch.randn(S, T).abs()
    Xk, Xag, t_ends = build_windows(F_k, F_agg, window_len=8)
    assert Xk.shape[0] == S * len(t_ends)
    assert Xk.shape[1:] == (8, K)
    head = EscalationHead(ForecastConfig(num_cleavages=K, window_len=8))
    out = head(Xk, Xag)
    assert out.shape == (Xk.shape[0], 3, 2)
