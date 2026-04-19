"""ExponentialMemory: shape, strict causality, monotone-decreasing kernel.

Guards audit §1.1 (memory kernel had off-by-one and wrong permutation).
"""
from __future__ import annotations

import pytest
import torch

from src.models.temporal_kernel import ExponentialMemory


@pytest.mark.parametrize("S,T,K,W", [(3, 20, 6, 8), (1, 5, 4, 3), (7, 40, 2, 12)])
def test_shape_is_preserved(S, T, K, W):
    mem = ExponentialMemory(num_cleavages=K, window=W)
    x = torch.randn(S, T, K)
    out = mem(x)
    assert out.shape == (S, T, K)
    assert torch.isfinite(out).all()


def test_kernel_is_monotone_decreasing():
    K, W = 6, 10
    mem = ExponentialMemory(num_cleavages=K, window=W)
    w = mem.kernel()                                      # [K, W]
    # exp(-(i+1)/theta) is strictly decreasing in i for any theta > 0.
    diffs = w[:, 1:] - w[:, :-1]
    assert (diffs <= 1e-6).all(), "kernel weights must be non-increasing per cleavage"


def test_strict_causality():
    """mem[t] depends ONLY on x[0..t-1], not on x[t] or any future step.

    Changing x at time t0 must leave mem[:, :t0, :] exactly unchanged.
    """
    K, W = 3, 4
    S, T = 2, 15
    mem = ExponentialMemory(num_cleavages=K, window=W).eval()
    x = torch.randn(S, T, K)
    y = mem(x)
    t0 = 6
    x2 = x.clone()
    x2[:, t0:, :] += 10.0                                # perturb present+future
    y2 = mem(x2)
    # Past should be identical.
    assert torch.allclose(y[:, :t0, :], y2[:, :t0, :], atol=1e-6)
    # Future should be different.
    assert not torch.allclose(y[:, t0:, :], y2[:, t0:, :], atol=1e-6)
