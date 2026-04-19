"""Temporal memory correctness tests (Task 4 of v1 cleanup).

Verifies:
  1. `ExponentialMemory.memory_step` agrees bit-for-bit with the conv1d path
     when fed the same rolling window.
  2. Per-cleavage half-life initialization maps to the expected θ_k values
     (θ = halflife / ln 2) and produces the expected initial kernel.
  3. `FrictionAggregator` now unrolls AR: the sequence of F_k values matches
     hand-rolled per-step softplus(base(t) + memory_step(prev F_k's)).
  4. A longer per-cleavage half-life yields a slower-decaying response to a
     unit impulse (sanity check that the config wiring is end-to-end).

Run with:    python -m pytest tests/test_temporal_memory.py -q
(Requires torch; skipped on import failure.)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")

from src.models.temporal_kernel import ExponentialMemory        # noqa: E402
from src.models.friction_aggregator import (                   # noqa: E402
    AggregatorConfig, FrictionAggregator,
)


def test_memory_step_matches_conv1d_on_known_input():
    """Driving the AR unroll with an EXTERNAL signal (not its own output)
    should reproduce the conv1d path exactly.
    """
    torch.manual_seed(0)
    K, W = 4, 6
    mem = ExponentialMemory(num_cleavages=K, window=W, init_halflife=10.0)
    # Build an arbitrary [S, L, K] stimulus and manually step through memory_step
    # using the stimulus itself (not recursive F_k) — this should match
    # forward(...).
    S, L = 2, 20
    x = torch.randn(S, L, K)
    conv_out = mem(x)                                   # [S, L, K]

    # Roll x through memory_step as if it were the previous F_k series.
    prev = torch.zeros(S, W, K)
    ar_out = torch.zeros(S, L, K)
    for t in range(L):
        ar_out[:, t, :] = mem.memory_step(prev)
        # The conv1d path's memory at time t uses x[t-W..t-1]. Match that
        # by shifting x INTO prev (not the AR F_k output).
        prev = torch.cat([prev[:, 1:, :], x[:, t:t+1, :]], dim=1)

    torch.testing.assert_close(ar_out, conv_out, rtol=1e-5, atol=1e-5)


def test_halflife_init_matches_kernel():
    """θ_k = halflife_k / ln 2, so at τ=halflife the kernel should equal 0.5."""
    K = 3
    halflives = [4.0, 8.0, 16.0]
    mem = ExponentialMemory(num_cleavages=K, window=20, init_halflife=halflives)
    w = mem.kernel().detach()                             # [K, W]
    for k, hl in enumerate(halflives):
        # kernel[k, i] = exp(-(i+1) / θ_k); at i+1 == hl → 0.5.
        tau = int(hl) - 1
        assert abs(float(w[k, tau]) - 0.5) < 5e-3, (
            f"cleavage {k}: kernel at τ={hl} was {float(w[k, tau])}, expected ≈0.5"
        )


def test_aggregator_is_ar_unroll():
    """FrictionAggregator.forward must equal the hand-rolled AR recursion."""
    torch.manual_seed(42)
    S, L, K, W = 3, 25, 4, 6
    agg = FrictionAggregator(AggregatorConfig(
        num_cleavages=K, window_weeks=W, halflife_weeks=[4.0, 8.0, 16.0, 32.0],
    ))
    E = torch.randn(S, L, K); T = torch.randn(S, L, K)
    F_k_model, _ = agg(E, T)

    # Hand roll.
    base = agg.alpha * E + agg.beta * T
    prev = torch.zeros(S, W, K)
    manual = []
    for t in range(L):
        mem_t = agg.memory.memory_step(prev)
        F_t = F.softplus(base[:, t, :] + mem_t)
        manual.append(F_t)
        prev = torch.cat([prev[:, 1:, :], F_t.unsqueeze(1)], dim=1)
    F_k_manual = torch.stack(manual, dim=1)
    torch.testing.assert_close(F_k_model, F_k_manual, rtol=1e-6, atol=1e-6)


def test_longer_halflife_means_slower_decay():
    """Unit-impulse test: cleavage with longer half-life should retain more
    friction at a later time step."""
    K = 2
    S, L, W = 1, 40, 8
    # One fast-decay cleavage, one slow-decay cleavage.
    agg = FrictionAggregator(AggregatorConfig(
        num_cleavages=K, window_weeks=W, halflife_weeks=[2.0, 30.0],
    ))
    # Impulse at t=0 on both cleavages, then zero.
    E = torch.zeros(S, L, K); E[0, 0, :] = 5.0
    T = torch.zeros(S, L, K)
    F_k, _ = agg(E, T)
    # After the impulse the slow-decay cleavage should have larger friction
    # for at least several weeks.
    # Compare F_k at t=5: slow > fast.
    fast = float(F_k[0, 5, 0])
    slow = float(F_k[0, 5, 1])
    assert slow > fast, f"expected slow-decay cleavage to dominate at t=5, got fast={fast}, slow={slow}"
