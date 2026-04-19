"""Asserts the Stage-B target tensor lines up with build_windows' row order.

Guards against the audit-§1.1 CRITICAL bug: previously `_targets` wrote rows
in (t_end outer, state inner) order while `build_windows` produced (state
outer, t_end inner) via C-order reshape.  The regression test constructs a
trivial F_agg that IS the target — if ordering is right, a tiny MLP trained
for a few dozen steps drives val correlation toward 1. If ordering is wrong,
it plateaus near 0.
"""
from __future__ import annotations

import numpy as np
import torch

from src.models.forecasting_head import build_windows, poisson_nll
from src.training.train_stage_b import build_targets


def test_build_targets_matches_build_windows_order():
    S, T, K = 5, 20, 3
    horizons = (1, 2, 4)
    L = 4

    # Deterministic F so we can check alignment exactly.
    F_k = torch.arange(S * T * K, dtype=torch.float32).reshape(S, T, K) * 0.01
    F_agg = F_k.sum(dim=-1)

    Xk, Xag, t_ends = build_windows(F_k, F_agg, window_len=L)
    N_t = len(t_ends)
    assert Xk.shape == (S * N_t, L, K)

    # Targets: y[s, n, h, target] = F_agg[s, t_end]  (a trivial, state-specific
    # function of the window's last step). We use the same value for protests
    # and fatalities, and for every horizon — the test is about ORDER, not
    # prediction quality.
    y_p = {h: F_agg.numpy() for h in horizons}
    y_f = {h: F_agg.numpy() for h in horizons}

    tgts = build_targets(y_p, y_f, horizons, t_ends, S)  # [N, H, 2]
    assert tgts.shape == (S * N_t, len(horizons), 2)

    # Row i = s * N_t + n corresponds to state s, t_end = t_ends[n].
    # F_agg[s, t_ends[n]] must equal tgts[i, h, 0] and tgts[i, h, 1] for all h.
    for s in range(S):
        for n, te in enumerate(t_ends):
            i = s * N_t + n
            expected = float(F_agg[s, te])
            for h in range(len(horizons)):
                assert abs(tgts[i, h, 0] - expected) < 1e-6
                assert abs(tgts[i, h, 1] - expected) < 1e-6

    # And the Xk/Xag rows line up with the same (s, n) row order.
    Xk_r = Xk.reshape(S, N_t, L, K)
    Xag_r = Xag.reshape(S, N_t, L)
    for s in range(S):
        for n, te in enumerate(t_ends):
            np.testing.assert_allclose(
                Xag_r[s, n].numpy(),
                F_agg[s, te - L + 1 : te + 1].numpy(),
                atol=1e-6,
            )
            np.testing.assert_allclose(
                Xk_r[s, n].numpy(),
                F_k[s, te - L + 1 : te + 1, :].numpy(),
                atol=1e-6,
            )


def test_trivial_regression_converges_with_correct_alignment():
    """If F_agg IS the target (noise-free), a tiny Poisson MLP on top should
    drive train loss down quickly -- but only when targets and windows are
    aligned. Misalignment keeps the loss near its mean-prediction baseline."""
    torch.manual_seed(0)
    S, T, K = 4, 40, 2
    horizons = (1,)
    L = 4
    # Make F_agg a smooth positive signal; y_count = ~Poisson(F_agg).
    F_k = torch.rand(S, T, K) * 0.3
    F_agg = F_k.sum(dim=-1) + 0.5
    Xk, Xag, t_ends = build_windows(F_k, F_agg, window_len=L)
    y_counts = F_agg.numpy()                   # treat as integer-like rate
    y_p = {1: y_counts}
    y_f = {1: np.zeros_like(y_counts)}

    tgts = torch.from_numpy(
        build_targets(y_p, y_f, horizons, t_ends, S)
    ).float()                                                   # [N, 1, 2]

    # Small MLP that reads the window and predicts log-rate.
    in_dim = L * (K + 1)
    mlp = torch.nn.Sequential(
        torch.nn.Linear(in_dim, 16), torch.nn.GELU(),
        torch.nn.Linear(16, 2),                                 # (protests, fatalities) log-rates
    )
    optim = torch.optim.Adam(mlp.parameters(), lr=5e-2)
    X = torch.cat([Xk, Xag.unsqueeze(-1)], dim=-1).flatten(1)

    init = None
    for step in range(200):
        pred = mlp(X).unsqueeze(1)                              # [N, 1, 2]
        loss = poisson_nll(pred, tgts)
        if init is None:
            init = float(loss)
        optim.zero_grad(); loss.backward(); optim.step()

    final = float(loss)
    # With correct alignment we easily beat init by a wide margin.
    assert final < 0.5 * init, f"loss didn't decrease enough: {init:.3f} -> {final:.3f}"
