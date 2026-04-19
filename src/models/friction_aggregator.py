"""M7 — Trust-weighted friction aggregator.

F_k(r, t) = softplus( α_k · Ẽ_k(r, t) + β_k · T̃_k(r, t)
                    + λ_k · Σ_{τ=1..W} exp(-τ / θ_k) · F_k(r, t-τ) )

F(r, t) = Σ_k softmax(ω)_k · F_k(r, t)

Tildes are per-state z-scores, precomputed on the training split.
All weights (α, β, ω, λ, θ) are learned against ACLED through an escalation
forecasting head in M8.

v1 changes:
  * Removed the optional R / relational-strain channel (see actor_graph.py
    tombstone): aggregator is now strictly (E, T).
  * Per-cleavage memory half-life is now wired from
    `cfg.aggregator.memory_halflife_weeks` instead of the hard-coded 16-week
    scalar default. Communal/caste/linguistic tensions decay slowly, political
    and economic ones decay quickly.
  * The memory sum now accumulates over the past F_k values (the AR
    recursion in the formula above), not over the pre-softplus stimulus
    `base = αE + βT`. The previous convolutional shortcut replaced `F_k(t-τ)`
    with `base(t-τ)` — a structural divergence from the documented model.
    We now unroll over time explicitly; for T ~ 500 weeks and K = 6 this
    costs a handful of milliseconds per epoch on GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_kernel import ExponentialMemory


@dataclass
class AggregatorConfig:
    num_cleavages: int = 6
    window_weeks: int = 12
    # Per-cleavage memory half-life in weeks. Length must equal num_cleavages
    # when non-None; if None, the kernel initializes to a uniform 16-week
    # half-life (training may still move it, but the prior is weaker).
    halflife_weeks: Sequence[float] | None = None


class FrictionAggregator(nn.Module):
    def __init__(self, cfg: AggregatorConfig | None = None):
        super().__init__()
        self.cfg = cfg or AggregatorConfig()
        K = self.cfg.num_cleavages
        # Per-cleavage mixing weights.
        self.alpha = nn.Parameter(torch.ones(K) * 0.5)
        self.beta = nn.Parameter(torch.ones(K) * 0.5)
        self.omega = nn.Parameter(torch.zeros(K))  # softmaxed → uniform init
        halflife = self.cfg.halflife_weeks if self.cfg.halflife_weeks is not None else 16.0
        self.memory = ExponentialMemory(
            num_cleavages=K, window=self.cfg.window_weeks,
            init_halflife=halflife,
        )

    def forward(self, E: torch.Tensor, T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """AR unroll over time.

        E, T: [S, L, K] (normalized).
        Returns:
            F_k: [S, L, K] per-cleavage friction, strictly non-negative.
            F_agg: [S, L] aggregate friction.
        """
        S, L, K = E.shape
        assert K == self.cfg.num_cleavages
        assert T.shape == E.shape

        base = self.alpha * E + self.beta * T                          # [S, L, K]
        W = self.memory.W

        # Rolling window of the past W F_k values; newest is at index -1.
        # Initialize to zeros — memory_step for t < W implicitly windows with
        # zero-padding, matching the conv1d path's left-pad behaviour.
        prev = torch.zeros(S, W, K, device=base.device, dtype=base.dtype)
        F_k_list: list[torch.Tensor] = []
        for t in range(L):
            mem_t = self.memory.memory_step(prev)                      # [S, K]
            F_t = F.softplus(base[:, t, :] + mem_t)                    # [S, K]
            F_k_list.append(F_t)
            # Shift left, append F_t as the newest entry.
            prev = torch.cat([prev[:, 1:, :], F_t.unsqueeze(1)], dim=1)

        F_k = torch.stack(F_k_list, dim=1)                             # [S, L, K]
        w_agg = torch.softmax(self.omega, dim=0)
        F_agg = (F_k * w_agg).sum(dim=-1)                              # [S, L]
        return F_k, F_agg
