"""Learned exponential-decay memory kernel.

Design:
  * Per-cleavage monotone-decay kernel w[k, i] = exp(-(i+1) / θ_k), i = 0..W-1.
    Index 0 corresponds to τ=1 (most recent lag), W-1 to τ=W (oldest).
  * θ_k is parameterized via softplus so it stays > 0; initialization is
    set from a per-cleavage half-life (weeks) passed at construction.
  * `log_lambda[k]` — gating scalar (softplus → λ_k > 0) multiplied into
    the memory sum.

There are two usage modes:

  * `forward(x)` — the original strictly-causal 1D conv over `x`. Handy for
    sanity tests and for the case where the input is an exogenous stimulus
    rather than the previous friction value. Fast.

  * `memory_step(prev)` — given a rolling window of the past W values of
    F_k (shape [S, W, K], `prev[:, -1, :]` is the most recent), return the
    weighted sum Σ λ_k · w[k, τ-1] · prev[:, -τ, :] at the current step.
    This is the per-step operator used by `FrictionAggregator` to unroll
    the AR recursion  F_k(t) = softplus(base(t) + memory_step(F_k(t-W..t-1))).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(y: torch.Tensor) -> torch.Tensor:
    """x such that softplus(x) == y. Stable for y >> 1."""
    return torch.log(torch.expm1(y.clamp(min=1e-6)))


class ExponentialMemory(nn.Module):
    def __init__(self, num_cleavages: int, window: int = 12,
                 init_halflife: float | Sequence[float] = 16.0):
        """
        init_halflife:
          * scalar → every cleavage starts with the same half-life (weeks).
          * sequence of length num_cleavages → per-cleavage initialization
            (recommended; read from `cfg.aggregator.memory_halflife_weeks`).
        """
        super().__init__()
        self.K = num_cleavages
        self.W = window
        # Convert half-life → decay time-constant θ via θ = halflife / ln 2.
        if isinstance(init_halflife, (int, float)):
            hl = torch.full((num_cleavages,), float(init_halflife))
        else:
            hl = torch.tensor([float(x) for x in init_halflife], dtype=torch.float32)
            if hl.numel() != num_cleavages:
                raise ValueError(
                    f"init_halflife has {hl.numel()} entries, expected {num_cleavages}"
                )
        theta = hl / 0.6931  # ln 2
        self.log_theta = nn.Parameter(_inverse_softplus(theta))
        # λ_k — log-space to keep positive.
        self.log_lambda = nn.Parameter(torch.zeros(num_cleavages))

    def kernel(self) -> torch.Tensor:
        """Returns [K, W] with kernel[k, i] = exp(-(i+1) / θ_k)."""
        theta = F.softplus(self.log_theta).unsqueeze(-1)               # [K, 1]
        tau = torch.arange(1, self.W + 1, device=theta.device,
                           dtype=theta.dtype).unsqueeze(0)             # [1, W]
        return torch.exp(-tau / theta)                                 # [K, W]

    def lam(self) -> torch.Tensor:
        return F.softplus(self.log_lambda)                             # [K]

    # ----- conv1d path (used for sanity tests / fast feed-forward) -----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [S, T, K]  →  mem: [S, T, K], strictly causal."""
        S, T, K = x.shape
        assert K == self.K, f"kernel K={self.K} but input K={K}"
        W = self.W
        w = self.kernel()                                              # [K, W]
        lam = self.lam()                                               # [K]
        x_ch = x.permute(0, 2, 1)                                      # [S, K, T]
        x_pad = F.pad(x_ch, (W, 0))                                    # [S, K, T+W]
        flipped = torch.flip(w, dims=[-1]).unsqueeze(1)                # [K, 1, W]
        mem = F.conv1d(x_pad, flipped, groups=K)                       # [S, K, T+1]
        mem = mem[..., :T] * lam.view(1, K, 1)                         # [S, K, T]
        return mem.permute(0, 2, 1).contiguous()                       # [S, T, K]

    # ----- per-step AR path (used by FrictionAggregator) -----
    def memory_step(self, prev: torch.Tensor) -> torch.Tensor:
        """Single-step memory operator.

        prev: [S, W, K] — the past W values of F_k, with `prev[:, -1, :]`
          being the most recent (t-1) and `prev[:, 0, :]` the oldest (t-W).
        Returns mem_t: [S, K] = λ_k · Σ_{τ=1..W} exp(-τ/θ_k) · prev[:, -τ, :].
        """
        S, W, K = prev.shape
        assert W == self.W and K == self.K
        w = self.kernel()                                              # [K, W]
        lam = self.lam()                                               # [K]
        # w[k, i] multiplies prev[:, -(i+1), :] — i.e., the reversed time axis.
        # Equivalently, flip w along the W axis to align with prev directly.
        w_rev = torch.flip(w, dims=[-1])                               # [K, W]; w_rev[k, j] = w[k, W-1-j]
        # prev[s, j, k] * w_rev[k, j], sum over j → [s, k].
        mem_t = torch.einsum("swk,kw->sk", prev, w_rev) * lam          # [S, K]
        return mem_t

    def extra_repr(self) -> str:
        return f"K={self.K}, W={self.W}"
