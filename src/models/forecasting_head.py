"""M8 — Escalation forecasting head.

Takes a temporal window F_{t-L:t}(r, :) ∈ R^{L × K} and predicts ACLED-style
count targets at horizons {1, 2, 4} weeks. Two likelihoods are supported:
  * Poisson (default, for count targets that aren't over-dispersed)
  * Negative Binomial NB2 with learned dispersion (recommended for fatalities,
    which are strongly over-dispersed — see audit §1.2)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ForecastConfig:
    num_cleavages: int = 6
    window_len: int = 8
    hidden: int = 128
    dropout: float = 0.2
    horizons: tuple[int, ...] = (1, 2, 4)
    num_targets: int = 2  # protests, fatalities


class EscalationHead(nn.Module):
    def __init__(self, cfg: ForecastConfig | None = None):
        super().__init__()
        self.cfg = cfg or ForecastConfig()
        in_dim = self.cfg.window_len * (self.cfg.num_cleavages + 1)  # + aggregate
        out_dim = len(self.cfg.horizons) * self.cfg.num_targets
        self.net = nn.Sequential(
            nn.Linear(in_dim, self.cfg.hidden),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden, self.cfg.hidden),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden, out_dim),
        )

    def forward(self, F_k: torch.Tensor, F_agg: torch.Tensor) -> torch.Tensor:
        """
        F_k:  [N, L, K]   per-cleavage windowed friction
        F_agg:[N, L]      aggregate
        Returns log-rates [N, H, 2]   (log λ for Poisson head)
        """
        x = torch.cat([F_k, F_agg.unsqueeze(-1)], dim=-1)  # [N, L, K+1]
        x = x.flatten(1)
        y = self.net(x)
        return y.view(-1, len(self.cfg.horizons), self.cfg.num_targets)


def poisson_nll(log_rate: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Poisson negative log-likelihood (target can be float count).

    Clamps log_rate for numerical stability when the head isn't warmed up yet.
    """
    log_rate = log_rate.clamp(min=-20.0, max=20.0)
    return (torch.exp(log_rate) - y * log_rate).mean()


# ------------------------ Negative Binomial ------------------------ #

class NegativeBinomialHead(nn.Module):
    """Learnable dispersion wrapper; the mean is produced by `EscalationHead`.

    The model uses the NB2 (variance = mu + mu^2 / r) parameterisation:
      * log_mu   -- per-target log-mean (float, passed in by the caller)
      * r        -- positive dispersion (softplus of a learned raw parameter)

    `log_mu` has shape [..., H, T_targets] just like `EscalationHead` output;
    `y` must broadcast to that shape. `r` is one parameter per (horizon,
    target) combination so protests and fatalities can have different
    dispersion.
    """

    def __init__(self, num_horizons: int, num_targets: int = 2,
                 init_log_r: float = 0.0) -> None:
        super().__init__()
        self.num_horizons = num_horizons
        self.num_targets = num_targets
        # Raw parameter; softplus(raw) = r so r stays > 0.
        self.raw_log_r = nn.Parameter(
            torch.full((num_horizons, num_targets), float(init_log_r))
        )

    def dispersion(self) -> torch.Tensor:
        return F.softplus(self.raw_log_r) + 1e-3  # [H, T_targets]

    def nll(self, log_mu: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return negative_binomial_nll(log_mu, y, self.dispersion())

    def forward(self, log_mu: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.nll(log_mu, y)


def negative_binomial_nll(
    log_mu: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """NB2 negative log-likelihood.

    log p(y | mu, r) = lgamma(y + r) - lgamma(r) - lgamma(y + 1)
                       + r * log(r / (r + mu)) + y * log(mu / (r + mu))

    Parameters
    ----------
    log_mu : Tensor [..., H, T_targets]
    y      : Tensor broadcastable to `log_mu`
    r      : Tensor broadcastable to `log_mu` (dispersion; scalar or [H,T])
    """
    log_mu = log_mu.clamp(min=-20.0, max=20.0)
    mu = torch.exp(log_mu)
    r = r.to(log_mu.dtype).clamp(min=1e-3)
    # work in log-space where possible for stability
    log_r_over_rmu = torch.log(r + eps) - torch.log(r + mu + eps)
    log_mu_over_rmu = log_mu - torch.log(r + mu + eps)
    y_f = y.to(log_mu.dtype)
    ll = (
        torch.lgamma(y_f + r)
        - torch.lgamma(r)
        - torch.lgamma(y_f + 1.0)
        + r * log_r_over_rmu
        + y_f * log_mu_over_rmu
    )
    return (-ll).mean()


def build_windows(F_k: torch.Tensor, F_agg: torch.Tensor,
                  window_len: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """
    F_k: [S, T, K], F_agg: [S, T]
    Returns:
        Xk:  [N, L, K]
        Xag: [N, L]
        t_end: list[int] of the last week index in each window (length N)
    """
    S, T, K = F_k.shape
    L = window_len
    t_ends = list(range(L - 1, T))
    N = S * len(t_ends)
    Xk = torch.stack([F_k[:, te - L + 1: te + 1, :] for te in t_ends], dim=1)  # [S, N_t, L, K]
    Xag = torch.stack([F_agg[:, te - L + 1: te + 1] for te in t_ends], dim=1)  # [S, N_t, L]
    Xk = Xk.reshape(N, L, K)
    Xag = Xag.reshape(N, L)
    return Xk, Xag, t_ends
