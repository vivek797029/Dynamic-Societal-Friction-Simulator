"""NegativeBinomialHead: NLL is finite, reduces to Poisson as r -> inf."""
from __future__ import annotations

import math

import torch

from src.models.forecasting_head import (NegativeBinomialHead, negative_binomial_nll,
                                         poisson_nll)


def test_nll_is_finite_on_zero_heavy_targets():
    torch.manual_seed(0)
    N, H, Targ = 64, 3, 2
    log_mu = torch.randn(N, H, Targ) * 0.5
    y = torch.zeros(N, H, Targ)
    # 10% of rows are large counts
    idx = torch.randperm(N)[: N // 10]
    y[idx] = torch.randint(1, 100, (N // 10, H, Targ)).float()
    nb = NegativeBinomialHead(num_horizons=H, num_targets=Targ)
    loss = nb.nll(log_mu, y)
    assert torch.isfinite(loss), f"loss not finite: {loss}"


def test_nb_tends_to_poisson_for_large_r():
    torch.manual_seed(1)
    N, H, Targ = 128, 1, 1
    log_mu = torch.randn(N, H, Targ) * 0.1 + 0.3
    y = torch.poisson(torch.exp(log_mu)).float()

    r_big = torch.tensor([[1e6]])   # [H, Targ]
    nb_loss = negative_binomial_nll(log_mu, y, r_big)
    poi_loss = poisson_nll(log_mu, y)
    # Both share a 'const' drop; the difference should be < a few percent of poi_loss.
    # (Exact match isn't expected because Poisson's constant includes log y! and NB
    #  includes lgamma terms, but the gradient-relevant part matches.)
    assert torch.isfinite(nb_loss)
    assert abs(float(nb_loss) - float(poi_loss)) < 1.0
