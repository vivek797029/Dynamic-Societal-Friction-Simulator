"""M5 — TRUST LEARNING (v2).

Upgrades vs v1:
  * EMA consensus centroids per cluster (stable reference, variance reduction).
  * Hard-negative mining in the contrastive loss.
  * Source-diversity regularizer to prevent trivial τ solutions.
  * τ-clamp in the consensus gate to avoid degenerate fixed points.
  * Everything consumes a *unified* DSFSArticleEncoder via its dict output.

The losses are functions on top of the encoder output — no encoder class here.
Kept for backward compat: the older `TrustEncoder` + `source_trust_from_logits`
still import cleanly, but the new training loop uses `DSFSArticleEncoder`.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-export for backward compat.
from .article_encoder import DSFSArticleEncoder, EncoderConfig  # noqa: F401


# -------- EMA centroid bank --------

class ClusterCentroidBank:
    """Keeps a running unit-norm centroid per cluster_id across batches.

    Updates: μ̂_c ← normalize((1-α) μ̂_c + α · weighted-batch-mean).
    """
    def __init__(self, dim: int, momentum: float = 0.95, device: str = "cuda"):
        self.dim = dim
        self.m = momentum
        self.device = device
        self.centroids: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, z: torch.Tensor, trust_gate: torch.Tensor,
               cluster_ids: torch.Tensor) -> None:
        """z: [B, D] unit-norm; trust_gate: [B] in [0,1]; cluster_ids: [B]."""
        unique = torch.unique(cluster_ids)
        for c in unique.tolist():
            if c < 0:
                continue
            mask = (cluster_ids == c)
            zc = z[mask]
            gc = trust_gate[mask]
            if zc.size(0) == 0:
                continue
            gw = gc / (gc.sum() + 1e-8)
            mu_batch = (gw.unsqueeze(-1) * zc).sum(dim=0)
            mu_batch = F.normalize(mu_batch, dim=-1)
            prev = self.centroids.get(c)
            if prev is None:
                self.centroids[c] = mu_batch.detach()
            else:
                new = self.m * prev + (1.0 - self.m) * mu_batch
                self.centroids[c] = F.normalize(new, dim=-1).detach()

    def lookup(self, cluster_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (μ, has_centroid) for each row; rows without a centroid get
        a zero vector and has_centroid=False so the caller can mask them out."""
        B = cluster_ids.size(0)
        mu = torch.zeros(B, self.dim, device=cluster_ids.device)
        has = torch.zeros(B, dtype=torch.bool, device=cluster_ids.device)
        for i, c in enumerate(cluster_ids.tolist()):
            if c in self.centroids:
                mu[i] = self.centroids[c].to(mu.device)
                has[i] = True
        return mu, has


# -------- Losses --------

def agreement_contrastive_loss(z: torch.Tensor,
                               cluster_ids: torch.Tensor,
                               jaccard: torch.Tensor,
                               tau: float = 0.07,
                               jaccard_weight: float = 0.5,
                               hard_negative_alpha: float = 2.0) -> torch.Tensor:
    """Supervised-contrastive with factual reweighting + hard-negative focusing.

    The hard-negative term multiplies negative similarities by
    softmax_α(sim) so that the hardest negatives dominate the denominator;
    equivalent in spirit to SupCon + focal reweighting.
    """
    B = z.size(0)
    sim = z @ z.t() / tau
    self_mask = torch.eye(B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(self_mask, -1e9)

    pos_mask = (cluster_ids.unsqueeze(0) == cluster_ids.unsqueeze(1)) & (cluster_ids.unsqueeze(0) >= 0)
    pos_mask = pos_mask & (~self_mask)
    has_pos = pos_mask.any(dim=1)
    if not has_pos.any():
        return sim.sum() * 0.0

    # Hard-negative reweighting of the *denominator* only.
    neg_mask = (~pos_mask) & (~self_mask)
    # softmax over negatives with temperature α to focus on the hardest.
    neg_sim = sim.masked_fill(~neg_mask, -1e9)
    w_neg = torch.softmax(neg_sim * hard_negative_alpha, dim=1).detach()
    # Effective logits: positives unchanged, negatives reweighted by B·w_neg
    # (average-neg weight is 1 in an unweighted SupCon, so we renormalize).
    logits_eff = sim.clone()
    logits_eff = torch.where(neg_mask, sim + torch.log(B * w_neg + 1e-9), logits_eff)

    w_pos = 1.0 + jaccard_weight * jaccard
    log_prob = logits_eff - torch.logsumexp(logits_eff, dim=1, keepdim=True)
    numer = (log_prob * pos_mask.float() * w_pos).sum(dim=1)
    denom = (pos_mask.float() * w_pos).sum(dim=1).clamp(min=1e-8)
    loss_per_row = -numer / denom
    return loss_per_row[has_pos].mean()


def consensus_deviation_loss(z: torch.Tensor,
                             s_logits: torch.Tensor,
                             cluster_ids: torch.Tensor,
                             centroid_bank: ClusterCentroidBank,
                             gate_clamp: tuple[float, float] = (0.1, 0.9)) -> torch.Tensor:
    """Penalty gated by clamped σ(ŝ_i) against an EMA cluster centroid.

    The centroid_bank is updated AFTER this loss is computed (caller's
    responsibility). Rows whose cluster has no centroid yet are skipped.
    """
    mu, has = centroid_bank.lookup(cluster_ids)
    if not has.any():
        return z.sum() * 0.0
    zc = z[has]
    sc = s_logits[has]
    muh = mu[has]
    cos = (zc * muh).sum(dim=-1)
    gate = torch.sigmoid(sc).clamp(*gate_clamp)
    return (gate * (1.0 - cos)).mean()


def source_diversity_regularizer(s_logits: torch.Tensor,
                                 source_ids: list[str],
                                 target_std: float = 0.25) -> torch.Tensor:
    """Encourage τ to spread (penalize collapse to 0.5 AND to extremes).

    Groups article-level logits by source, takes the mean per source, then
    penalizes |std(τ_s) - target_std|. This is a distributional prior that
    keeps the trust axis identifiable without hard labels.
    """
    tau_by_src: dict[str, list[torch.Tensor]] = defaultdict(list)
    for sid, v in zip(source_ids, s_logits):
        tau_by_src[sid].append(v)
    if len(tau_by_src) < 4:
        return s_logits.sum() * 0.0
    taus = torch.stack([torch.sigmoid(torch.stack(vs).mean())
                        for vs in tau_by_src.values()])
    std = taus.std()
    # L2 around target std.
    return (std - target_std) ** 2


# -------- Aggregation --------

def source_trust_from_logits(source_ids: list[str],
                             s_logits_per_article: torch.Tensor) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    vals = s_logits_per_article.detach().cpu().tolist()
    for sid, v in zip(source_ids, vals):
        acc[sid].append(v)
    return {s: float(torch.sigmoid(torch.tensor(sum(vs) / len(vs))))
            for s, vs in acc.items()}


# -------- Backward-compat shim --------
# Older code imported `TrustEncoder, TrustConfig`; route them to the unified encoder.

@dataclass
class TrustConfig(EncoderConfig):
    """Alias that keeps the old field names working."""
    contrastive_tau: float = 0.07
    jaccard_weight: float = 0.5
    consensus_weight: float = 0.3


class TrustEncoder(DSFSArticleEncoder):
    """Alias so old `from .trust_learner import TrustEncoder` still works.

    New code should import `DSFSArticleEncoder` directly.
    """
    def __init__(self, cfg: TrustConfig | None = None):
        c = cfg or TrustConfig()
        # Copy EncoderConfig fields across.
        super().__init__(EncoderConfig(
            model_name=c.model_name, proj_dim=c.proj_dim,
            num_cleavages=c.num_cleavages, cleavage_emb_dim=c.cleavage_emb_dim,
            dropout=c.dropout, use_lora=c.use_lora,
            lora_rank=c.lora_rank, lora_alpha=c.lora_alpha,
            lora_dropout=c.lora_dropout,
        ))

    def forward(self, input_ids, attention_mask):  # type: ignore[override]
        """Legacy tuple-return for old call sites."""
        out = super().forward(input_ids, attention_mask)
        return out["z"], out["trust_logit"]
