"""M6b — Hostility regression head conditioned on cleavage.

Output: h(x, k) ∈ [0, 1], where k is a cleavage index. Conditioning is done
via a learned cleavage embedding concatenated to CLS.

Weak supervision target: we build a rough hostility target from GDELT
AvgTone + GKG V2Tone (negative tone → high hostility) and refine later with
a small human-labeled subset. This function constructs the target.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cleavage_classifier import CLEAVAGES


@dataclass
class HostilityConfig:
    hidden: int = 768
    cleavage_emb_dim: int = 32
    num_cleavages: int = len(CLEAVAGES)
    dropout: float = 0.1


class HostilityHead(nn.Module):
    def __init__(self, cfg: HostilityConfig | None = None):
        super().__init__()
        self.cfg = cfg or HostilityConfig()
        self.cleavage_emb = nn.Embedding(self.cfg.num_cleavages, self.cfg.cleavage_emb_dim)
        in_dim = self.cfg.hidden + self.cfg.cleavage_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(in_dim // 2, 1),
        )

    def forward(self, cls: torch.Tensor, cleavage_idx: torch.Tensor) -> torch.Tensor:
        """cls: [B, H], cleavage_idx: [B] long → logit h ∈ ℝ; apply sigmoid for score."""
        ke = self.cleavage_emb(cleavage_idx)
        x = torch.cat([cls, ke], dim=-1)
        return self.net(x).squeeze(-1)


def tone_to_hostility(avg_tone: float) -> float:
    """Map GDELT AvgTone (roughly -10..+10) to [0,1] hostility score."""
    # Clip and invert: -10 → 1, +10 → 0
    v = max(-10.0, min(10.0, float(avg_tone)))
    return 0.5 * (1.0 - v / 10.0)


def weak_hostility_targets(avg_tones: np.ndarray) -> np.ndarray:
    return np.clip(0.5 * (1.0 - avg_tones / 10.0), 0.0, 1.0).astype(np.float32)


def hostility_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """MSE on sigmoid — robust to noisy weak labels."""
    return F.mse_loss(torch.sigmoid(logits), y)
