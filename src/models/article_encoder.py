"""Unified article encoder: one module owns the MuRIL+LoRA backbone plus every
per-article head. Replaces the stitched-together `TrustEncoder + CleavageHead +
HostilityHead + function-attribute proj` setup with a single `state_dict`.

Interface (everything takes raw `(input_ids, attention_mask)` or a cached CLS):
  enc = DSFSArticleEncoder(cfg)
  out = enc(input_ids, attention_mask)
  out["cls"]          # [B, H]         raw CLS (768 for MuRIL-base)
  out["z"]            # [B, D]         unit-norm contrastive projection
  out["trust_logit"]  # [B]            per-article trust logit ŝ_i
  out["cleavage"]     # [B, K]         cleavage logits
  out["hostility"]    # [B, K]         hostility logits (one per cleavage)

Training uses `forward_all`; inference paths that only need some heads can
call `cls_only()` and then hit the individual head methods without a backbone
pass — useful after caching CLS to disk.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

try:
    from peft import LoraConfig, get_peft_model
    _HAS_PEFT = True
except Exception:  # pragma: no cover
    _HAS_PEFT = False


@dataclass
class EncoderConfig:
    model_name: str = "google/muril-base-cased"
    proj_dim: int = 128
    num_cleavages: int = 6
    cleavage_emb_dim: int = 32
    dropout: float = 0.1
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1


class DSFSArticleEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig | None = None):
        super().__init__()
        self.cfg = cfg or EncoderConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        backbone = AutoModel.from_pretrained(self.cfg.model_name)
        if self.cfg.use_lora and _HAS_PEFT:
            lora = LoraConfig(
                r=self.cfg.lora_rank,
                lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout,
                target_modules=["query", "value"],
                bias="none",
            )
            backbone = get_peft_model(backbone, lora)
        self.backbone = backbone
        H = self._hidden_size()

        # Heads.
        self.proj = nn.Sequential(
            nn.Linear(H, H), nn.GELU(), nn.Linear(H, self.cfg.proj_dim)
        )
        self.trust_head = nn.Linear(H, 1)
        self.cleavage_head = nn.Sequential(
            nn.Linear(H, H // 2), nn.GELU(), nn.Dropout(self.cfg.dropout),
            nn.Linear(H // 2, self.cfg.num_cleavages),
        )
        self.cleavage_emb = nn.Embedding(self.cfg.num_cleavages, self.cfg.cleavage_emb_dim)
        self.hostility_head = nn.Sequential(
            nn.Linear(H + self.cfg.cleavage_emb_dim, H // 2),
            nn.GELU(), nn.Dropout(self.cfg.dropout),
            nn.Linear(H // 2, 1),
        )

    def _hidden_size(self) -> int:
        b = self.backbone
        # Handle PEFT-wrapped vs raw.
        if hasattr(b, "config") and hasattr(b.config, "hidden_size"):
            return b.config.hidden_size
        return b.base_model.config.hidden_size  # peft wrap

    # ---------- backbone ----------
    def cls_only(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]

    # ---------- per-head ----------
    def project(self, cls: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(cls), dim=-1)

    def trust_logit(self, cls: torch.Tensor) -> torch.Tensor:
        return self.trust_head(cls).squeeze(-1)

    def cleavage_logits(self, cls: torch.Tensor) -> torch.Tensor:
        return self.cleavage_head(cls)

    def hostility_per_cleavage(self, cls: torch.Tensor) -> torch.Tensor:
        """Return [B, K] hostility logits by evaluating the head K times with
        cleavage embedding concatenated."""
        B, H = cls.shape
        K = self.cfg.num_cleavages
        ke = self.cleavage_emb.weight  # [K, E]
        # broadcast-concat: [B, K, H+E]
        cls_b = cls.unsqueeze(1).expand(B, K, H)
        ke_b = ke.unsqueeze(0).expand(B, K, ke.size(-1))
        x = torch.cat([cls_b, ke_b], dim=-1).reshape(B * K, -1)
        y = self.hostility_head(x).reshape(B, K)
        return y

    # ---------- one-shot ----------
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        cls = self.cls_only(input_ids, attention_mask)
        return {
            "cls": cls,
            "z": self.project(cls),
            "trust_logit": self.trust_logit(cls),
            "cleavage": self.cleavage_logits(cls),
            "hostility": self.hostility_per_cleavage(cls),
        }
