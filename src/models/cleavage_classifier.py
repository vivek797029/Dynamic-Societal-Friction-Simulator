"""M6a — Multi-label cleavage classifier.

Two-stage approach:
  1. Weak labels from keyword lexicon (per cleavage, English + Hindi seed terms).
  2. Train a small LoRA head on top of MuRIL CLS using BCE with label smoothing.

This gives every article a probability vector over 6 cleavage types.

v2 fix (audit §1.1): the old `weak_label()` did a bare substring check
`w.lower() in text.lower()`, which means "bjp" matched "bjpa", "aap" matched
"aapta", and "obc" matched "obcuring". We now precompile IGNORECASE regexes
with a language-aware word boundary so token hits are real.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

# torch is only needed for the neural-network classes below.
# Lazy-import so that CLEAVAGES / weak_label / weak_labels can be used
# from the query layer (and in tests) without a GPU/torch install.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

CLEAVAGES = ["communal", "caste", "political_party", "centre_state", "economic", "linguistic"]

# Keyword lexicon — deliberately seed-level; the LoRA head generalizes beyond.
LEXICON: dict[str, list[str]] = {
    "communal": [
        "communal", "hindu-muslim", "riot", "mosque", "temple", "hijab",
        "cow vigilante", "lynching", "hate crime", "love jihad",
        "मुस्लिम", "हिंदू", "मंदिर", "मस्जिद", "साम्प्रदायिक",
    ],
    "caste": [
        "dalit", "adivasi", "scheduled caste", "scheduled tribe", "obc",
        "caste violence", "honor killing", "reservation",
        "दलित", "आदिवासी", "जाति", "आरक्षण",
    ],
    "political_party": [
        "bjp", "congress", "aap", "tmc", "shiv sena", "dmk", "aiadmk",
        "opposition", "modi", "rahul gandhi", "mamata", "kejriwal",
        "भाजपा", "कांग्रेस", "विपक्ष",
    ],
    "centre_state": [
        "governor", "centre-state", "article 356", "gst council",
        "state autonomy", "federalism", "central government", "state government",
        "राज्यपाल", "केंद्र सरकार", "राज्य सरकार",
    ],
    "economic": [
        "farmer", "farm law", "msp", "minimum wage", "strike", "layoff",
        "inflation", "unemployment", "labour", "trade union", "agriculture",
        "किसान", "मजदूर", "महंगाई",
    ],
    "linguistic": [
        "hindi imposition", "tamil", "marathi signboard", "language row",
        "mother tongue", "three-language formula", "local language",
        "हिंदी थोपना", "मातृभाषा",
    ],
}


# Word-boundary regex per cleavage. `(?<!\w)` and `(?!\w)` work for Latin
# tokens; Devanagari and other scripts have no case so IGNORECASE is a no-op
# for them, and they also don't share `\w` word characters with Latin so
# `\w`-boundaries still split cleanly. `re.UNICODE` is the default in Py3.
def _compile_lexicon() -> dict[str, re.Pattern]:
    out: dict[str, re.Pattern] = {}
    for c, ws in LEXICON.items():
        parts = [re.escape(w) for w in ws if w]
        if not parts:
            out[c] = re.compile(r"$^")  # matches nothing
            continue
        out[c] = re.compile(
            r"(?<!\w)(?:" + "|".join(parts) + r")(?!\w)",
            re.IGNORECASE | re.UNICODE,
        )
    return out


_PATTERNS: dict[str, re.Pattern] = _compile_lexicon()


def weak_label(text: str) -> np.ndarray:
    """Return a [K] multi-hot vector from lexicon hits.

    Word-boundary match so "bjp" matches "the BJP did X" but not "bjpa".
    """
    y = np.zeros(len(CLEAVAGES), dtype=np.float32)
    if not text:
        return y
    for i, c in enumerate(CLEAVAGES):
        if _PATTERNS[c].search(text):
            y[i] = 1.0
    return y


def weak_labels(texts: Iterable[str]) -> np.ndarray:
    return np.stack([weak_label(t) for t in texts])


@dataclass
class CleavageConfig:
    hidden: int = 768
    num_labels: int = len(CLEAVAGES)
    dropout: float = 0.1
    label_smoothing: float = 0.05


if _TORCH_AVAILABLE:
    class CleavageHead(nn.Module):  # type: ignore[misc]
        """Tiny MLP head over a frozen MuRIL CLS embedding."""

        def __init__(self, cfg: "CleavageConfig | None" = None):
            super().__init__()
            self.cfg = cfg or CleavageConfig()
            self.net = nn.Sequential(
                nn.Linear(self.cfg.hidden, self.cfg.hidden // 2),
                nn.GELU(),
                nn.Dropout(self.cfg.dropout),
                nn.Linear(self.cfg.hidden // 2, self.cfg.num_labels),
            )

        def forward(self, cls: "torch.Tensor") -> "torch.Tensor":
            return self.net(cls)

    def bce_with_smoothing(
        logits: "torch.Tensor", y: "torch.Tensor", smoothing: float = 0.05
    ) -> "torch.Tensor":
        y_smoothed = y * (1 - smoothing) + 0.5 * smoothing
        return F.binary_cross_entropy_with_logits(logits, y_smoothed)

else:  # pragma: no cover
    def CleavageHead(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("CleavageHead requires torch. Install pytorch first.")

    def bce_with_smoothing(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("bce_with_smoothing requires torch. Install pytorch first.")
