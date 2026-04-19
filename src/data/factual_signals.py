"""M4. Factual signal extractor.

Represents each article as a set of verifiable atoms:
  - named entities (PER / ORG / GPE) via XLM-R NER
  - numeric claims with units (deaths, injuries, arrests, ages)
  - (subject, predicate, object) triples from a lightweight rule-based OpenIE
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


NUMERIC_UNIT_PATTERNS = [
    # matches: "12 killed", "3 injured", "dozens arrested"
    (r"(\d+)\s+(killed|dead|died|fatalities)", "casualty"),
    (r"(\d+)\s+(injured|wounded|hurt)", "injury"),
    (r"(\d+)\s+(arrested|detained|held|taken into custody)", "arrest"),
    (r"(\d+)\s+(year[- ]old|yr[- ]old)", "age"),
    (r"(?:₹|Rs\.?|INR)\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(crore|lakh|lakhs|cr)?", "money"),
    (r"(\d+)\s+(protesters?|demonstrators?|people|civilians?|officers?|police|soldiers?)", "count"),
]


@dataclass
class FactualAtoms:
    entities: set[str] = field(default_factory=set)
    numerics: set[tuple[str, float]] = field(default_factory=set)  # (unit, value)
    triples: set[tuple[str, str, str]] = field(default_factory=set)

    def to_token_set(self) -> set[str]:
        """Flat token set used for Jaccard in trust learning."""
        toks: set[str] = set()
        toks |= {f"ENT::{e.lower()}" for e in self.entities}
        toks |= {f"NUM::{u}::{v:.0f}" for (u, v) in self.numerics}
        toks |= {f"TRI::{s.lower()}::{p.lower()}::{o.lower()}" for (s, p, o) in self.triples}
        return toks


def extract_numerics(text: str) -> set[tuple[str, float]]:
    out: set[tuple[str, float]] = set()
    t = text.lower()
    for pat, unit in NUMERIC_UNIT_PATTERNS:
        for m in re.finditer(pat, t):
            raw = m.group(1).replace(",", "")
            try:
                v = float(raw)
            except ValueError:
                continue
            if unit == "money" and m.lastindex and m.lastindex >= 2:
                scale = {"crore": 1e7, "cr": 1e7, "lakh": 1e5, "lakhs": 1e5}.get(m.group(2) or "", 1.0)
                v *= scale
            out.add((unit, v))
    return out


class NERExtractor:
    """Thin wrapper around a HuggingFace token-classification pipeline.

    Defaults to Davlan/xlm-roberta-base-ner-hrl which covers Hindi + English.
    """
    def __init__(self, model_name: str = "Davlan/xlm-roberta-base-ner-hrl", device: int = -1):
        self.model_name = model_name
        self.device = device
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline  # lazy import
            self._pipe = pipeline(
                "token-classification",
                model=self.model_name,
                aggregation_strategy="simple",
                device=self.device,
            )
        return self._pipe

    def __call__(self, text: str) -> set[str]:
        pipe = self._load()
        out = pipe(text[:4000])
        keep = {"PER", "ORG", "LOC", "GPE"}
        return {h["word"].strip() for h in out if h.get("entity_group") in keep and len(h["word"].strip()) > 2}


def extract_simple_triples(text: str, entities: Iterable[str]) -> set[tuple[str, str, str]]:
    """Very simple entity1-verb-entity2 extraction with a verb whitelist."""
    action_verbs = {
        "killed", "attacked", "arrested", "protested", "clashed", "injured",
        "demanded", "accused", "banned", "released", "rejected", "approved",
        "raided", "charged", "filed", "lathi-charged", "stoned",
    }
    ent_list = list(entities)
    out: set[tuple[str, str, str]] = set()
    if len(ent_list) < 2:
        return out
    t = text.lower()
    for i, a in enumerate(ent_list):
        for j, b in enumerate(ent_list):
            if i == j:
                continue
            pat = rf"{re.escape(a.lower())}\s+(\w+)\s+(?:\w+\s+){{0,5}}{re.escape(b.lower())}"
            for m in re.finditer(pat, t):
                verb = m.group(1)
                if verb in action_verbs:
                    out.add((a, verb, b))
    return out


def extract_atoms(text: str, ner: NERExtractor | None = None) -> FactualAtoms:
    ents = ner(text) if ner is not None else set()
    nums = extract_numerics(text)
    tris = extract_simple_triples(text, ents)
    return FactualAtoms(entities=ents, numerics=nums, triples=tris)


def jaccard(a: FactualAtoms, b: FactualAtoms) -> float:
    ta, tb = a.to_token_set(), b.to_token_set()
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
