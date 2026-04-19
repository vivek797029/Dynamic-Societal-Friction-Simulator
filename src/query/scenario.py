"""Scenario extraction: free-form prompt -> structured Scenario.

A `Scenario` is the unit of input to the intervention simulator (§3). We
keep it deliberately minimal so it can be auto-filled AND hand-edited:

    policy_type       : short tag ("farm_law", "hijab_ban", ...)
    affected_states   : subset of india_geo.STATES (empty list = nationwide)
    cleavages         : subset of cleavage_classifier.CLEAVAGES
    effective_week    : int iso_week index (see india_geo.iso_week_index)
    severity          : float in [0, 1]  -- how large the intended perturbation is
    raw_text          : the original user prompt (for citations / narratives)

Two extractors ship:
  * `extract_rule_based`  -- no LLM needed, keyword matcher, good for tests
                             and as a fallback if the LLM returns nothing.
  * `extract_llm`         -- prompts the LLM with a JSON schema hint.

`extract_scenario()` tries the LLM first, falls back to rules.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

from ..data.india_geo import STATES, STATE_TO_IDX, iso_week_index
from ..models.cleavage_classifier import CLEAVAGES
from .llm import LLMProvider, StubLLM


# ------------------------------ dataclass ------------------------------ #

@dataclass
class Scenario:
    policy_type: str = "unspecified"
    affected_states: list[str] = field(default_factory=list)  # [] = nationwide
    cleavages: list[str] = field(default_factory=list)        # [] = all
    effective_week: int | None = None
    severity: float = 0.5
    raw_text: str = ""
    source: str = "rule_based"   # "llm" | "rule_based" | "user_edited"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        keep = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**keep)

    def validate(self) -> list[str]:
        """Return a list of human-readable validation issues (empty = valid)."""
        issues = []
        for s in self.affected_states:
            if s not in STATE_TO_IDX:
                issues.append(f"unknown state: {s!r}")
        for c in self.cleavages:
            if c not in CLEAVAGES:
                issues.append(f"unknown cleavage: {c!r}")
        if not (0.0 <= self.severity <= 1.0):
            issues.append(f"severity must be in [0,1], got {self.severity}")
        return issues


# --------------------------- rule-based extractor ---------------------- #

# Keyword -> cleavage mapping. Deliberately lexical; the LLM path handles
# nuance. Lowercased substring matches (word-boundary to avoid 'obc' ⊂ 'obcuring').
_CLEAVAGE_HINTS: dict[str, list[str]] = {
    "communal": ["hindu", "muslim", "mosque", "temple", "hijab", "communal", "riot"],
    "caste": ["dalit", "adivasi", "obc", "reservation", "caste", "scheduled"],
    "political_party": ["bjp", "congress", "aap", "tmc", "dmk", "shiv sena"],
    "centre_state": ["governor", "centre", "federal", "article 356", "gst council"],
    "economic": ["farm", "farmer", "msp", "inflation", "unemploy", "layoff", "labour"],
    "linguistic": ["hindi imposition", "tamil", "language row", "marathi signboard"],
}

# Rough policy-type keyword tags (extend as needed).
_POLICY_TYPES: dict[str, list[str]] = {
    "farm_law": ["farm law", "farm bill", "msp", "farmer protest", "kisan"],
    "caa_nrc": ["caa", "citizenship amendment", "nrc"],
    "hijab_ban": ["hijab"],
    "article_370": ["article 370", "kashmir special status"],
    "demonetization": ["demonetization", "demonetisation"],
    "triple_talaq": ["triple talaq"],
    "gst": ["gst reform", "gst rate"],
    "reservation": ["reservation bill", "quota bill"],
}


_STATE_PATTERNS: dict[str, re.Pattern] = {
    s: re.compile(r"(?<!\w)" + re.escape(s.lower()) + r"(?!\w)") for s in STATES
}

# Common short forms.
_STATE_ALIASES: dict[str, str] = {
    "up": "Uttar Pradesh", "mp": "Madhya Pradesh", "tn": "Tamil Nadu",
    "wb": "West Bengal", "ap": "Andhra Pradesh", "hp": "Himachal Pradesh",
    "j&k": "Jammu and Kashmir", "j & k": "Jammu and Kashmir",
    "ncr": "Delhi", "ncr delhi": "Delhi",
    "ncr of delhi": "Delhi", "nct of delhi": "Delhi",
    "orissa": "Odisha", "uttaranchal": "Uttarakhand",
}


def _hit(text: str, needles: list[str]) -> bool:
    t = text.lower()
    for n in needles:
        pat = re.compile(r"(?<!\w)" + re.escape(n) + r"(?!\w)")
        if pat.search(t):
            return True
    return False


def _extract_states(text: str) -> list[str]:
    t = text.lower()
    hits = set()
    for s, pat in _STATE_PATTERNS.items():
        if pat.search(t):
            hits.add(s)
    for alias, canon in _STATE_ALIASES.items():
        pat = re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)")
        if pat.search(t):
            hits.add(canon)
    return sorted(hits)


def _extract_cleavages(text: str) -> list[str]:
    return [c for c in CLEAVAGES if _hit(text, _CLEAVAGE_HINTS[c])]


def _extract_policy_type(text: str) -> str:
    for tag, needles in _POLICY_TYPES.items():
        if _hit(text, needles):
            return tag
    return "unspecified"


_SEVERITY_HINTS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\b(nationwide|emergency|drastic|sweeping|unprecedented)\b", re.I), 0.9),
    (re.compile(r"\b(major|significant|large-?scale|broad)\b", re.I), 0.75),
    (re.compile(r"\b(moderate|partial|phased)\b", re.I), 0.5),
    (re.compile(r"\b(minor|limited|local|small)\b", re.I), 0.25),
]


def _extract_severity(text: str) -> float:
    for pat, v in _SEVERITY_HINTS:
        if pat.search(text):
            return v
    return 0.5


_WEEK_RE = re.compile(r"\bweek\s+(\d{1,4})\b", re.I)
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def _extract_effective_week(text: str) -> int | None:
    m = _WEEK_RE.search(text)
    if m:
        return int(m.group(1))
    m = _DATE_RE.search(text)
    if m:
        try:
            return iso_week_index(dt.date.fromisoformat(m.group(1)))
        except ValueError:
            return None
    return None


def extract_rule_based(text: str) -> Scenario:
    """Deterministic extractor used as fallback + in tests."""
    return Scenario(
        policy_type=_extract_policy_type(text),
        affected_states=_extract_states(text),
        cleavages=_extract_cleavages(text),
        effective_week=_extract_effective_week(text),
        severity=_extract_severity(text),
        raw_text=text,
        source="rule_based",
    )


# ------------------------------ LLM path ------------------------------- #

_SCHEMA_HINT = """{
  "policy_type": "short_snake_case_tag",
  "affected_states": ["Exact state name from India", ...],
  "cleavages": ["communal"|"caste"|"political_party"|"centre_state"|"economic"|"linguistic"],
  "effective_week": integer or null,
  "severity": 0.0 to 1.0
}"""

_SYSTEM = (
    "You are extracting a policy/event scenario from a user's question about "
    "societal friction in India. Return ONLY valid JSON following the schema. "
    "States must be exact: use canonical names like 'Tamil Nadu', 'Uttar Pradesh'. "
    "Cleavages are a closed set of 6 tags. If the prompt is not about a "
    "specific policy/event, set policy_type to 'unspecified' and leave lists empty."
)


def extract_llm(text: str, llm: LLMProvider) -> Scenario | None:
    raw = llm.complete_json(text, schema_hint=_SCHEMA_HINT, system=_SYSTEM)
    if not raw:
        return None
    try:
        # Coerce the LLM's loose JSON into the strict dataclass shape.
        states = [s for s in raw.get("affected_states") or [] if s in STATE_TO_IDX]
        cleavs = [c for c in raw.get("cleavages") or [] if c in CLEAVAGES]
        scen = Scenario(
            policy_type=str(raw.get("policy_type") or "unspecified"),
            affected_states=states,
            cleavages=cleavs,
            effective_week=(int(raw["effective_week"])
                            if raw.get("effective_week") is not None else None),
            severity=max(0.0, min(1.0, float(raw.get("severity") or 0.5))),
            raw_text=text,
            source="llm",
        )
        return scen
    except (TypeError, ValueError):
        return None


# --------------------------- compose both ------------------------------ #

def extract_scenario(text: str, llm: LLMProvider | None = None) -> Scenario:
    """LLM first, rule-based fallback. Merges results when LLM is partial."""
    llm = llm or StubLLM()
    rb = extract_rule_based(text)
    llm_scen = extract_llm(text, llm) if not isinstance(llm, StubLLM) else None
    if llm_scen is None:
        return rb
    # Merge: prefer LLM's high-signal fields but fill gaps from rule-based.
    merged = Scenario(
        policy_type=(llm_scen.policy_type if llm_scen.policy_type != "unspecified"
                     else rb.policy_type),
        affected_states=llm_scen.affected_states or rb.affected_states,
        cleavages=llm_scen.cleavages or rb.cleavages,
        effective_week=(llm_scen.effective_week
                        if llm_scen.effective_week is not None else rb.effective_week),
        severity=llm_scen.severity,
        raw_text=text,
        source="llm",
    )
    return merged
