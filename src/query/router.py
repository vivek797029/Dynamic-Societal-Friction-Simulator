"""Intent router for the query layer.

A user prompt can ask three qualitatively different things:

    what_if       -- a hypothetical ("what if the CAA is extended to Kerala?").
                     The policy-scenario simulator runs end-to-end: extract
                     Scenario, perturb tensors, forecast deltas, pull
                     analogues, render narrative + heatmap.

    explain_now   -- grounded questions about the pipeline's current state
                     ("why is Punjab flagged high this week?", "which
                     cleavages drove the Manipur uptick?"). No perturbation;
                     we interpret the baseline forecast + evidence.

    off_domain    -- questions outside what this system can ground ("who
                     won the 2024 cricket world cup?", "how do I make dal
                     makhani?"). We answer from the LLM with an explicit
                     "not model-grounded" flag in the response so the UI
                     can show a disclaimer.

The router is deliberately conservative: it uses fast rule-based
heuristics first, and only consults the LLM when the rules are ambiguous.
Mis-routing a what-if question into off-domain loses UX; mis-routing an
off-domain question into what-if fabricates a forecast. So the default
for ambiguous-but-plausibly-Indian-society prompts is `explain_now`, not
`what_if`, since `explain_now` is read-only over the existing state.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..data.india_geo import STATES
from ..models.cleavage_classifier import CLEAVAGES
from .llm import LLMProvider, StubLLM


class Intent(str, Enum):
    WHAT_IF = "what_if"
    EXPLAIN_NOW = "explain_now"
    OFF_DOMAIN = "off_domain"


@dataclass
class RouteDecision:
    intent: Intent
    confidence: float                            # rough [0,1]
    reason: str                                  # human-readable rationale
    is_model_grounded: bool                      # False => off_domain LLM fallback
    signals: dict = field(default_factory=dict)  # which rules fired

    def to_dict(self) -> dict:
        return {
            "intent": self.intent.value,
            "confidence": float(self.confidence),
            "reason": self.reason,
            "is_model_grounded": bool(self.is_model_grounded),
            "signals": dict(self.signals),
        }


# --------------------------- lexical features --------------------------- #

# Phrases that strongly suggest a counterfactual / hypothetical ask.
_WHAT_IF_PHRASES: list[re.Pattern] = [
    re.compile(r"\bwhat(?:\s+would|\s+if|\'s)\b", re.I),
    re.compile(r"\bif\s+(?:the\s+)?(?:government|centre|state|india|bjp|congress)\b", re.I),
    re.compile(r"\bsuppose\b", re.I),
    re.compile(r"\bimagine\s+(?:that|if)\b", re.I),
    re.compile(r"\bhypothetical(?:ly)?\b", re.I),
    re.compile(r"\b(?:impact|effect|consequences?)\s+of\s+(?:passing|banning|extending|imposing|repealing|announcing)\b", re.I),
    re.compile(r"\b(?:passing|banning|extending|imposing|repealing)\s+(?:the\s+)?\w+\s+(?:bill|law|act|order)\b", re.I),
    re.compile(r"\bsimulate\b", re.I),
    re.compile(r"\bcounter-?factual\b", re.I),
    re.compile(r"\bpredict\s+.*\bif\b", re.I),
    re.compile(r"\bwould\s+happen\b", re.I),
]

# Phrases suggesting the user wants to understand the *current* state.
_EXPLAIN_NOW_PHRASES: list[re.Pattern] = [
    re.compile(r"\bwhy\s+is\b", re.I),
    re.compile(r"\bwhy\s+are\b", re.I),
    re.compile(r"\bwhat(?:\'s|\s+is)\s+(?:happening|driving|going on)\b", re.I),
    re.compile(r"\bcurrent(?:ly)?\s+(?:status|situation|outlook)\b", re.I),
    re.compile(r"\bthis\s+(?:week|month|quarter)\b", re.I),
    re.compile(r"\b(?:right\s+now|at\s+the\s+moment|these\s+days)\b", re.I),
    re.compile(r"\b(?:explain|describe|summari[sz]e)\b", re.I),
    re.compile(r"\bhow\s+(?:bad|serious|tense)\b", re.I),
    re.compile(r"\bflagg?ed\s+high\b", re.I),
]

# Topic gate: if none of these appear AND no state/cleavage keyword fires,
# we lean off-domain. This keeps "how do I cook biryani" or "who won IPL"
# out of the model-grounded path.
_DOMAIN_KEYWORDS: list[str] = [
    "india", "indian", "bharat", "delhi", "mumbai", "bengaluru", "kolkata",
    "protest", "protests", "riot", "riots", "unrest", "violence", "conflict",
    "communal", "caste", "reservation", "hindutva", "secular",
    "bjp", "congress", "aap", "tmc", "dmk", "shiv sena", "nda", "india alliance",
    "parliament", "lok sabha", "rajya sabha", "modi", "rahul gandhi", "kejriwal",
    "farmer", "farm law", "msp", "kisan", "caa", "nrc", "article 370",
    "hijab", "triple talaq", "ayodhya", "babri", "manipur", "kashmir",
    "dalit", "adivasi", "obc", "scheduled tribe", "scheduled caste",
    "friction", "tension", "escalation", "forecast", "prediction",
    "cleavage", "society", "societal",
]
_DOMAIN_RE = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(w) for w in _DOMAIN_KEYWORDS) + r")(?!\w)",
    re.IGNORECASE,
)

# State / cleavage keyword detection mirrors scenario.py but is used as a
# domain signal here rather than for extraction.
_STATE_RE = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(s) for s in STATES) + r")(?!\w)",
    re.IGNORECASE,
)


def _count_matches(patterns: list[re.Pattern], text: str) -> int:
    return sum(1 for p in patterns if p.search(text))


def _has_domain_signal(text: str) -> bool:
    if _DOMAIN_RE.search(text):
        return True
    if _STATE_RE.search(text):
        return True
    for c in CLEAVAGES:
        if re.search(r"(?<!\w)" + re.escape(c) + r"(?!\w)", text, re.I):
            return True
    return False


# --------------------------- rule-based router -------------------------- #

def route_rule_based(text: str) -> RouteDecision:
    """Fast path. Returns a RouteDecision with confidence in [0, 1].

    Confidence is low (<0.6) when rules disagree or signal is weak --
    callers can consult the LLM to arbitrate in that case.
    """
    if not text or not text.strip():
        return RouteDecision(
            intent=Intent.OFF_DOMAIN, confidence=1.0,
            reason="empty prompt", is_model_grounded=False,
            signals={"empty": True},
        )

    what_if_hits = _count_matches(_WHAT_IF_PHRASES, text)
    explain_hits = _count_matches(_EXPLAIN_NOW_PHRASES, text)
    domain = _has_domain_signal(text)

    signals = {
        "what_if_hits": what_if_hits,
        "explain_hits": explain_hits,
        "domain_keyword": domain,
    }

    # No India-domain signal at all -> off domain.
    if not domain and what_if_hits == 0 and explain_hits == 0:
        return RouteDecision(
            intent=Intent.OFF_DOMAIN, confidence=0.85,
            reason="no India/policy/cleavage keywords detected",
            is_model_grounded=False, signals=signals,
        )

    # Strong what-if signal and domain-relevant -> WHAT_IF.
    if what_if_hits >= 1 and domain:
        conf = 0.8 if what_if_hits >= 2 else 0.7
        return RouteDecision(
            intent=Intent.WHAT_IF, confidence=conf,
            reason=f"what-if phrase(s) hit ({what_if_hits}) + domain keyword",
            is_model_grounded=True, signals=signals,
        )

    # What-if phrase but no domain signal -> off domain hypothetical.
    # ("what if I eat too much sugar" isn't a friction forecast.)
    if what_if_hits >= 1 and not domain:
        return RouteDecision(
            intent=Intent.OFF_DOMAIN, confidence=0.7,
            reason="hypothetical but no India/policy/cleavage keywords",
            is_model_grounded=False, signals=signals,
        )

    # Explain-now phrase with domain -> EXPLAIN_NOW.
    if explain_hits >= 1 and domain:
        conf = 0.8 if explain_hits >= 2 else 0.7
        return RouteDecision(
            intent=Intent.EXPLAIN_NOW, confidence=conf,
            reason=f"explain-now phrase(s) hit ({explain_hits}) + domain keyword",
            is_model_grounded=True, signals=signals,
        )

    # Domain keyword only, no question markers -- default to EXPLAIN_NOW
    # (safer than fabricating a hypothetical).
    if domain:
        return RouteDecision(
            intent=Intent.EXPLAIN_NOW, confidence=0.55,
            reason="domain keyword(s) only; defaulting to explain-now",
            is_model_grounded=True, signals=signals,
        )

    # Shouldn't reach here given the earlier clauses, but be defensive.
    return RouteDecision(
        intent=Intent.OFF_DOMAIN, confidence=0.5,
        reason="no strong signal either way",
        is_model_grounded=False, signals=signals,
    )


# ------------------------------- LLM path ------------------------------- #

_ROUTER_SCHEMA = """{
  "intent": "what_if" | "explain_now" | "off_domain",
  "confidence": 0.0 to 1.0,
  "reason": "short one-sentence rationale"
}"""

_ROUTER_SYSTEM = (
    "You are routing a user prompt to one of three handlers in a societal-"
    "friction forecaster for India. Choose exactly one intent:\n"
    "- 'what_if' : the user asks a hypothetical about a policy/event and "
    "wants a prediction of its effect.\n"
    "- 'explain_now' : the user asks about the current/recent state of "
    "friction in India (why is a state flagged, which cleavages are hot).\n"
    "- 'off_domain' : the user asks something this model can't ground: not "
    "about India, not about societal friction, or purely conversational.\n"
    "Respond with ONLY the JSON object. Never answer the user's question."
)


def route_llm(text: str, llm: LLMProvider) -> RouteDecision | None:
    raw = llm.complete_json(text, schema_hint=_ROUTER_SCHEMA, system=_ROUTER_SYSTEM)
    if not raw:
        return None
    try:
        intent_str = str(raw.get("intent") or "").lower().strip()
        if intent_str not in {"what_if", "explain_now", "off_domain"}:
            return None
        conf = float(raw.get("confidence") or 0.6)
        conf = max(0.0, min(1.0, conf))
        reason = str(raw.get("reason") or "(llm)")
        intent = Intent(intent_str)
        return RouteDecision(
            intent=intent, confidence=conf, reason=reason,
            is_model_grounded=(intent != Intent.OFF_DOMAIN),
            signals={"source": "llm"},
        )
    except (TypeError, ValueError):
        return None


# --------------------------- composed router --------------------------- #

# When the rule-based router's confidence is at or above this threshold,
# we trust it outright and don't pay the LLM round-trip.
_TRUST_THRESHOLD = 0.65


def route(text: str, llm: LLMProvider | None = None) -> RouteDecision:
    """Rule-based first; escalate to the LLM only for low-confidence cases.

    Never raises -- LLM failures silently fall back to the rule-based
    decision. This keeps the query layer usable with no API keys.
    """
    rb = route_rule_based(text)
    if rb.confidence >= _TRUST_THRESHOLD:
        return rb
    if llm is None or isinstance(llm, StubLLM):
        return rb
    llm_decision = route_llm(text, llm)
    if llm_decision is None:
        return rb
    # Blend the two if they agree -> higher confidence.
    if llm_decision.intent == rb.intent:
        llm_decision = RouteDecision(
            intent=llm_decision.intent,
            confidence=min(1.0, max(rb.confidence, llm_decision.confidence) + 0.1),
            reason=f"rule-based + LLM agree: {llm_decision.reason}",
            is_model_grounded=llm_decision.is_model_grounded,
            signals={**rb.signals, **llm_decision.signals, "agree": True},
        )
    else:
        # Disagreement: prefer whichever has higher confidence.
        if llm_decision.confidence > rb.confidence:
            llm_decision = RouteDecision(
                intent=llm_decision.intent,
                confidence=llm_decision.confidence,
                reason=f"LLM overrode rules ({rb.intent.value} -> {llm_decision.intent.value}): {llm_decision.reason}",
                is_model_grounded=llm_decision.is_model_grounded,
                signals={**rb.signals, **llm_decision.signals, "agree": False},
            )
        else:
            return rb
    return llm_decision
