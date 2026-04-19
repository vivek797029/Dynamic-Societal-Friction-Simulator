"""Natural-language answer composer.

Given the structured pieces produced by the query pipeline -- a
`SimulationResult` (or just a baseline forecast), a list of `Analogue`s,
and a small set of cited articles -- produce a short, faithful
natural-language answer for the user.

Two modes, picked automatically:

    * Template mode (no LLM or StubLLM): deterministic, short paragraphs
      composed from the structured data. Good enough for tests and the
      "no API key" deployment.

    * LLM mode: we build a compact, grounded prompt that includes ONLY
      the structured facts (states, deltas, cleavages, analogues, short
      article titles + source + iso_week). The LLM is told explicitly
      not to invent numbers. This is generation-from-structured-data,
      not free-form chat -- the goal is fluency, not new facts.

Off-domain answers live in `off_domain_answer()` and go through the LLM
directly with a "not model-grounded" disclaimer baked into the prompt
and returned as a flag in the AnswerBundle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..models.cleavage_classifier import CLEAVAGES
from .analogues import Analogue
from .intervention import SimulationResult
from .llm import LLMProvider, StubLLM
from .scenario import Scenario


# --------------------------- template helpers -------------------------- #

def _fmt_pct_delta(x: float) -> str:
    """Render a log-rate delta as a rough % change in the underlying rate."""
    # log-delta -> multiplicative factor - 1; show +/- and one decimal.
    try:
        pct = 100.0 * (pow(2.718281828, float(x)) - 1.0)
    except OverflowError:
        pct = 0.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def _top_n_state_deltas(sim: SimulationResult, n: int = 5,
                        horizon_idx: int = 0, target_idx: int = 0
                        ) -> list[tuple[str, float]]:
    return sim.state_delta_ranked(horizon_idx, target_idx)[:n]


def _bullet_analogue(a: Analogue) -> str:
    tail = ""
    if a.articles:
        titles = ", ".join(
            f"{art.get('title') or art.get('source') or 'article'}"
            for art in a.articles[:2]
        )
        tail = f" (e.g. {titles})"
    return f"{a.state} in week {a.iso_week} (similarity {a.similarity:.2f}){tail}"


# --------------------------- what-if narrative ------------------------- #

def render_what_if_template(scenario: Scenario,
                             sim: SimulationResult,
                             analogues: list[Analogue],
                             horizons: tuple[int, ...] = (1, 2, 4),
                             ) -> str:
    """Deterministic what-if narrative, no LLM."""
    states = scenario.affected_states or ["(nationwide)"]
    cleavs = scenario.cleavages or ["(all cleavages)"]
    lines: list[str] = []
    lines.append(
        f"Scenario: {scenario.policy_type} at severity {scenario.severity:.2f}, "
        f"effective around iso-week {sim.effective_week}, "
        f"affecting {', '.join(states)} along {', '.join(cleavs)} lines."
    )

    # Top states per shortest horizon for protests.
    top = _top_n_state_deltas(sim, n=5, horizon_idx=0, target_idx=0)
    if top:
        human = ", ".join(f"{s} ({_fmt_pct_delta(d)})" for s, d in top)
        lines.append(
            f"Projected change in protest rate vs baseline at +{horizons[0]} "
            f"week(s): top movers are {human}."
        )

    # Mid/long horizon quick summary.
    if len(horizons) >= 2:
        mid = _top_n_state_deltas(sim, n=3, horizon_idx=1, target_idx=0)
        if mid:
            human = ", ".join(f"{s} ({_fmt_pct_delta(d)})" for s, d in mid)
            lines.append(f"At +{horizons[1]} week(s): {human}.")
    if len(horizons) >= 3:
        lg = _top_n_state_deltas(sim, n=3, horizon_idx=2, target_idx=0)
        if lg:
            human = ", ".join(f"{s} ({_fmt_pct_delta(d)})" for s, d in lg)
            lines.append(f"At +{horizons[2]} week(s): {human}.")

    # Cleavage breakdown for the top state.
    if top:
        top_state = top[0][0]
        per_k = sim.top_cleavages_per_state(top_state, k=3)
        if per_k:
            human = ", ".join(f"{c} ({v:+.2f})" for c, v in per_k)
            lines.append(
                f"In {top_state}, the largest friction shifts are along: {human}."
            )

    # Analogues.
    if analogues:
        bullet = "; ".join(_bullet_analogue(a) for a in analogues[:3])
        lines.append(f"Closest historical analogues: {bullet}.")

    lines.append(
        "These numbers are calibrated sensitivity estimates, not ground-truth "
        "counterfactuals. Treat them as directional."
    )
    return "\n\n".join(lines)


# ------------------------- explain-now narrative ----------------------- #

@dataclass
class ExplainState:
    """Lightweight bundle for explain_now. Populated by api.answer()."""
    focus_state: str | None = None
    top_states_now: list[tuple[str, float]] = field(default_factory=list)   # [(state, score)]
    top_cleavages: list[tuple[str, float]] = field(default_factory=list)    # [(cleavage, weight)]
    analogues: list[Analogue] = field(default_factory=list)
    recent_events: list[dict] = field(default_factory=list)                 # [{title, source, iso_week, state}]
    horizon_weeks: int = 1


def render_explain_now_template(explain: ExplainState) -> str:
    lines: list[str] = []
    if explain.focus_state:
        lines.append(f"Current outlook for {explain.focus_state}:")
    else:
        lines.append("Current national friction outlook:")

    if explain.top_states_now:
        human = ", ".join(f"{s} ({v:+.2f})" for s, v in explain.top_states_now[:5])
        lines.append(f"States with the highest modeled tension score: {human}.")

    if explain.top_cleavages:
        human = ", ".join(f"{c} ({v:+.2f})" for c, v in explain.top_cleavages[:5])
        lines.append(f"Dominant cleavage contributions: {human}.")

    if explain.analogues:
        bullets = "; ".join(_bullet_analogue(a) for a in explain.analogues[:3])
        lines.append(f"Comparable historical windows: {bullets}.")

    if explain.recent_events:
        ev = "; ".join(
            f"{e.get('title') or '(untitled)'} [{e.get('source') or '?'}]"
            for e in explain.recent_events[:3]
        )
        lines.append(f"Recent events feeding the signal: {ev}.")

    if len(lines) == 1:
        lines.append(
            "No strong state-level anomalies were found relative to recent "
            "baselines."
        )
    return "\n\n".join(lines)


# ------------------------------- LLM mode ------------------------------ #

_WHAT_IF_SYSTEM = (
    "You are a careful analyst for an Indian societal-friction forecaster. "
    "A simulation has produced the structured facts below. Your job is to "
    "write a crisp answer in 3-5 short paragraphs. Rules:\n"
    "- Use ONLY the numbers provided. Never invent states, rates, or events.\n"
    "- Keep the tone factual and neutral. Avoid speculation.\n"
    "- Call out the largest-magnitude per-state changes and the dominant "
    "cleavages.\n"
    "- When mentioning historical analogues, cite state + iso-week + any "
    "listed article titles.\n"
    "- End with a one-line caveat that these are calibrated sensitivities, "
    "not ground-truth counterfactuals."
)

_EXPLAIN_NOW_SYSTEM = (
    "You are a careful analyst for an Indian societal-friction forecaster. "
    "Answer the user's question using ONLY the structured facts below. "
    "Do not invent states, events, or numbers. Keep it to 2-4 short "
    "paragraphs. Cite article titles verbatim when referencing them."
)

_OFF_DOMAIN_SYSTEM = (
    "You are Claude, a general-purpose assistant. The user's question is "
    "NOT grounded in the societal-friction forecaster, so answer it from "
    "general knowledge. Be concise (under 180 words). If the question is "
    "about India, say so but do NOT pretend it was answered by the model. "
    "Do not fabricate statistics."
)


def _facts_block_whatif(scenario: Scenario, sim: SimulationResult,
                        analogues: list[Analogue]) -> str:
    top = _top_n_state_deltas(sim, n=10, horizon_idx=0, target_idx=0)
    top_lines = "\n".join(f"  - {s}: {_fmt_pct_delta(d)}" for s, d in top)
    ana_lines = "\n".join(
        f"  - {a.state} iso_week={a.iso_week} sim={a.similarity:.2f} "
        f"titles={[x.get('title') for x in (a.articles or [])][:2]}"
        for a in analogues[:5]
    ) or "  (none)"
    top_state = top[0][0] if top else None
    cleavage_line = ""
    if top_state:
        per_k = sim.top_cleavages_per_state(top_state, k=3)
        cleavage_line = f"Top cleavages in {top_state}: " + ", ".join(
            f"{c}={v:+.2f}" for c, v in per_k
        )
    return (
        f"Scenario: policy_type={scenario.policy_type} "
        f"severity={scenario.severity:.2f} "
        f"affected_states={scenario.affected_states or '(nationwide)'} "
        f"cleavages={scenario.cleavages or '(all)'}\n"
        f"Effective iso_week: {sim.effective_week}\n"
        f"Top state deltas at +1 week (log-rate):\n{top_lines}\n"
        f"{cleavage_line}\n"
        f"Analogues:\n{ana_lines}"
    )


def _facts_block_explain(explain: ExplainState) -> str:
    top = "\n".join(f"  - {s}: {v:+.2f}" for s, v in explain.top_states_now[:10])
    cl = ", ".join(f"{c}={v:+.2f}" for c, v in explain.top_cleavages[:6])
    ana = "\n".join(
        f"  - {a.state} iso_week={a.iso_week} sim={a.similarity:.2f}"
        for a in explain.analogues[:3]
    ) or "  (none)"
    ev = "\n".join(
        f"  - [{e.get('source') or '?'}] {e.get('title') or '(untitled)'}"
        for e in explain.recent_events[:5]
    ) or "  (none)"
    focus = explain.focus_state or "(national)"
    return (
        f"Focus: {focus}\n"
        f"Horizon weeks: {explain.horizon_weeks}\n"
        f"Top states now:\n{top}\n"
        f"Top cleavages: {cl}\n"
        f"Analogues:\n{ana}\n"
        f"Recent articles:\n{ev}"
    )


def render_what_if(scenario: Scenario, sim: SimulationResult,
                   analogues: list[Analogue], llm: LLMProvider | None = None,
                   horizons: tuple[int, ...] = (1, 2, 4)) -> str:
    """Pick LLM or template based on provider."""
    llm = llm or StubLLM()
    template = render_what_if_template(scenario, sim, analogues, horizons)
    if isinstance(llm, StubLLM):
        return template
    facts = _facts_block_whatif(scenario, sim, analogues)
    user = (
        f"User question: {scenario.raw_text or '(no prompt text)'}\n\n"
        f"Structured facts:\n{facts}"
    )
    out = llm.complete(user, system=_WHAT_IF_SYSTEM, max_tokens=700, temperature=0.2)
    return out.strip() or template


def render_explain_now(user_prompt: str, explain: ExplainState,
                       llm: LLMProvider | None = None) -> str:
    llm = llm or StubLLM()
    template = render_explain_now_template(explain)
    if isinstance(llm, StubLLM):
        return template
    facts = _facts_block_explain(explain)
    user = f"User question: {user_prompt}\n\nStructured facts:\n{facts}"
    out = llm.complete(user, system=_EXPLAIN_NOW_SYSTEM, max_tokens=500, temperature=0.2)
    return out.strip() or template


def off_domain_answer(user_prompt: str, llm: LLMProvider | None = None) -> str:
    """Pure-LLM answer for off-domain queries. Not model-grounded.

    If the LLM is stubbed or fails, we return a clear message explaining
    that this question falls outside the forecaster's grounding, which
    is more honest than pretending.
    """
    llm = llm or StubLLM()
    if isinstance(llm, StubLLM):
        return (
            "This question is outside the societal-friction model's grounding, "
            "and no LLM provider is configured to answer it. Set DSFS_LLM to "
            "'anthropic' or 'openai' (with the matching API key) for general "
            "answers."
        )
    out = llm.complete(user_prompt, system=_OFF_DOMAIN_SYSTEM,
                        max_tokens=400, temperature=0.3)
    fallback = (
        "I couldn't ground this question in the Indian friction model, and "
        "the LLM fallback returned nothing. Try rephrasing, or ask about "
        "an Indian state, policy, or cleavage."
    )
    return out.strip() or fallback
