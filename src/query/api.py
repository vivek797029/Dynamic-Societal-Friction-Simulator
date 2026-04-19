"""Top-level query API.

A single entry point -- `answer(prompt, ctx=None, llm=None, ...)` --
does the whole thing: route the intent, extract a scenario if needed,
run the simulator or the explain-now summary, pull analogues, generate
a narrative, and render a choropleth. The return is an `AnswerBundle`
that contains every artifact a UI layer might want (numeric forecast,
state heatmap as SVG, narrative text, cited evidence, off-domain flag,
the raw RouteDecision and Scenario for debugging).

No step of this function raises on missing inputs: when the caller
passes `ctx=None` we automatically build a `PipelineContext.dry_run()`
so the module is usable on a cold install with no trained artifacts.
When `llm=None` we use `StubLLM`, so there are no network calls by
default.

This keeps the "happy path" trivial:

    from src.query import answer
    bundle = answer("What if the CAA is extended to Kerala?")
    print(bundle.narrative)
    Path("map.svg").write_text(bundle.heatmap_svg or "")
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

from ..data.india_geo import STATES, STATE_TO_IDX
from ..models.cleavage_classifier import CLEAVAGES
from .analogues import Analogue, find_analogues
from .intervention import PipelineContext, SimulationResult, simulate
from .llm import LLMProvider, StubLLM, get_llm
from .narrative import (ExplainState, off_domain_answer,
                         render_explain_now, render_what_if)
from .router import Intent, RouteDecision, route
from .scenario import Scenario, extract_scenario
from .viz import choropleth


@dataclass
class AnswerBundle:
    """Everything a UI might want from a single user query."""
    prompt: str
    intent: str
    is_model_grounded: bool                      # False => off-domain
    narrative: str                               # main text answer
    scenario: dict | None = None                 # scenario.to_dict() or None
    route: dict | None = None                    # RouteDecision.to_dict()
    state_deltas: list[dict] = field(default_factory=list)  # [{state, h1, h2, h4}]
    horizons: list[int] = field(default_factory=list)
    top_cleavages: list[dict] = field(default_factory=list)  # per focus state
    analogues: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)      # {title, source, iso_week, state, url?}
    heatmap_svg: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------- evidence / citations ------------------------ #

def _collect_citations(analogues: list[Analogue],
                       limit: int = 8) -> list[dict]:
    """Flatten the top articles attached to analogues into a dedup'd list."""
    seen = set()
    out: list[dict] = []
    for a in analogues:
        for art in a.articles:
            key = (art.get("title"), art.get("source"), a.iso_week, a.state)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "title": art.get("title"),
                "source": art.get("source"),
                "state": a.state,
                "iso_week": int(a.iso_week),
                "url": art.get("url"),
                "article_id": art.get("article_id"),
            })
            if len(out) >= limit:
                return out
    return out


# ---------------------------- explain_now ------------------------------ #

def _build_explain_state(ctx: PipelineContext,
                          focus_state: str | None,
                          analogues: list[Analogue],
                          window_weeks: int = 8,
                          ) -> ExplainState:
    """Summarize the current baseline state of the pipeline."""
    w = min(window_weeks, ctx.T_len)
    # Per-state friction score: recent E+T mean over the last window.
    score = (ctx.E[:, -w:, :].mean(axis=(1, 2))
             + ctx.T[:, -w:, :].mean(axis=(1, 2)))
    order = np.argsort(-score)
    top_states = [(STATES[i], float(score[i])) for i in order[:10]]

    # Per-cleavage contribution either national or for focus_state.
    if focus_state and focus_state in STATE_TO_IDX:
        i = STATE_TO_IDX[focus_state]
        per_k = (ctx.E[i, -w:, :].mean(axis=0)
                 + ctx.T[i, -w:, :].mean(axis=0))
    else:
        per_k = (ctx.E[:, -w:, :].mean(axis=(0, 1))
                 + ctx.T[:, -w:, :].mean(axis=(0, 1)))
    k_order = np.argsort(-per_k)
    top_cleavages = [(CLEAVAGES[j], float(per_k[j])) for j in k_order[:6]]

    # Recent events come from the analogues' article payloads.
    recent: list[dict] = []
    for a in analogues[:3]:
        for art in a.articles[:2]:
            recent.append({
                "title": art.get("title"),
                "source": art.get("source"),
                "iso_week": int(a.iso_week),
                "state": a.state,
            })

    return ExplainState(
        focus_state=focus_state,
        top_states_now=top_states,
        top_cleavages=top_cleavages,
        analogues=analogues,
        recent_events=recent,
        horizon_weeks=1,
    )


def _pick_focus_state(prompt: str) -> str | None:
    """Case-insensitive exact-name match against STATES. Returns the first."""
    low = prompt.lower()
    for s in STATES:
        # Use word-ish boundaries so 'Goa' doesn't match 'goals'.
        padded = f" {low} "
        if f" {s.lower()} " in padded or low.startswith(s.lower() + " ") \
                or low.endswith(" " + s.lower()):
            return s
    return None


# ------------------------- what-if orchestration ----------------------- #

def _state_deltas_table(sim: SimulationResult,
                        horizons: tuple[int, ...],
                        target_idx: int = 0) -> list[dict]:
    """[{'state':..., 'h1': x, 'h2': y, 'h4': z}, ...] sorted by h1 desc."""
    rows: list[tuple[str, list[float]]] = []
    H = sim.delta.shape[1]
    for i, s in enumerate(STATES):
        vals = [float(sim.delta[i, h, target_idx]) for h in range(H)]
        rows.append((s, vals))
    rows.sort(key=lambda kv: -kv[1][0])
    out: list[dict] = []
    for s, vals in rows:
        row = {"state": s}
        for h, w in zip(range(H), horizons):
            row[f"h{w}"] = vals[h]
        out.append(row)
    return out


# ------------------------------- entry --------------------------------- #

def answer(prompt: str,
           ctx: PipelineContext | None = None,
           llm: LLMProvider | None = None,
           *,
           k_analogues: int = 5,
           article_index: dict | None = None,
           render_heatmap: bool = True,
           ) -> AnswerBundle:
    """Single entry point used by the CLI and the UI.

    Parameters
    ----------
    prompt          : user's natural-language question.
    ctx             : pipeline context; built as a dry-run if None.
    llm             : LLMProvider; defaults to env-configured `get_llm()`.
    k_analogues     : top-k historical analogues to retrieve.
    article_index   : optional {(state_idx, week_rel) -> [article dicts]}.
                      When present, analogues + citations are populated.
    render_heatmap  : if False, skip SVG rendering (faster for chat UIs).
    """
    ctx = ctx or PipelineContext.dry_run()
    llm = llm or get_llm()

    decision = route(prompt, llm=llm)

    # ---- off-domain: pure LLM, no ctx touches. ----
    if decision.intent is Intent.OFF_DOMAIN:
        text = off_domain_answer(prompt, llm=llm)
        return AnswerBundle(
            prompt=prompt,
            intent=decision.intent.value,
            is_model_grounded=False,
            narrative=("[Not grounded in the friction model]\n\n" + text),
            route=decision.to_dict(),
            warnings=["off_domain: answered from general LLM, no model grounding"],
        )

    warnings: list[str] = []
    if ctx.aggregator is None or ctx.head is None:
        warnings.append("dry-run mode: numeric forecasts are placeholder "
                         "values, not a trained model.")

    # ---- what-if: full simulation. ----
    if decision.intent is Intent.WHAT_IF:
        scenario = extract_scenario(prompt, llm=llm)
        issues = scenario.validate()
        if issues:
            warnings.extend(issues)
        sim = simulate(ctx, scenario)
        analogues = find_analogues(
            ctx, scenario, k=k_analogues,
            state_restrict=scenario.affected_states or None,
            article_index=article_index,
        )
        narrative = render_what_if(scenario, sim, analogues, llm=llm,
                                    horizons=ctx.horizons)
        heatmap = None
        if render_heatmap:
            # Color the +1 week protest delta by state.
            state_vals = {
                STATES[i]: float(sim.delta[i, 0, 0]) for i in range(ctx.S)
            }
            heatmap = choropleth(
                state_vals,
                title=f"Projected protest-rate change at +{ctx.horizons[0]} week",
                subtitle=f"Scenario: {scenario.policy_type} (severity={scenario.severity:.2f})",
            )
        top_cleavages_bundle: list[dict] = []
        if scenario.affected_states:
            for s in scenario.affected_states[:3]:
                top_cleavages_bundle.append({
                    "state": s,
                    "cleavages": [{"name": c, "weight": v}
                                  for c, v in sim.top_cleavages_per_state(s, k=5)],
                })
        return AnswerBundle(
            prompt=prompt,
            intent=decision.intent.value,
            is_model_grounded=True,
            narrative=narrative,
            scenario=scenario.to_dict(),
            route=decision.to_dict(),
            state_deltas=_state_deltas_table(sim, ctx.horizons),
            horizons=list(ctx.horizons),
            top_cleavages=top_cleavages_bundle,
            analogues=[a.to_dict() for a in analogues],
            citations=_collect_citations(analogues),
            heatmap_svg=heatmap,
            warnings=warnings,
        )

    # ---- explain_now: baseline read-only summary. ----
    assert decision.intent is Intent.EXPLAIN_NOW
    focus = _pick_focus_state(prompt)
    # Build a "signature-free" scenario so analogues compare to recent state.
    recent_sig = Scenario(
        policy_type="explain_now",
        affected_states=[focus] if focus else [],
        cleavages=[],
        effective_week=None,
        severity=0.5,
        raw_text=prompt,
        source="explain_now_stub",
    )
    analogues = find_analogues(
        ctx, recent_sig, k=k_analogues,
        state_restrict=[focus] if focus else None,
        article_index=article_index,
    )
    explain = _build_explain_state(ctx, focus, analogues)
    narrative = render_explain_now(prompt, explain, llm=llm)
    heatmap = None
    if render_heatmap:
        # Tile the national friction score by state.
        vals = {s: v for s, v in explain.top_states_now}
        heatmap = choropleth(
            vals,
            title="Current friction intensity by state",
            subtitle=("Focus: " + focus) if focus else "National view",
        )
    return AnswerBundle(
        prompt=prompt,
        intent=decision.intent.value,
        is_model_grounded=True,
        narrative=narrative,
        scenario=None,
        route=decision.to_dict(),
        state_deltas=[{"state": s, "score": v} for s, v in explain.top_states_now],
        horizons=[explain.horizon_weeks],
        top_cleavages=[{"state": focus or "(national)",
                         "cleavages": [{"name": c, "weight": v}
                                       for c, v in explain.top_cleavages]}],
        analogues=[a.to_dict() for a in analogues],
        citations=_collect_citations(analogues),
        heatmap_svg=heatmap,
        warnings=warnings,
    )
