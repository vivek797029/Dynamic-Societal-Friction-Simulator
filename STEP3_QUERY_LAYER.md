# Step 3 — Query layer (`src/query/`)

This step adds a natural-language interface on top of the friction
pipeline so users can ask policy/society questions about India and get
back a numeric forecast, a state-level heatmap, a narrative answer, and
cited evidence. It also supports purely off-domain questions via a pure
LLM fallback with a clear "not model-grounded" flag.

## Scope (from the user's Step 3 choices)

- **Interfaces enabled**: all three
  - Policy-scenario simulator (what-if)
  - Q&A over the current pipeline state (explain-now)
  - General India Q&A / off-domain fallback
- **Input mode**: chat with optional form
- **Outputs bundled in every answer**: numeric forecast, state-level
  heatmap (SVG), natural-language narrative, cited evidence
- **Fallback behaviour**: pure LLM for off-domain prompts

## Module map

```
src/query/
├── __init__.py          # public surface: answer, AnswerBundle, ...
├── api.py               # top-level answer() orchestrator
├── cli.py               # `python -m src.query.cli "..."`
├── loader.py            # PipelineContext loader (numpy tensors + .pt)
├── llm.py               # LLMProvider protocol + Stub/Anthropic/OpenAI
├── router.py            # intent classifier (what_if / explain_now / off_domain)
├── scenario.py          # prompt -> Scenario dataclass (rule + LLM)
├── intervention.py      # Scenario -> perturbed tensors -> SimulationResult
├── analogues.py         # top-k historical (state, week) neighbours
├── narrative.py         # template + LLM generators for all three branches
└── viz.py               # SVG choropleth tilegram of the 36 states
```

## Data flow

```
prompt ──▶ route()               # rule-based, escalates to LLM if confidence<0.65
           │
   ┌───────┴──────────────────────┐
   │                              │
   ▼ off_domain                    ▼ what_if / explain_now
off_domain_answer()             extract_scenario()  +  find_analogues()
  (pure LLM, flagged)             │                     │
                                   ▼                     ▼
                                 simulate()          _build_explain_state()
                                   │                     │
                                   └─────────┬───────────┘
                                             ▼
                                   render_what_if() | render_explain_now()
                                   choropleth()
                                             │
                                             ▼
                                        AnswerBundle
```

## Key design choices

### Pluggable LLM behind a Protocol

`LLMProvider` has just two methods (`complete`, `complete_json`). Three
implementations ship — `StubLLM` (deterministic, no network), `AnthropicLLM`,
`OpenAILLM`. `get_llm()` reads the `DSFS_LLM` env var, falling back to stub
so the entire query layer runs without any API keys. Every call to an LLM
is wrapped in try/except and returns `""` or `None` on failure; callers
treat that as an automatic fall-through to the rule-based / template path.

### Router is conservative by default

Rule-based heuristics (state-name / policy-keyword / hypothetical-phrase
matches) decide the intent for anything with strong signal. For ambiguous
prompts we escalate to the LLM only if a non-stub provider is configured,
and only accept its override if it's more confident than the rule-based
hit. The default for domain-relevant but question-marker-free prompts is
**explain_now** rather than what_if — read-only summaries are safer than
fabricated forecasts.

### Perturbation as calibrated sensitivity, not causal inference

`apply_perturbation` adds a `severity * state_std` impulse with
exponential decay across `impulse_weeks` to both the event (E) and
media-tone (T) tensors for each (affected_state, cleavage, week) cell.
We rerun the aggregator + head (when available) and diff. This is
flagged to the user in every what-if narrative:

> "These numbers are calibrated sensitivity estimates, not ground-truth
> counterfactuals. Treat them as directional."

The paper-grade counterfactual evaluation still lives in `evaluation/`
(difference-in-differences on real historical interventions); the
simulator here is a fast interactive UX.

### Analogues via cosine similarity over cleavage space

`find_analogues` z-scores the summed E+T tensor per-state (so baseline-
noisy states don't dominate), compresses each (state, week) slice to a
K-dim cleavage profile, then takes cosine similarity against the
scenario signature. Optional `state_restrict`, `exclude_window_end` (to
avoid future leakage in historical studies), and `article_index` (to
attach source articles as evidence) parameters cover the common calls.

### Choropleth is a tilegram, not a geographic map

Shipping true vector geometry for 36 states would add >100kB of
GeoJSON + a shapely dependency. Instead `viz.py` renders each state as
a fixed-size tile on a hand-laid 8×10 grid that roughly matches India's
topology. The SVG is self-contained (no external CSS/JS) so it embeds
directly in Streamlit, HTML emails, or notebooks. A diverging
red-white-blue ramp with a grey "no data" fill and a legend come for
free.

### AnswerBundle is the contract

Every call to `answer()` returns the same dataclass, regardless of the
intent branch taken. Fields that don't apply to a given branch (e.g.
`scenario` for an explain-now answer) are left as `None` or empty
lists. Serializable via `.to_dict()` → JSON so UIs can consume it over
a thin HTTP wrapper.

## Tests (all stdlib-runnable with a torch stub)

```
tests/test_query_router.py      9 tests  — intent routing rules
tests/test_query_scenario.py    9 tests  — policy/state/cleavage/severity extraction
tests/test_query_analogues.py   6 tests  — shape + semantics of retrieval
tests/test_query_viz.py         7 tests  — SVG shape + color ramp
tests/test_query_api.py         6 tests  — end-to-end bundle shape per branch
```

Running in-sandbox (no pytest, no torch):

```
cd dynamic-societal-friction-simulator
PYTHONPATH=. python tests/test_query_router.py
PYTHONPATH=. python tests/test_query_scenario.py
PYTHONPATH=. python tests/test_query_analogues.py
PYTHONPATH=. python tests/test_query_viz.py
PYTHONPATH=. python tests/test_query_api.py
```

All 37 new tests pass.

## Usage

```python
from src.query import answer

# Dry-run mode: no trained model needed.
bundle = answer("What if the CAA is extended to Kerala?")
print(bundle.narrative)
print(bundle.intent)                        # "what_if"
print(bundle.heatmap_svg[:80])              # '<svg xmlns="...'

# With a real context + Anthropic LLM:
import os
os.environ["DSFS_LLM"] = "anthropic"
os.environ["ANTHROPIC_API_KEY"] = "..."
from src.query.loader import load_context
ctx = load_context("artifacts/stage_b/")
bundle = answer("Why is Manipur flagged high this week?", ctx=ctx)
```

CLI:

```
python -m src.query.cli "What if a new farm law passes?"
python -m src.query.cli --save-heatmap map.svg --json out.json \
    --llm anthropic "Effects of extending Article 370 reforms to Ladakh"
```

## What still belongs to Step 2's backlog (unchanged)

These items from §1.8 of the original audit weren't in the Step 3
scope and are still pending:

- Time-split (train/val/test) eval harness
- ACLED admin1 canonicalization
- FAISS clustering for SupCon positives
- Vectorized `build_T_tensor` and `relational_strain_tensor`
- Multi-label event cleavage classifier
- Monthly parquet partitioning
