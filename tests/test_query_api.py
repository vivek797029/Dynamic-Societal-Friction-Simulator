"""End-to-end answer() smoke tests with StubLLM + dry-run context.

These tests exercise the full query layer without torch or network calls.
They verify the shape of the returned AnswerBundle and check that each of
the three routing branches (what_if, explain_now, off_domain) produces
the expected fields populated.
"""
from __future__ import annotations

import sys
import types

if "torch" not in sys.modules:
    fake_torch = types.ModuleType("torch")
    fake_torch.nn = types.ModuleType("torch.nn")
    fake_torch.nn.functional = types.ModuleType("torch.nn.functional")
    fake_torch.nn.Module = object                                   # type: ignore[attr-defined]
    sys.modules["torch"] = fake_torch
    sys.modules["torch.nn"] = fake_torch.nn
    sys.modules["torch.nn.functional"] = fake_torch.nn.functional

from src.query.api import AnswerBundle, answer                        # noqa: E402
from src.query.intervention import PipelineContext                    # noqa: E402
from src.query.llm import StubLLM                                     # noqa: E402


def _ctx() -> PipelineContext:
    return PipelineContext.dry_run(S=36, T=60, seed=42)


def test_what_if_bundle_populated():
    b = answer("What if the CAA is extended to Kerala next month?",
               ctx=_ctx(), llm=StubLLM())
    assert isinstance(b, AnswerBundle)
    assert b.intent == "what_if"
    assert b.is_model_grounded
    assert b.scenario is not None
    assert b.scenario["policy_type"] == "caa_nrc"
    assert "Kerala" in b.scenario["affected_states"]
    assert b.narrative                                                # non-empty
    assert b.state_deltas                                             # list populated
    assert b.heatmap_svg and b.heatmap_svg.startswith("<svg")
    assert "dry-run" in " ".join(b.warnings).lower()                  # dry-run warning fired


def test_explain_now_branch():
    b = answer("Why is Punjab flagged high this week?",
               ctx=_ctx(), llm=StubLLM())
    assert b.intent == "explain_now"
    assert b.is_model_grounded
    assert b.scenario is None                                         # no scenario for explain_now
    assert b.narrative
    assert b.state_deltas                                             # top_states_now rows
    assert b.heatmap_svg and b.heatmap_svg.startswith("<svg")


def test_off_domain_branch_not_grounded():
    b = answer("How do I cook dal makhani?", ctx=_ctx(), llm=StubLLM())
    assert b.intent == "off_domain"
    assert not b.is_model_grounded
    # We prepend a "not grounded" note in the narrative.
    assert "not grounded" in b.narrative.lower() \
        or "not model-grounded" in b.narrative.lower() \
        or "outside" in b.narrative.lower()
    assert b.heatmap_svg is None
    assert b.state_deltas == []


def test_answer_handles_missing_ctx_and_llm():
    # Both should default to dry-run / stub.
    b = answer("What if UP sees a new reservation bill?")
    assert isinstance(b, AnswerBundle)
    assert b.intent in {"what_if", "explain_now"}


def test_render_heatmap_false_skips_svg():
    b = answer("What if Haryana passes a new farm law?",
               ctx=_ctx(), llm=StubLLM(), render_heatmap=False)
    assert b.heatmap_svg is None


def test_bundle_to_dict_serializable():
    import json
    b = answer("Simulate a farm law in Punjab", ctx=_ctx(), llm=StubLLM())
    # Must be round-trippable to JSON (no numpy / objects leaking through).
    blob = json.dumps(b.to_dict(), default=str)
    assert '"intent"' in blob
    assert '"narrative"' in blob


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok  {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    raise SystemExit(0 if failures == 0 else 1)
