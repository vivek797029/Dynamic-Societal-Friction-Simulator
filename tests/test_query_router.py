"""Router smoke tests -- no torch, no network."""
from __future__ import annotations

import sys
import types

# Stub torch so importing the models.* modules pulled in by query.* is cheap.
# The router itself never touches torch, but scenario.py / analogues.py import
# from ..models.cleavage_classifier which does. This shim lets the tests run
# on a bare Python install.
if "torch" not in sys.modules:
    fake_torch = types.ModuleType("torch")
    fake_torch.nn = types.ModuleType("torch.nn")
    fake_torch.nn.functional = types.ModuleType("torch.nn.functional")
    fake_torch.nn.Module = object                                    # type: ignore[attr-defined]
    sys.modules["torch"] = fake_torch
    sys.modules["torch.nn"] = fake_torch.nn
    sys.modules["torch.nn.functional"] = fake_torch.nn.functional

from src.query.router import Intent, route_rule_based  # noqa: E402


def test_empty_prompt_is_off_domain():
    d = route_rule_based("")
    assert d.intent is Intent.OFF_DOMAIN
    assert not d.is_model_grounded


def test_what_if_india_policy_routes_what_if():
    d = route_rule_based("What if the CAA is extended to Kerala?")
    assert d.intent is Intent.WHAT_IF
    assert d.is_model_grounded
    assert d.confidence >= 0.65


def test_hypothetical_without_india_is_off_domain():
    d = route_rule_based("What if it rains tomorrow in Tokyo?")
    assert d.intent is Intent.OFF_DOMAIN
    assert not d.is_model_grounded


def test_explain_now_with_state_name():
    d = route_rule_based("Why is Punjab flagged high this week?")
    assert d.intent is Intent.EXPLAIN_NOW
    assert d.is_model_grounded


def test_domain_keyword_only_defaults_to_explain_now():
    d = route_rule_based("Karnataka communal tension")
    assert d.intent is Intent.EXPLAIN_NOW


def test_cricket_trivia_is_off_domain():
    d = route_rule_based("Who won the 2024 ICC T20 World Cup?")
    # Cricket isn't in the domain lexicon -> off domain.
    assert d.intent is Intent.OFF_DOMAIN


def test_cooking_is_off_domain():
    d = route_rule_based("How do I make dal makhani?")
    assert d.intent is Intent.OFF_DOMAIN


def test_simulate_verb_with_state_is_what_if():
    d = route_rule_based("Simulate a farm law in Maharashtra")
    assert d.intent is Intent.WHAT_IF


def test_route_decision_to_dict_shape():
    d = route_rule_based("Why is Bihar flagged?")
    out = d.to_dict()
    assert set(out.keys()) >= {"intent", "confidence", "reason",
                                 "is_model_grounded", "signals"}
    assert isinstance(out["signals"], dict)


if __name__ == "__main__":
    # Minimal self-runner so this file works without pytest.
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
