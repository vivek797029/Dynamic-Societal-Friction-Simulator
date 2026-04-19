"""Rule-based scenario extraction tests."""
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

from src.query.scenario import (Scenario, extract_rule_based,          # noqa: E402
                                  extract_scenario)


def test_policy_type_detected():
    s = extract_rule_based("Effects of a new farm law on rural Punjab")
    assert s.policy_type == "farm_law"


def test_states_detected_by_name():
    s = extract_rule_based("What if Karnataka bans hijab in schools?")
    assert "Karnataka" in s.affected_states


def test_state_aliases():
    s = extract_rule_based("Signal from UP and MP protests")
    assert "Uttar Pradesh" in s.affected_states
    assert "Madhya Pradesh" in s.affected_states


def test_severity_hints():
    s = extract_rule_based("A nationwide emergency measure")
    assert s.severity >= 0.85


def test_default_severity():
    s = extract_rule_based("Some modest local change")
    # 'local' or 'minor' => 0.25; otherwise default 0.5
    assert 0.2 <= s.severity <= 0.6


def test_cleavage_keyword():
    s = extract_rule_based("Communal tension over a temple-mosque dispute")
    assert "communal" in s.cleavages


def test_validate_catches_unknown_state():
    s = Scenario(policy_type="x", affected_states=["Atlantis"])
    issues = s.validate()
    assert any("unknown state" in i for i in issues)


def test_validate_catches_bad_severity():
    s = Scenario(policy_type="x", severity=2.0)
    issues = s.validate()
    assert any("severity" in i for i in issues)


def test_extract_scenario_with_stub_llm_falls_back_to_rules():
    # No DSFS_LLM set -> StubLLM -> extract_scenario returns the rule-based result.
    s = extract_scenario("What if the farm law returns in Punjab?")
    assert s.policy_type == "farm_law"
    assert "Punjab" in s.affected_states


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
