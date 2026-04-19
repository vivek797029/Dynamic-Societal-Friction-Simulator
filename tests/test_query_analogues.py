"""Analogue retrieval tests against the dry-run PipelineContext."""
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

import numpy as np                                                   # noqa: E402

from src.query.analogues import (Analogue, find_analogues,             # noqa: E402
                                   scenario_signature)
from src.query.intervention import PipelineContext                     # noqa: E402
from src.query.scenario import Scenario                                # noqa: E402


def test_scenario_signature_shape_and_base_rate():
    sig = scenario_signature(Scenario(cleavages=["communal"], severity=0.5))
    assert sig.ndim == 1
    assert sig.min() > 0.0                                            # base rate added
    # The 'communal' index gets severity boost over the base.
    assert sig.max() >= 0.5


def test_find_analogues_returns_k_items():
    ctx = PipelineContext.dry_run(S=12, T=100, seed=1)
    out = find_analogues(ctx, Scenario(cleavages=["communal"], severity=0.6), k=5)
    assert len(out) == 5
    # Similarities should be sorted descending.
    sims = [a.similarity for a in out]
    assert sims == sorted(sims, reverse=True)
    assert all(isinstance(a, Analogue) for a in out)


def test_exclude_window_end_bounds_weeks():
    ctx = PipelineContext.dry_run(S=8, T=50, seed=2)
    out = find_analogues(ctx, Scenario(cleavages=["caste"]),
                          k=5, exclude_window_end=20)
    # iso_week == t + min_week (min_week=0 in dry_run), so all weeks < 20.
    assert all(a.iso_week < 20 for a in out)


def test_state_restrict_filters():
    ctx = PipelineContext.dry_run(S=36, T=80, seed=3)
    out = find_analogues(ctx, Scenario(cleavages=["caste"]),
                          k=4, state_restrict=["Kerala", "Punjab"])
    assert all(a.state in {"Kerala", "Punjab"} for a in out)


def test_article_index_attaches_articles():
    ctx = PipelineContext.dry_run(S=8, T=20, seed=4)
    # Attach a fake article to every (state, week) cell.
    idx = {(s, t): [{"article_id": f"a{s}_{t}",
                     "source": "ndtv.com",
                     "title": f"headline {s}-{t}"}]
           for s in range(ctx.S) for t in range(ctx.T_len)}
    out = find_analogues(ctx, Scenario(cleavages=["communal"]), k=3,
                          article_index=idx)
    assert out
    assert all(a.articles for a in out)
    assert out[0].articles[0]["source"] == "ndtv.com"


def test_analogue_to_dict_roundtrip_shape():
    a = Analogue(state="Kerala", iso_week=100, similarity=0.5,
                  cleavage_profile=[0.1, 0.2], articles=[])
    d = a.to_dict()
    assert set(d.keys()) >= {"state", "iso_week", "similarity",
                               "cleavage_profile", "articles"}


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
