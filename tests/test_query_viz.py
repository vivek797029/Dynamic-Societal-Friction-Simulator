"""Choropleth renderer: string-shape and stability tests."""
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

from src.query.viz import (ColorRamp, choropleth,                     # noqa: E402
                            choropleth_from_state_deltas, _TILE_LAYOUT)
from src.data.india_geo import STATES                                 # noqa: E402


def test_all_36_states_in_layout():
    assert set(STATES) == set(_TILE_LAYOUT.keys())
    assert len(_TILE_LAYOUT) == 36


def test_choropleth_returns_svg():
    svg = choropleth({"Kerala": 0.2, "Punjab": -0.1, "Maharashtra": 0.4})
    assert svg.startswith("<svg")
    assert svg.rstrip().endswith("</svg>")
    assert "Kerala" in svg                       # tooltip title
    assert "KL" in svg                            # abbreviation label


def test_choropleth_accepts_empty_values():
    svg = choropleth({}, title="empty")
    assert svg.startswith("<svg")
    # All tiles should render as "no data" -- abbreviation still present.
    assert "KL" in svg
    assert "no data" in svg


def test_choropleth_legend_reflects_vmax():
    svg = choropleth({"Kerala": 0.5}, vmax=1.0)
    assert "+1.00" in svg
    assert "-1.00" in svg


def test_color_ramp_neutral_at_zero():
    r = ColorRamp()
    c = r.color(0.0, vmax=1.0)
    # Neutral color for zero -> near-white.
    inside = c[c.index("(") + 1: c.index(")")]
    vals = [int(x) for x in inside.split(",")]
    assert all(v >= 240 for v in vals)


def test_color_ramp_no_data():
    r = ColorRamp()
    c = r.color(None, vmax=1.0)
    assert "220" in c                             # r of no_data is 220


def test_choropleth_from_state_deltas_accepts_iterable():
    svg = choropleth_from_state_deltas([("Kerala", 0.3), ("Goa", -0.2)])
    assert "Kerala" in svg
    assert "Goa" in svg


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
