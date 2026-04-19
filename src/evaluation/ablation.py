"""Ablation driver.

Runs the Stage-B training several times with different input tensor
configurations and reports FSA, EEP, LTROC for each plus the delta vs. `full`:

  * full      — E + T(trust-weighted)
  * no_trust  — E + T(un-weighted; τ_s ≡ 1)
  * no_text   — E + zeros_for_T
  * no_events — zeros_for_E + T(trust-weighted)

Changed in v1 cleanup: the `no_graph` variant (E + T with R removed) and the
R input were dropped alongside `src/models/actor_graph.py`. The aggregator
no longer takes an R channel.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..training.train_stage_b import train_stage_b
from .metrics import (event_escalation_prediction, friction_score_accuracy,
                      lead_time_auc)


def _eval(F_agg: np.ndarray, y: np.ndarray, horizon_label: str) -> dict:
    out = {}
    out.update({f"{horizon_label}/" + k: v for k, v in friction_score_accuracy(F_agg, y).items()})
    out.update({f"{horizon_label}/" + k: v for k, v in event_escalation_prediction(F_agg, y).items()})
    out.update({f"{horizon_label}/" + k: v for k, v in lead_time_auc(F_agg, y).items()})
    return out


def run_ablation(E: np.ndarray, T_trust: np.ndarray, T_plain: np.ndarray,
                 targets: dict, min_week: int,
                 train_cutoff_week: int, val_cutoff_week: int,
                 cfg_path: str, out_dir: str,
                 primary_horizon: int = 2) -> dict:
    root = Path(out_dir); root.mkdir(parents=True, exist_ok=True)
    variants = {
        "full":       (E, T_trust),
        "no_trust":   (E, T_plain),
        "no_text":    (E, np.zeros_like(T_trust)),
        "no_events":  (np.zeros_like(E), T_trust),
    }
    y_fatal = targets[f"y_fatalities_h{primary_horizon}"]
    results: dict[str, dict] = {}

    for name, (E_v, T_v) in variants.items():
        var_dir = root / name
        train_stage_b(E_v, T_v,
                      y_protests={h: targets[f"y_protests_h{h}"] for h in (1, 2, 4)},
                      y_fatalities={h: targets[f"y_fatalities_h{h}"] for h in (1, 2, 4)},
                      min_week=min_week,
                      train_week_cutoff=train_cutoff_week,
                      val_week_cutoff=val_cutoff_week,
                      cfg_path=cfg_path, out_dir=str(var_dir),
                      epochs=80)
        F_agg = np.load(var_dir / "F_agg.npy")
        m = _eval(F_agg, y_fatal, horizon_label=f"h{primary_horizon}")
        results[name] = m

    # Deltas vs full.
    base = results["full"]
    deltas = {}
    for name, m in results.items():
        if name == "full":
            continue
        deltas[name] = {k: m[k] - base[k] for k in m}

    (root / "ablation.json").write_text(json.dumps({"metrics": results, "deltas": deltas}, indent=2))
    return {"metrics": results, "deltas": deltas}
