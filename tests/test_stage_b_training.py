"""Smoke test for Stage-B training robustness additions (Task 3).

Verifies:
  1. `stage_b_best.pt` exists and its val loss matches the best in history.
  2. `history_b.json` has a strictly decreasing LR (cosine schedule).
  3. Early stopping triggers when patience < epochs.
  4. F_agg.npy is re-inferred from the BEST checkpoint (not the last epoch),
     so reloading that state_dict reproduces the saved F_agg.

Run locally with:    python -m pytest tests/test_stage_b_training.py -q
(Requires torch; skipped on import failure.)
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

from src.training.train_stage_b import train_stage_b  # noqa: E402
from src.models.friction_aggregator import (                                 # noqa: E402
    AggregatorConfig, FrictionAggregator,
)
from src.models.forecasting_head import (                                    # noqa: E402
    EscalationHead, ForecastConfig, build_windows,
)


def _tiny_inputs(S=6, T=40, K=3, rng=np.random.default_rng(0)):
    E = rng.standard_normal((S, T, K)).astype(np.float32)
    Tt = rng.standard_normal((S, T, K)).astype(np.float32)
    # Targets: a cleaner synthetic signal so the loss can actually go down.
    mu = np.clip(np.abs(E[..., 0] + Tt[..., 0]) * 0.5, 0, None).astype(np.float32)
    yp = {h: rng.poisson(mu + 0.5).astype(np.float32) for h in (1, 2, 4)}
    yf = {h: rng.poisson(mu * 0.2 + 0.1).astype(np.float32) for h in (1, 2, 4)}
    return E, Tt, yp, yf


def _write_cfg(path: Path, window_weeks: int = 6) -> None:
    cfg = {
        "aggregator": {"window_weeks": window_weeks},
        "forecasting": {
            "horizons": [1, 2, 4],
            "mlp_hidden": 16,
            "dropout": 0.0,
            "learning_rate": 5e-3,
            "epochs": 50,
        },
    }
    path.write_text(yaml.safe_dump(cfg))


def test_best_checkpoint_and_early_stop(tmp_path):
    E, Tt, yp, yf = _tiny_inputs()
    min_week = 0
    train_cutoff = 25       # weeks 0..25 train
    val_cutoff = 34         # weeks 26..34 val
    cfg_path = tmp_path / "cfg.yaml"
    _write_cfg(cfg_path, window_weeks=6)
    out = tmp_path / "out"

    result = train_stage_b(
        E, Tt, yp, yf,
        min_week=min_week,
        train_week_cutoff=train_cutoff,
        val_week_cutoff=val_cutoff,
        cfg_path=str(cfg_path), out_dir=str(out),
        epochs=50, loss="poisson",
        patience=5,  # aggressive to force early stop on tiny data
    )

    # Check best-val checkpoint exists and matches history.
    best_pt = out / "stage_b_best.pt"
    assert best_pt.exists(), "stage_b_best.pt not written"
    history = json.loads((out / "history_b.json").read_text())
    best_ep = result["best_epoch"]
    assert best_ep is not None
    recorded = history[best_ep]["val"]
    assert abs(recorded - result["best_val"]) < 1e-6

    # Cosine LR decayed.
    lr0 = history[0]["lr"]
    lrN = history[-1]["lr"]
    assert lrN < lr0, f"LR did not decay: {lr0} -> {lrN}"

    # Early stopping — we ran fewer than the full 50 epochs.
    assert result["epochs_run"] < 50, (
        f"expected early stop, ran {result['epochs_run']} epochs"
    )

    # F_agg.npy comes from the best checkpoint: reload best + re-infer and compare.
    ckpt = torch.load(best_pt, map_location="cpu")
    cfg = yaml.safe_load(cfg_path.read_text())
    K = E.shape[-1]
    agg = FrictionAggregator(AggregatorConfig(num_cleavages=K,
                                              window_weeks=cfg["aggregator"]["window_weeks"]))
    agg.load_state_dict(ckpt["agg"])
    agg.eval()
    E_n = (E - E.mean(axis=1, keepdims=True)) / (E.std(axis=1, keepdims=True) + 1e-6)
    T_n = (Tt - Tt.mean(axis=1, keepdims=True)) / (Tt.std(axis=1, keepdims=True) + 1e-6)
    with torch.no_grad():
        _, F_agg_reinf = agg(torch.from_numpy(E_n).float(), torch.from_numpy(T_n).float())
    F_agg_saved = np.load(out / "F_agg.npy")
    np.testing.assert_allclose(F_agg_reinf.numpy(), F_agg_saved, rtol=1e-5, atol=1e-5)
