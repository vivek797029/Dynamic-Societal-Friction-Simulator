"""Final evaluation — FSA, EEP, LTROC on the proper test split.

Run:
    python -m scripts.evaluate \
        --stage-b-dir artifacts/stage_b \
        --tensors     data/processed/targets.npz \
        --config      config.yaml \
        --min-week    $MIN_WEEK

The script loads the Stage-B `F_agg.npy` (re-inferred from the BEST checkpoint
by `train_stage_b.py`, per Task 3), masks it to weeks strictly after
`cfg.dates.val_cutoff`, and reports the three project metrics per horizon
against both ACLED target streams (protests, fatalities).

Metric definitions live in `src.evaluation.metrics`:
  * FSA   — friction_score_accuracy:    state-level Spearman ρ of F vs y_{t+h}
  * EEP   — event_escalation_prediction: F1 on top-quintile escalation weeks
  * LTROC — lead_time_auc:              state-level AUC of F predicting top-10%
                                         escalation t+h

All three are `dict`-returning (not scalars) — the script prints each field
and also writes a single `metrics.json` so downstream dashboards can pick up
the results without re-parsing the log.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

# Allow `python scripts/evaluate.py` as well as `python -m scripts.evaluate`.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

LOG = logging.getLogger("evaluate")


# ---------------------------------------------------------------------------
# Core evaluation (imported by run_pipeline.stage_evaluate too).
# ---------------------------------------------------------------------------

def _iso_week_from_date(date_str: str) -> int:
    """Convert an ISO date string to a monotonically-increasing week index.
    Uses `src.data.india_geo.iso_week_index` so the calculation is identical
    to the rest of the pipeline."""
    from src.data.india_geo import iso_week_index                       # noqa: PLC0415
    return int(iso_week_index(date_str))


def evaluate_test_split(F_agg_path: str | Path,
                        targets_path: str | Path,
                        horizons: Iterable[int],
                        min_week: int,
                        val_week_cutoff: int,
                        target_streams: tuple[str, ...] = ("y_protests", "y_fatalities"),
                        ) -> dict[str, Any]:
    """Compute FSA / EEP / LTROC on weeks strictly after `val_week_cutoff`.

    Arguments:
      F_agg_path     — artifact path to `F_agg.npy` from Stage B, [S, T].
      targets_path   — path to an .npz holding y_protests_h{h} / y_fatalities_h{h}
                        arrays, each shape [S, T].
      horizons       — iterable of forecast horizons (weeks), e.g. [1, 2, 4].
      min_week       — first week represented in F_agg's time axis.
      val_week_cutoff— last week to EXCLUDE from the test split.
      target_streams — which ACLED channels to score against.

    Returns a nested dict keyed by `f"h{h}"` then stream name with all metric
    fields, plus a `_meta` block describing the split.
    """
    # Lazy imports — keeps `--help` working without scipy / sklearn installed.
    from src.evaluation.metrics import (                                # noqa: PLC0415
        event_escalation_prediction,
        friction_score_accuracy,
        lead_time_auc,
    )
    F_agg = np.load(F_agg_path)
    if F_agg.ndim != 2:
        raise ValueError(f"F_agg.npy must be [S, T]; got shape {F_agg.shape}")
    targets = dict(np.load(targets_path))
    S, T_len = F_agg.shape

    abs_weeks = np.arange(T_len) + min_week
    test_idx = np.where(abs_weeks > val_week_cutoff)[0]
    results: dict[str, Any] = {
        "_meta": {
            "n_states": int(S),
            "n_test_weeks": int(test_idx.size),
            "first_test_week": int(abs_weeks[test_idx[0]]) if test_idx.size else None,
            "last_test_week": int(abs_weeks[test_idx[-1]]) if test_idx.size else None,
            "val_week_cutoff": int(val_week_cutoff),
            "F_agg_path": str(F_agg_path),
            "targets_path": str(targets_path),
        },
    }
    if test_idx.size == 0:
        results["error"] = "no_test_weeks"
        return results

    F_test = F_agg[:, test_idx]
    horizons = [int(h) for h in horizons]
    for h in horizons:
        horizon_block: dict[str, dict] = {}
        for stream in target_streams:
            key = f"{stream}_h{h}"
            if key not in targets:
                LOG.warning("missing target array %s — skipping", key)
                continue
            y = targets[key]
            if y.shape != F_agg.shape:
                # Trim or pad along time axis to match F_agg.
                T_y = y.shape[1]
                if T_y < T_len:
                    pad = np.zeros((y.shape[0], T_len - T_y), dtype=y.dtype)
                    y = np.concatenate([y, pad], axis=1)
                else:
                    y = y[:, :T_len]
            y_test = y[:, test_idx]
            horizon_block[stream] = {
                "FSA": friction_score_accuracy(F_test, y_test),
                "EEP": event_escalation_prediction(F_test, y_test),
                "LTROC": lead_time_auc(F_test, y_test),
            }
        results[f"h{h}"] = horizon_block
    return results


# ---------------------------------------------------------------------------
# Pretty-printer.
# ---------------------------------------------------------------------------

_NUMERIC_FIELDS = {
    "fsa_mean", "fsa_median",
    "eep_f1", "eep_pos_rate",
    "lt_auc_mean",
}


def format_report(results: dict[str, Any]) -> str:
    """Render the nested results dict into a stable, human-readable report."""
    lines = []
    meta = results.get("_meta", {})
    lines.append("=" * 72)
    lines.append("DSFS — final evaluation (test split)")
    lines.append("=" * 72)
    if meta:
        lines.append(f"  states            : {meta.get('n_states')}")
        lines.append(f"  test weeks (count): {meta.get('n_test_weeks')}")
        lines.append(f"  test week range   : {meta.get('first_test_week')} .. "
                     f"{meta.get('last_test_week')}  (> {meta.get('val_week_cutoff')})")
        lines.append(f"  F_agg             : {meta.get('F_agg_path')}")
        lines.append(f"  targets           : {meta.get('targets_path')}")
    lines.append("")

    if "error" in results:
        lines.append(f"  ERROR: {results['error']}")
        return "\n".join(lines)

    horizons = sorted(
        (k for k in results if k.startswith("h")),
        key=lambda s: int(s[1:]),
    )
    for hk in horizons:
        block = results[hk]
        lines.append(f"Horizon {hk}  (predict t + {hk[1:]} weeks):")
        for stream, metrics in block.items():
            lines.append(f"  target = {stream}")
            for mname in ("FSA", "EEP", "LTROC"):
                m = metrics.get(mname, {})
                kvs = ", ".join(
                    f"{k}={v:.4f}" if isinstance(v, (int, float)) and k in _NUMERIC_FIELDS
                    else f"{k}={v}"
                    for k, v in m.items()
                )
                lines.append(f"    {mname:<6s} {kvs}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="DSFS final evaluation — FSA / EEP / LTROC on test split.",
    )
    p.add_argument("--stage-b-dir", required=True,
                   help="Directory with F_agg.npy (Stage B output).")
    p.add_argument("--tensors", required=True,
                   help=".npz holding y_protests_h{h} / y_fatalities_h{h} arrays.")
    p.add_argument("--config", default="config.yaml",
                   help="YAML config (reads cleavages + dates + horizons).")
    p.add_argument("--min-week", type=int, default=None,
                   help="Absolute iso_week index of column 0 of F_agg.npy. "
                        "If omitted, read it from the Stage B manifest sitting "
                        "next to F_agg.npy if available.")
    p.add_argument("--val-cutoff-week", type=int, default=None,
                   help="Overrides cfg.dates.val_cutoff. Weeks > this index "
                        "form the test split.")
    p.add_argument("--out", default=None,
                   help="Output metrics.json path (default: <stage-b-dir>/metrics.json).")
    p.add_argument("--quiet", action="store_true", help="Only emit the JSON file.")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    stage_b_dir = Path(args.stage_b_dir)
    F_agg_path = stage_b_dir / "F_agg.npy"
    if not F_agg_path.exists():
        LOG.error("F_agg.npy not found under %s — did Stage B run?", stage_b_dir)
        return 2

    cfg = yaml.safe_load(open(args.config)) if Path(args.config).exists() else {}
    horizons = cfg.get("forecasting", {}).get("horizons", [1, 2, 4])

    # Resolve min_week.
    min_week = args.min_week
    if min_week is None:
        manifest_path = Path(cfg.get("paths", {}).get("artifacts", "artifacts")) / "manifest.json"
        if manifest_path.exists():
            try:
                mdata = json.loads(manifest_path.read_text())
                min_week = int(mdata["stages"]["build_e"]["min_week"])
                LOG.info("min_week auto-resolved from %s = %d", manifest_path, min_week)
            except Exception as e:
                LOG.warning("couldn't read min_week from %s (%s)", manifest_path, e)
    if min_week is None:
        LOG.error("--min-week is required (no manifest found to infer it)")
        return 2

    # Resolve val cutoff.
    val_cutoff = args.val_cutoff_week
    if val_cutoff is None:
        date_str = cfg.get("dates", {}).get("val_cutoff")
        if date_str is None:
            LOG.error("--val-cutoff-week is required (cfg.dates.val_cutoff also missing)")
            return 2
        val_cutoff = _iso_week_from_date(date_str)
        LOG.info("val_week_cutoff auto-resolved from cfg.dates.val_cutoff = %s -> week %d",
                 date_str, val_cutoff)

    results = evaluate_test_split(
        F_agg_path=F_agg_path,
        targets_path=args.tensors,
        horizons=horizons,
        min_week=min_week,
        val_week_cutoff=val_cutoff,
    )

    out_path = Path(args.out) if args.out else stage_b_dir / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    if not args.quiet:
        print(format_report(results))
        print(f"Wrote {out_path}")
    return 0 if "error" not in results else 1


if __name__ == "__main__":
    raise SystemExit(main())
