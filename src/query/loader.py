"""PipelineContext loader from on-disk artifacts.

The training scripts produce a handful of artifacts per stage:

    artifacts/stage_b/stage_b.pt        trained FrictionAggregator + head
    artifacts/tensors/E.npy             [S, T, K] event tensor
    artifacts/tensors/T.npy             [S, T, K] media-tone tensor
    artifacts/tensors/R.npy             [S, T, K] relational-strain tensor (optional)
    artifacts/tensors/meta.json         {"min_week": int, "horizons": [...]}

This module stitches them into a `PipelineContext`. If torch is missing
or the `.pt` file can't be loaded, we still build a context with the
numpy tensors -- the simulator's dry-run forecast path will run.

The full format is intentionally minimal and easy to produce by hand
for testing: a caller can write E/T/R as numpy files and a one-line
meta.json, and this loader will be happy.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .intervention import PipelineContext

log = logging.getLogger(__name__)


def _load_npy(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return np.load(path, allow_pickle=False)
    except (OSError, ValueError) as e:                           # pragma: no cover
        log.warning("failed to load %s: %s", path, e)
        return None


def load_context(path: str | Path) -> PipelineContext:
    """Accept either a directory (with the tensor layout described above) or
    a single `.pt` checkpoint (we look for tensors/ next to it)."""
    p = Path(path)
    root = p if p.is_dir() else p.parent

    tensor_dir = root if (root / "E.npy").exists() else root / "tensors"
    E = _load_npy(tensor_dir / "E.npy")
    T = _load_npy(tensor_dir / "T.npy")
    R = _load_npy(tensor_dir / "R.npy")

    if E is None or T is None:
        log.warning("tensor files missing under %s; using dry-run context",
                    tensor_dir)
        return PipelineContext.dry_run()

    meta_path = tensor_dir / "meta.json"
    min_week = 0
    horizons: tuple[int, ...] = (1, 2, 4)
    if meta_path.exists():
        try:
            meta: dict[str, Any] = json.loads(meta_path.read_text("utf-8"))
            min_week = int(meta.get("min_week", 0))
            if "horizons" in meta:
                horizons = tuple(int(h) for h in meta["horizons"])
        except (OSError, ValueError) as e:                       # pragma: no cover
            log.warning("bad meta.json: %s", e)

    aggregator = None
    head = None
    ckpt_path = p if p.suffix == ".pt" else root / "stage_b.pt"
    if ckpt_path.exists():
        try:
            aggregator, head = _load_stage_b(ckpt_path)
        except Exception as e:                                   # noqa: BLE001
            log.warning("failed to load %s: %s", ckpt_path, e)

    return PipelineContext(
        E=E.astype(np.float32, copy=False),
        T=T.astype(np.float32, copy=False),
        R=R.astype(np.float32, copy=False) if R is not None else None,
        min_week=min_week,
        aggregator=aggregator,
        head=head,
        horizons=horizons,
    )


def _load_stage_b(ckpt_path: Path):
    """Return (aggregator, head). Lazy-imports torch + project modules."""
    try:
        import torch                                            # type: ignore
    except ImportError as e:
        raise RuntimeError("torch not installed; cannot load .pt") from e

    # Lazy imports -- these live in the training package and pull torch.
    from ..models.friction_aggregator import FrictionAggregator  # type: ignore
    from ..models.forecasting_head import EscalationHead          # type: ignore

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    aggregator = FrictionAggregator(**blob["aggregator_cfg"])
    aggregator.load_state_dict(blob["aggregator"])
    aggregator.eval()

    head = EscalationHead(**blob["head_cfg"])
    head.load_state_dict(blob["head"])
    head.eval()
    return aggregator, head
