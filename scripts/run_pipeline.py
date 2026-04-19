#!/usr/bin/env python
"""End-to-end pipeline driver for Dynamic Societal Friction Simulator.

Chains every step from raw data to trained artifacts and test-set metrics.

This driver matches the ACTUAL signatures in the repo (not aspirational
ones). Two execution paths:

    A. GDELT-only path (default, runs today with no external article text):
        1. download GDELT CSVs            -> data/raw/gdelt/{export,mentions,gkg}/*.parquet
        2. load + attach cleavage + E     -> data/processed/E.npy
        3. load ACLED + build targets     -> data/processed/targets.npz
        4. build a GDELT-tone T proxy     -> data/processed/T.npy      (NOT Stage-A T)
        5. Stage B train + eval           -> artifacts/stage_b/*
        6. test-split metrics             -> artifacts/metrics.json

    B. Full path (if articles.parquet + atoms.pkl exist under data/processed/):
        Same steps, plus Stage A training and the real trust-weighted T.

Resume-friendly: every stage records completion in artifacts/manifest.json
so reruns (after a Colab timeout) pick up where they left off. Seeds are
set once at startup.

Usage:
    # Full run (downloads ~5 GB of GDELT for the configured date range):
    python scripts/run_pipeline.py --config config.yaml

    # Skip download if you already have raw data:
    python scripts/run_pipeline.py --skip-download

    # Fast iteration on Stage B only:
    python scripts/run_pipeline.py \\
        --skip-download --skip-build-e --skip-targets \\
        --skip-stage-a --skip-build-t

    # Small sanity run on a 6-month slice:
    python scripts/run_pipeline.py --start 2022-01-01 --end 2022-06-30
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

LOG = logging.getLogger("run_pipeline")


# --------------------------- configuration ----------------------------- #

def load_config(path: Path) -> dict:
    """Load YAML config and canonicalize paths to absolute."""
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = path.resolve().parent
    for k, v in cfg.get("paths", {}).items():
        if isinstance(v, str) and not os.path.isabs(v):
            cfg["paths"][k] = str((root / v).resolve())
    return cfg


def set_seeds(seed: int) -> None:
    """Make everything we can reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch                                                     # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


# --------------------------- manifest ---------------------------------- #

class Manifest:
    """Tracks stage completions. JSON at artifacts/manifest.json."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, Any] = {"stages": {}, "meta": {}}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text("utf-8"))
            except json.JSONDecodeError:
                LOG.warning("manifest malformed; starting fresh")

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2, default=str), "utf-8")

    def done(self, stage: str) -> bool:
        return stage in self.data["stages"] and \
            self.data["stages"][stage].get("status") == "ok"

    def mark(self, stage: str, **kwargs) -> None:
        self.data["stages"][stage] = {
            "status": "ok",
            "finished_at": dt.datetime.utcnow().isoformat() + "Z",
            **kwargs,
        }
        self.save()


# -------------------------- stage wrappers ----------------------------- #
# Every stage lazy-imports its dependencies so `--help` works without
# torch, and so a broken optional dep doesn't kill the whole driver.

def stage_download(cfg: dict, args, manifest: Manifest) -> None:
    if args.skip_download or manifest.done("download"):
        LOG.info("[1/7] download: skip")
        return
    LOG.info("[1/7] download: GDELT %s .. %s", args.start, args.end)
    from src.data.gdelt_downloader import download_range                 # noqa: PLC0415
    start_dt = dt.datetime.fromisoformat(args.start)
    # Inclusive end-of-day.
    end_dt = dt.datetime.fromisoformat(args.end) + dt.timedelta(hours=23, minutes=45)
    out_dir = Path(cfg["paths"]["raw_gdelt"])
    download_range(start_dt, end_dt, out_dir, kinds=("export",))
    manifest.mark("download", out=str(out_dir),
                   start=args.start, end=args.end, kinds=["export"])


def stage_build_e(cfg: dict, args, manifest: Manifest) -> tuple[Path, int, int]:
    """Load GDELT events → attach cleavage → event-intensity tensor E.

    Returns (E_path, min_week, max_week).
    """
    e_path = Path(cfg["paths"]["processed"]) / "E.npy"
    meta_path = Path(cfg["paths"]["processed"]) / "E_meta.json"
    if not args.skip_build_e and not manifest.done("build_e"):
        LOG.info("[2/7] build_e: GDELT -> cleavage -> E[S,T,K]")
        from src.data.preprocessing import (                             # noqa: PLC0415
            attach_cleavage_from_actors, event_intensity_tensor, load_gdelt_events,
        )
        gdelt_dir = Path(cfg["paths"]["raw_gdelt"]) / "export"
        df = load_gdelt_events(gdelt_dir)
        if df.empty:
            raise RuntimeError(
                f"No GDELT parquets under {gdelt_dir}. "
                f"Run without --skip-download first."
            )
        df = attach_cleavage_from_actors(df)
        min_week = int(df["iso_week"].min())
        max_week = int(df["iso_week"].max())
        E = event_intensity_tensor(df, min_week=min_week, max_week=max_week)
        e_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(e_path, E.astype(np.float32))
        meta_path.write_text(json.dumps({
            "min_week": min_week, "max_week": max_week,
            "shape": list(E.shape),
        }, indent=2))
        manifest.mark("build_e", out=str(e_path),
                       min_week=min_week, max_week=max_week,
                       shape=list(E.shape))
        LOG.info("[2/7] build_e: %s shape=%s", e_path, E.shape)
    # Load metadata to return even if skipped.
    meta = json.loads(meta_path.read_text("utf-8"))
    return e_path, int(meta["min_week"]), int(meta["max_week"])


def stage_targets(cfg: dict, args, manifest: Manifest,
                   min_week: int, max_week: int) -> Path:
    out = Path(cfg["paths"]["processed"]) / "targets.npz"
    if args.skip_targets or manifest.done("targets"):
        LOG.info("[3/7] targets: skip")
        return out
    LOG.info("[3/7] targets: ACLED -> [S,T] protest/fatality tensors")
    from src.data.acled_loader import build_target_tensor, load_acled    # noqa: PLC0415
    acled_dir = Path(cfg["paths"]["raw_acled"])
    csvs = sorted(acled_dir.glob("*.csv"))
    if not csvs:
        raise RuntimeError(
            f"No ACLED CSVs under {acled_dir}. "
            f"Download from https://acleddata.com/data-export-tool/ "
            f"and place under {acled_dir}."
        )
    # ACLED exports are usually a single CSV; concatenate if multiple.
    import pandas as pd                                                  # noqa: PLC0415
    df = pd.concat([load_acled(p) for p in csvs], ignore_index=True)
    horizons = tuple(cfg["forecasting"]["horizons"])
    tgts = build_target_tensor(df, horizons=horizons,
                                 min_week=min_week, max_week=max_week)
    # Drop non-array metadata before saving.
    saveable = {k: v for k, v in tgts.items()
                if isinstance(v, np.ndarray)}
    np.savez(out, **saveable)
    manifest.mark("targets", out=str(out), horizons=list(horizons),
                   n_acled=int(len(df)))
    return out


def stage_stage_a(cfg: dict, args, manifest: Manifest) -> Path | None:
    """Run Stage A if articles.parquet + atoms.pkl exist; otherwise skip cleanly.

    Returns path to checkpoint, or None if skipped.
    """
    out_dir = Path(cfg["paths"]["artifacts"]) / "stage_a"
    articles_path = Path(cfg["paths"]["processed"]) / "articles.parquet"
    atoms_path = Path(cfg["paths"]["processed"]) / "atoms.pkl"
    if args.skip_stage_a or manifest.done("stage_a"):
        LOG.info("[4/7] stage_a: skip")
        return out_dir if out_dir.exists() else None
    if not articles_path.exists() or not atoms_path.exists():
        LOG.warning("[4/7] stage_a: %s or %s missing; skipping "
                     "(will fall back to GDELT-tone T).",
                     articles_path.name, atoms_path.name)
        manifest.mark("stage_a", status="skipped",
                       reason="articles.parquet or atoms.pkl missing")
        return None
    LOG.info("[4/7] stage_a: trust learning (%d epochs)",
              cfg["trust_learning"]["epochs"])
    from src.training.train_stage_a import train_stage_a                 # noqa: PLC0415
    train_stage_a(
        articles_parquet=articles_path,
        atoms_pickle=atoms_path,
        cfg_path=args.config,
        out_dir=out_dir,
        seed=args.seed,
    )
    manifest.mark("stage_a", out_dir=str(out_dir))
    return out_dir


def stage_build_t(cfg: dict, args, manifest: Manifest,
                   stage_a_dir: Path | None,
                   min_week: int, max_week: int) -> Path:
    """Build T[S,T,K].

    Full path: use Stage-A outputs (source_trust, cleavage_probs, hostility).
    Fallback: derive a GDELT-tone-based T proxy from events' AvgTone so the
    rest of the pipeline runs end-to-end today.
    """
    t_path = Path(cfg["paths"]["processed"]) / "T.npy"
    if args.skip_build_t or manifest.done("build_t"):
        LOG.info("[5/7] build_t: skip")
        return t_path

    articles_path = Path(cfg["paths"]["processed"]) / "articles.parquet"
    cleavage_probs_path = (stage_a_dir / "cleavage_probs.npy") if stage_a_dir else None
    hostility_path = (stage_a_dir / "hostility.npy") if stage_a_dir else None
    source_trust_path = (stage_a_dir / "source_trust.parquet") if stage_a_dir else None
    have_full = (stage_a_dir is not None
                 and articles_path.exists()
                 and cleavage_probs_path.exists()
                 and hostility_path.exists()
                 and source_trust_path.exists())

    if have_full:
        LOG.info("[5/7] build_t: full trust-weighted T from Stage A")
        from src.training.build_T_tensor import build_T_tensor           # noqa: PLC0415
        import pandas as pd                                              # noqa: PLC0415
        articles = pd.read_parquet(articles_path)
        source_trust = pd.read_parquet(source_trust_path)
        cleavage_probs = np.load(cleavage_probs_path)
        hostility = np.load(hostility_path)
        T = build_T_tensor(articles, source_trust, cleavage_probs, hostility,
                            min_week=min_week, max_week=max_week)
    else:
        LOG.info("[5/7] build_t: GDELT-tone T proxy (no Stage A outputs)")
        T = _gdelt_tone_tensor_fallback(cfg, min_week, max_week)
    np.save(t_path, T.astype(np.float32))
    manifest.mark("build_t", out=str(t_path), mode="full" if have_full else "fallback",
                   shape=list(T.shape))
    return t_path


def _gdelt_tone_tensor_fallback(cfg: dict, min_week: int, max_week: int) -> np.ndarray:
    """Build a T[S,T,K] using GDELT's own AvgTone magnitude per cleavage.

    This is a legitimate proxy -- the media-tone channel is exactly what T
    is meant to capture. It just bypasses Stage A's trust weighting. Every
    article in a state-week contributes |AvgTone| to its (possibly 'other')
    cleavage slot; we drop 'other' and keep the 6 canonical cleavages.
    """
    from src.data.preprocessing import (                                 # noqa: PLC0415
        CLEAVAGES, attach_cleavage_from_actors, load_gdelt_events,
    )
    from src.data.india_geo import NUM_STATES, STATE_TO_IDX              # noqa: PLC0415
    gdelt_dir = Path(cfg["paths"]["raw_gdelt"]) / "export"
    df = load_gdelt_events(gdelt_dir)
    df = attach_cleavage_from_actors(df)
    import pandas as pd                                                  # noqa: PLC0415
    df["AvgTone"] = pd.to_numeric(df.get("AvgTone", 0), errors="coerce").fillna(0.0)
    S = NUM_STATES
    Tn = max_week - min_week + 1
    K = len(CLEAVAGES)
    T = np.zeros((S, Tn, K), dtype=np.float32)
    ci = {c: i for i, c in enumerate(CLEAVAGES)}
    mask = df["cleavage"].isin(CLEAVAGES)
    sub = df[mask]
    for state, w, c, tone in zip(sub["state"], sub["iso_week"],
                                    sub["cleavage"], sub["AvgTone"].abs()):
        si = STATE_TO_IDX.get(state)
        if si is None:
            continue
        ti = int(w) - min_week
        if ti < 0 or ti >= Tn:
            continue
        T[si, ti, ci[c]] += float(tone)
    return T


def stage_stage_b(cfg: dict, args, manifest: Manifest,
                   e_path: Path, t_path: Path, targets_path: Path,
                   min_week: int) -> Path:
    out_dir = Path(cfg["paths"]["artifacts"]) / "stage_b"
    if args.skip_stage_b or manifest.done("stage_b"):
        LOG.info("[6/7] stage_b: skip")
        return out_dir
    LOG.info("[6/7] stage_b: aggregator + forecasting head (loss=%s)", args.loss)
    from src.training.train_stage_b import train_stage_b                 # noqa: PLC0415
    E = np.load(e_path)
    T = np.load(t_path)
    targets = dict(np.load(targets_path))
    horizons = list(cfg["forecasting"]["horizons"])
    yp = {h: targets[f"y_protests_h{h}"] for h in horizons}
    yf = {h: targets[f"y_fatalities_h{h}"] for h in horizons}
    train_week = _iso_week(cfg["dates"]["train_cutoff"])
    val_week = _iso_week(cfg["dates"]["val_cutoff"])
    patience_b = args.patience_b if args.patience_b > 0 else 10**9
    train_stage_b(
        E, T, yp, yf,
        min_week=min_week,
        train_week_cutoff=train_week,
        val_week_cutoff=val_week,
        cfg_path=args.config,
        out_dir=out_dir,
        epochs=args.epochs_b,
        loss=args.loss,
        patience=patience_b,
    )
    manifest.mark("stage_b", out_dir=str(out_dir),
                   loss=args.loss,
                   train_cutoff_week=int(train_week),
                   val_cutoff_week=int(val_week))
    return out_dir


def stage_evaluate(cfg: dict, args, manifest: Manifest,
                    e_path: Path, targets_path: Path, stage_b_dir: Path,
                    min_week: int) -> Path:
    out = Path(cfg["paths"]["artifacts"]) / "metrics.json"
    if args.skip_eval or manifest.done("evaluate"):
        LOG.info("[7/7] evaluate: skip")
        return out
    LOG.info("[7/7] evaluate on TEST split (iso_week > %s)",
              cfg["dates"]["val_cutoff"])
    # Delegate to the standalone evaluation script so the pipeline and the
    # `scripts/evaluate.py` CLI produce identical numbers.
    from scripts.evaluate import evaluate_test_split                     # noqa: PLC0415
    F_agg_path = stage_b_dir / "F_agg.npy"
    if not F_agg_path.exists():
        LOG.warning("[7/7] %s missing; skipping eval", F_agg_path)
        return out
    val_week = _iso_week(cfg["dates"]["val_cutoff"])
    results = evaluate_test_split(
        F_agg_path=F_agg_path,
        targets_path=targets_path,
        horizons=cfg["forecasting"]["horizons"],
        min_week=min_week,
        val_week_cutoff=val_week,
    )
    out.write_text(json.dumps(results, indent=2))
    manifest.mark("evaluate", out=str(out),
                   test_weeks=results.get("_meta", {}).get("n_test_weeks"),
                   summary={k: v for k, v in results.items() if k != "_meta"})
    LOG.info("[7/7] test metrics:\n%s", json.dumps(results, indent=2))
    return out


# ------------------------------- helpers ------------------------------- #

def _iso_week(date_str: str) -> int:
    from src.data.india_geo import iso_week_index                        # noqa: PLC0415
    return iso_week_index(dt.date.fromisoformat(date_str))


# -------------------------------- main --------------------------------- #

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="End-to-end driver for Dynamic Societal Friction Simulator."
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start", default=None,
                   help="Override config dates.start (YYYY-MM-DD).")
    p.add_argument("--end", default=None,
                   help="Override config dates.end (YYYY-MM-DD).")
    p.add_argument("--loss", choices=["poisson", "nb"], default="nb")
    p.add_argument("--epochs-b", type=int, default=None)
    p.add_argument("--patience-b", type=int, default=20,
                   help="Stage B early-stop patience (epochs); 0 disables")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-build-e", action="store_true")
    p.add_argument("--skip-targets", action="store_true")
    p.add_argument("--skip-stage-a", action="store_true")
    p.add_argument("--skip-build-t", action="store_true")
    p.add_argument("--skip-stage-b", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Ignore manifest; rerun every non-skipped stage.")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose >= 2
        else logging.INFO if args.verbose == 1 else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    LOG.setLevel(logging.INFO)

    set_seeds(args.seed)
    cfg = load_config(Path(args.config))
    args.start = args.start or cfg["dates"]["start"]
    args.end = args.end or cfg["dates"]["end"]

    artifacts = Path(cfg["paths"]["artifacts"])
    artifacts.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(artifacts / "manifest.json")
    if args.force:
        manifest.data["stages"] = {}
        manifest.save()
    manifest.data["meta"] = {
        "config_path": str(Path(args.config).resolve()),
        "seed": args.seed,
        "start": args.start, "end": args.end,
        "python": sys.version.split()[0],
        "argv": sys.argv,
    }
    manifest.save()

    t0 = time.time()
    stage_download(cfg, args, manifest)
    e_path, min_week, max_week = stage_build_e(cfg, args, manifest)
    targets_path = stage_targets(cfg, args, manifest, min_week, max_week)
    stage_a_dir = stage_stage_a(cfg, args, manifest)
    t_path = stage_build_t(cfg, args, manifest, stage_a_dir, min_week, max_week)
    stage_b_dir = stage_stage_b(cfg, args, manifest, e_path, t_path,
                                   targets_path, min_week)
    stage_evaluate(cfg, args, manifest, e_path, targets_path, stage_b_dir,
                    min_week)

    LOG.info("pipeline finished in %.1fs; manifest=%s",
              time.time() - t0, manifest.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
