"""Stage B — aggregator + forecasting head.

Inputs:
  - E[S, T, K] — event intensity tensor  (from preprocessing.event_intensity_tensor)
  - T[S, T, K] — trust-weighted discourse tensor  (computed from Stage A outputs)
  - y_protests_h{h}, y_fatalities_h{h}  (from acled_loader.build_target_tensor)

Changed in v1 cleanup: the optional R[S, T, K] (relational strain) channel
was removed alongside `src/models/actor_graph.py`. The aggregator now operates
on (E, T) only. Older .npz files that contain an "R" array are ignored.

Trains on split by iso_week (train ≤ train_cutoff, val in (train_cutoff, val_cutoff],
test > val_cutoff). Supports Poisson (default) or Negative-Binomial NLL per
target for over-dispersed counts.

CRITICAL FIX (vs earlier version — see audit §1.1):
    The old `_targets` wrote rows in (t_end outer, state inner) order while
    `build_windows` reshapes `[S, N_t, L, K]` to `[N, L, K]` which flattens
    in (state outer, t_end inner) order. That scrambled every label.
    We now build targets in the SAME (S outer, N_t inner) order as
    `build_windows`, and assert agreement on a trivial synthetic case in the
    test harness (`tests/test_target_alignment.py`).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ..models.forecasting_head import (
    EscalationHead, ForecastConfig, NegativeBinomialHead,
    build_windows, negative_binomial_nll, poisson_nll,
)
from ..models.friction_aggregator import AggregatorConfig, FrictionAggregator

log = logging.getLogger(__name__)


def _zscore_per_state(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-6
    return (X - mu) / sd


def build_targets(
    y_protests: dict[int, np.ndarray],
    y_fatalities: dict[int, np.ndarray],
    horizons: tuple[int, ...],
    t_ends: list[int],
    S: int,
) -> np.ndarray:
    """Return a [N, H, 2] float32 array aligned with `build_windows` output.

    `build_windows` stacks windows along dim=1 to get `[S, N_t, L, K]`, then
    `reshape(N, L, K)` with N = S * N_t. The reshape uses C-order (row-major)
    so the OUTER axis is S (state) and the INNER axis is N_t (window/time).
    We build targets in the exact same order.
    """
    te = np.asarray(t_ends, dtype=np.int64)                      # [N_t]
    yp_stack = []
    yf_stack = []
    for h in horizons:
        yp_h = y_protests[h]                                     # [S, T]
        yf_h = y_fatalities[h]                                   # [S, T]
        # Safe gather: anywhere te >= T, return 0.
        def _gather(arr: np.ndarray) -> np.ndarray:
            S_, T_ = arr.shape
            out = np.zeros((S_, te.size), dtype=np.float32)
            ok = te < T_
            if ok.any():
                out[:, ok] = arr[:, te[ok]]
            return out
        yp_stack.append(_gather(yp_h))                           # [S, N_t]
        yf_stack.append(_gather(yf_h))                           # [S, N_t]
    yp = np.stack(yp_stack, axis=-1)                             # [S, N_t, H]
    yf = np.stack(yf_stack, axis=-1)                             # [S, N_t, H]
    tgts = np.stack([yp, yf], axis=-1)                           # [S, N_t, H, 2]
    assert tgts.shape[0] == S, f"S mismatch: {tgts.shape[0]} vs {S}"
    return tgts.reshape(S * te.size, len(horizons), 2)           # (S outer, N_t inner)


def _clone_state(*modules: torch.nn.Module) -> list[dict]:
    """Deep-copy state dicts to CPU so we don't hold GPU memory for the snapshot."""
    return [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in modules]


def train_stage_b(E: np.ndarray, T: np.ndarray,
                  y_protests: dict[int, np.ndarray], y_fatalities: dict[int, np.ndarray],
                  min_week: int, train_week_cutoff: int, val_week_cutoff: int,
                  cfg_path: str | Path, out_dir: str | Path,
                  epochs: int | None = None,
                  loss: str = "poisson",
                  patience: int = 20,
                  min_delta: float = 1e-4) -> dict:
    """Train Stage-B aggregator + forecasting head.

    Robustness additions (Task 3 of v1 cleanup):
      * Best-val checkpointing — `stage_b_best.pt` holds the state_dict at the
        epoch with lowest validation loss; `F_k.npy` / `F_agg.npy` are
        re-inferred from THAT checkpoint, not the last one.
      * Early stopping — halt if val loss hasn't improved by `min_delta` for
        `patience` epochs. Skipped if no val windows exist (tiny-demo case).
      * Cosine LR scheduler — `CosineAnnealingLR(T_max=epochs,
        eta_min=lr * 0.01)` so the learning rate smoothly decays to 1% of
        initial over the run.
    """
    cfg = yaml.safe_load(open(cfg_path))
    agg_cfg = cfg["aggregator"]
    fc_cfg = cfg["forecasting"]
    cleavage_names: list[str] = list(cfg.get("cleavages", []))
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Normalize.
    E_n = _zscore_per_state(E)
    T_n = _zscore_per_state(T)

    E_t = torch.from_numpy(E_n).float()
    T_t = torch.from_numpy(T_n).float()

    K = E_t.shape[-1]
    # Per-cleavage memory half-life (weeks) from config. Fall back to uniform
    # if the config is missing or the cleavage list doesn't match K.
    halflife_map = agg_cfg.get("memory_halflife_weeks") or {}
    halflife_weeks: list[float] | None = None
    if cleavage_names and len(cleavage_names) == K:
        try:
            halflife_weeks = [float(halflife_map[c]) for c in cleavage_names]
            log.info("per-cleavage memory half-lives (weeks): %s",
                     dict(zip(cleavage_names, halflife_weeks)))
        except KeyError as missing:
            log.warning("memory_halflife_weeks missing entry for %s — "
                        "falling back to uniform 16w init", missing)
            halflife_weeks = None
    else:
        log.warning("cleavage list in config has %d entries but K=%d — "
                    "falling back to uniform 16w memory init",
                    len(cleavage_names), K)

    agg = FrictionAggregator(AggregatorConfig(
        num_cleavages=K,
        window_weeks=agg_cfg["window_weeks"],
        halflife_weeks=halflife_weeks,
    ))
    head = EscalationHead(ForecastConfig(
        num_cleavages=K,
        window_len=agg_cfg["window_weeks"],
        hidden=fc_cfg["mlp_hidden"],
        dropout=fc_cfg["dropout"],
        horizons=tuple(fc_cfg["horizons"]),
    ))
    horizons = tuple(fc_cfg["horizons"])
    nb_head: NegativeBinomialHead | None = None
    if loss == "nb":
        nb_head = NegativeBinomialHead(num_horizons=len(horizons), num_targets=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agg.to(device); head.to(device)
    if nb_head is not None:
        nb_head.to(device)
    E_t, T_t = E_t.to(device), T_t.to(device)

    params = list(agg.parameters()) + list(head.parameters())
    if nb_head is not None:
        params += list(nb_head.parameters())
    lr0 = float(fc_cfg["learning_rate"])
    optim = torch.optim.AdamW(params, lr=lr0, weight_decay=1e-4)

    epochs = epochs or fc_cfg["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, epochs), eta_min=lr0 * 0.01,
    )

    history: list[dict] = []
    best_val = float("inf")
    best_epoch = -1
    best_state: list[dict] | None = None
    epochs_since_improve = 0
    S = E_t.shape[0]

    # Precompute the time-split masks ONCE (they don't change across epochs).
    # We still need t_ends from build_windows, so we do one dry run.
    with torch.no_grad():
        F_k0, F_agg0 = agg(E_t, T_t)
        _, _, t_ends = build_windows(F_k0, F_agg0, agg_cfg["window_weeks"])
    t_end_arr = np.asarray(t_ends)
    abs_weeks = t_end_arr + min_week
    train_mask_t = abs_weeks <= train_week_cutoff
    val_mask_t = (abs_weeks > train_week_cutoff) & (abs_weeks <= val_week_cutoff)
    train_mask = np.tile(train_mask_t[None, :], (S, 1)).reshape(-1)
    val_mask = np.tile(val_mask_t[None, :], (S, 1)).reshape(-1)
    has_val = bool(val_mask.any())
    if not has_val:
        log.warning("no validation windows between weeks (%s, %s]; "
                    "early-stopping + best-val checkpoint disabled",
                    train_week_cutoff, val_week_cutoff)

    # Targets are also static — build them once.
    tgts_np = build_targets(y_protests, y_fatalities, horizons, t_ends, S)
    y = torch.from_numpy(tgts_np).to(device)

    for ep in range(epochs):
        agg.train(); head.train()
        F_k, F_agg = agg(E_t, T_t)
        Xk, Xag, _ = build_windows(F_k, F_agg, agg_cfg["window_weeks"])

        pred = head(Xk, Xag)  # [N, H, 2]  -- log-rate / log-mu
        if loss == "poisson":
            loss_train = poisson_nll(pred[train_mask], y[train_mask])
        elif loss == "nb":
            r = nb_head.dispersion()                                 # [H, 2]
            loss_train = negative_binomial_nll(pred[train_mask], y[train_mask], r)
        else:
            raise ValueError(f"unknown loss={loss!r}")
        optim.zero_grad()
        loss_train.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optim.step()
        sched.step()

        # ---- validation ----
        agg.eval(); head.eval()
        with torch.no_grad():
            F_k_e, F_agg_e = agg(E_t, T_t)
            Xk_e, Xag_e, _ = build_windows(F_k_e, F_agg_e, agg_cfg["window_weeks"])
            pred_e = head(Xk_e, Xag_e)
            if has_val:
                if loss == "poisson":
                    loss_val = poisson_nll(pred_e[val_mask], y[val_mask])
                else:
                    loss_val = negative_binomial_nll(
                        pred_e[val_mask], y[val_mask], nb_head.dispersion()
                    )
            else:
                loss_val = torch.tensor(float("nan"))

        lr_now = optim.param_groups[0]["lr"]
        rec = {"epoch": ep, "train": float(loss_train),
               "val": float(loss_val), "lr": float(lr_now)}
        history.append(rec)

        # ---- best-val bookkeeping + early stopping ----
        improved = False
        if has_val and not (loss_val.isnan() or loss_val.isinf()):
            if float(loss_val) < best_val - min_delta:
                best_val = float(loss_val)
                best_epoch = ep
                modules = [agg, head] + ([nb_head] if nb_head is not None else [])
                best_state = _clone_state(*modules)
                epochs_since_improve = 0
                improved = True
            else:
                epochs_since_improve += 1

        if ep % 10 == 0 or improved:
            log.info("ep %3d  train=%.4f  val=%.4f  lr=%.2e%s",
                     ep, rec["train"], rec["val"], lr_now,
                     "  * new best" if improved else "")

        if has_val and epochs_since_improve >= patience:
            log.info("early stop at ep=%d (no val improvement for %d epochs; "
                     "best val=%.4f at ep=%d)",
                     ep, patience, best_val, best_epoch)
            break

    # ---- final checkpoint + best checkpoint ----
    last_ckpt = {"agg": agg.state_dict(), "head": head.state_dict(), "loss": loss}
    if nb_head is not None:
        last_ckpt["nb_head"] = nb_head.state_dict()
    torch.save(last_ckpt, Path(out_dir) / "stage_b.pt")

    if best_state is not None:
        best_ckpt = {
            "agg": best_state[0],
            "head": best_state[1],
            "loss": loss,
            "best_epoch": best_epoch,
            "best_val": best_val,
        }
        if nb_head is not None:
            best_ckpt["nb_head"] = best_state[2]
        torch.save(best_ckpt, Path(out_dir) / "stage_b_best.pt")
        # Restore best weights for the friction-tensor export below.
        agg.load_state_dict({k: v.to(device) for k, v in best_state[0].items()})
        head.load_state_dict({k: v.to(device) for k, v in best_state[1].items()})
        if nb_head is not None:
            nb_head.load_state_dict({k: v.to(device) for k, v in best_state[2].items()})
        log.info("loaded best checkpoint (ep=%d val=%.4f) before exporting F_*.npy",
                 best_epoch, best_val)

    Path(out_dir, "history_b.json").write_text(json.dumps(history, indent=2))

    # Persist friction tensor from the (now best) model for evaluation/viz.
    agg.eval(); head.eval()
    with torch.no_grad():
        F_k, F_agg = agg(E_t, T_t)
    np.save(Path(out_dir, "F_k.npy"), F_k.detach().cpu().numpy())
    np.save(Path(out_dir, "F_agg.npy"), F_agg.detach().cpu().numpy())

    return {
        "train": history[-1]["train"],
        "val": history[-1]["val"],
        "best_val": best_val if best_state is not None else None,
        "best_epoch": best_epoch if best_state is not None else None,
        "epochs_run": len(history),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tensors", required=True, help=".npz with E, T, R (optional), targets")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", default="./artifacts/stage_b")
    p.add_argument("--train-cutoff-week", type=int, required=True)
    p.add_argument("--val-cutoff-week", type=int, required=True)
    p.add_argument("--min-week", type=int, required=True)
    p.add_argument("--loss", choices=["poisson", "nb"], default="poisson")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=20,
                   help="early-stop patience in epochs; 0 disables early stopping")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    data = np.load(args.tensors, allow_pickle=True)
    E = data["E"]; T = data["T"]
    if "R" in data.files:
        log.warning("ignoring 'R' tensor in %s: relational strain was removed in v1", args.tensors)
    horizons = [1, 2, 4]
    y_p = {h: data[f"y_protests_h{h}"] for h in horizons}
    y_f = {h: data[f"y_fatalities_h{h}"] for h in horizons}

    # patience=0 → disable early stopping (set to a very large number).
    patience = args.patience if args.patience > 0 else 10**9
    train_stage_b(E, T, y_p, y_f, args.min_week,
                  args.train_cutoff_week, args.val_cutoff_week,
                  args.config, args.out, epochs=args.epochs, loss=args.loss,
                  patience=patience)


if __name__ == "__main__":
    main()
