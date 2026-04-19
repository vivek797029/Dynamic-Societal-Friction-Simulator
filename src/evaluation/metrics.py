"""Custom metrics for DSFS.

FSA — Friction Score Accuracy: Spearman correlation between aggregate F(r,t)
       and ACLED fatalities at (r, t+h), averaged across states.
EEP — Event Escalation Prediction: F1 on high-escalation weeks
       (top-quintile ACLED fatalities) predicted from F.
LTROC — Lead-Time ROC: AUC for detecting 'escalation imminent' windows.
TrustVal — Pearson correlation of τ_s with external MBFC/human reliability scores.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score


def friction_score_accuracy(F_agg: np.ndarray, y: np.ndarray) -> dict:
    """Spearman ρ between F(r, t) and y(r, t+h).  Both [S, T]."""
    rows = []
    for r in range(F_agg.shape[0]):
        fs = F_agg[r]; ys = y[r]
        # Drop zero-activity cells to avoid degeneracy.
        mask = (fs.var() > 0) & (ys.var() > 0)
        if not mask:
            continue
        rho, _ = spearmanr(fs, ys)
        if np.isnan(rho):
            continue
        rows.append(rho)
    return {"fsa_mean": float(np.mean(rows)) if rows else 0.0,
            "fsa_median": float(np.median(rows)) if rows else 0.0,
            "num_states_valid": len(rows)}


def event_escalation_prediction(F_agg: np.ndarray, y: np.ndarray,
                                quantile: float = 0.8) -> dict:
    """Binarize y at per-state quantile, binarize F at the same, return F1."""
    thr_y = np.quantile(y, quantile, axis=1, keepdims=True)
    thr_f = np.quantile(F_agg, quantile, axis=1, keepdims=True)
    y_bin = (y >= thr_y).astype(int).ravel()
    f_bin = (F_agg >= thr_f).astype(int).ravel()
    return {
        "eep_f1": float(f1_score(y_bin, f_bin, zero_division=0)),
        "eep_pos_rate": float(y_bin.mean()),
    }


def lead_time_auc(F_agg: np.ndarray, y: np.ndarray, quantile: float = 0.9) -> dict:
    """For each state, AUC of F at time t predicting y binarized at t+h."""
    aucs = []
    for r in range(F_agg.shape[0]):
        y_bin = (y[r] >= np.quantile(y[r], quantile)).astype(int)
        if y_bin.sum() < 2 or y_bin.sum() > len(y_bin) - 2:
            continue
        try:
            aucs.append(roc_auc_score(y_bin, F_agg[r]))
        except Exception:
            continue
    return {"lt_auc_mean": float(np.mean(aucs)) if aucs else 0.0,
            "num_states_scored": len(aucs)}


def trust_validation(tau_pred: dict[str, float],
                     tau_ref: dict[str, float]) -> dict:
    """Pearson ρ between predicted τ and an external reliability score (e.g. MBFC)."""
    common = sorted(set(tau_pred) & set(tau_ref))
    if len(common) < 5:
        return {"trust_pearson": 0.0, "n_common": len(common)}
    a = np.array([tau_pred[k] for k in common])
    b = np.array([tau_ref[k] for k in common])
    r = float(np.corrcoef(a, b)[0, 1])
    return {"trust_pearson": r, "n_common": len(common)}
