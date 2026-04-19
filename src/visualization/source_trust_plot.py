"""Ranked bar chart of learned source-trust τ_s with optional external-anchor overlay."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_source_trust(source_trust: pd.DataFrame,
                      out_path: str | Path,
                      top_n: int = 40,
                      external: pd.DataFrame | None = None) -> None:
    """`source_trust` columns: source_domain, tau.
    `external` optional columns: source_domain, ext_trust (same scale 0..1).
    """
    st = source_trust.sort_values("tau", ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.25)))
    ax.barh(st["source_domain"], st["tau"], color="steelblue", label="learned τ")
    if external is not None:
        ext = external.set_index("source_domain")["ext_trust"].to_dict()
        x = [ext.get(s, None) for s in st["source_domain"]]
        # Overlay as points for non-missing.
        for y_idx, (s, v) in enumerate(zip(st["source_domain"], x)):
            if v is not None:
                ax.scatter(v, y_idx, color="firebrick", zorder=3, label="external" if y_idx == 0 else None)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Trust score τ")
    ax.set_title(f"Top {top_n} sources by learned trust")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
