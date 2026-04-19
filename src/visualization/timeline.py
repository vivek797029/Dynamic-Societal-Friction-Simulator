"""Per-state friction timeline with cleavage stack bands."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..data.india_geo import STATE_TO_IDX
from ..data.preprocessing import CLEAVAGES


def plot_state_timeline(F_k: np.ndarray,  # [S, T, K]
                        F_agg: np.ndarray,  # [S, T]
                        state_name: str,
                        out_path: str | Path,
                        title: str | None = None) -> None:
    si = STATE_TO_IDX[state_name]
    T = F_agg.shape[1]
    t = np.arange(T)
    fig, ax = plt.subplots(figsize=(12, 5))
    # Stacked cleavage contributions.
    bottom = np.zeros(T)
    for k, c in enumerate(CLEAVAGES):
        ax.fill_between(t, bottom, bottom + F_k[si, :, k], label=c, alpha=0.85)
        bottom += F_k[si, :, k]
    # Overlay aggregate line.
    ax.plot(t, F_agg[si], color="black", linewidth=1.5, label="F(r,t) aggregate")
    ax.set_xlabel("ISO week index")
    ax.set_ylabel("Friction (z-scored components)")
    ax.set_title(title or f"Friction components — {state_name}")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
