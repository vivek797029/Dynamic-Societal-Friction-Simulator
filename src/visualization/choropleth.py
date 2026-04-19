"""India state-level choropleth of aggregate friction or of a specific cleavage."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..data.india_geo import STATES, STATE_TO_IDX


# Canonical GeoJSON URL for India ADM1 (free, community-maintained).
INDIA_GEOJSON = ("https://raw.githubusercontent.com/"
                 "geohacker/india/master/state/india_telengana.geojson")


def plot_state_choropleth(F_agg: np.ndarray, week_idx: int,
                          out_path: str | Path,
                          title: str = "Aggregate Friction",
                          cmap: str = "Reds") -> None:
    """F_agg: [S, T]. Plots a choropleth for F_agg[:, week_idx].

    Tries geopandas; falls back to a bar chart if no network/geopandas.
    """
    vals = F_agg[:, week_idx]
    try:
        import geopandas as gpd  # type: ignore
        gdf = gpd.read_file(INDIA_GEOJSON)
        # Try to match name column.
        name_col = next((c for c in gdf.columns if c.lower() in {"name", "st_nm", "state", "admin1"}), "NAME_1")
        gdf["_friction"] = gdf[name_col].map(lambda s: vals[STATE_TO_IDX[s]] if s in STATE_TO_IDX else np.nan)
        ax = gdf.plot(column="_friction", cmap=cmap, legend=True, edgecolor="black", linewidth=0.3,
                      missing_kwds={"color": "lightgrey"}, figsize=(8, 10))
        ax.set_axis_off()
        ax.set_title(f"{title} — week {week_idx}")
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()
    except Exception as e:
        # Fallback: horizontal bar chart.
        order = np.argsort(vals)[::-1]
        fig, ax = plt.subplots(figsize=(8, 14))
        ax.barh([STATES[i] for i in order][::-1], vals[order][::-1], color="firebrick")
        ax.set_title(f"{title} — week {week_idx}  (geopandas unavailable: {e})")
        ax.set_xlabel("Friction")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
