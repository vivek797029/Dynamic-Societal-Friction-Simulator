"""Choropleth renderer for state-level friction deltas.

India has 28 states + 8 UTs (36 total) and rendering a proper vector
choropleth requires shipping geometry (shapefile / GeoJSON). To keep
the query layer dependency-light and to make the picture legible even
at small thumbnail sizes, we use a *tilegram* -- each state is one
square on a hand-laid grid that approximates India's geography. This
is the same idea used by NPR / FiveThirtyEight for US maps and by the
Hindustan Times for India.

Output is a self-contained SVG string (no external CSS or JS) so it
embeds directly in a Streamlit/Gradio panel, a notebook, or HTML email.

Usage:

    svg = choropleth(values={"Kerala": 0.4, "Punjab": -0.1, ...},
                     title="Projected protest rate change (+1 week)")
    Path("out.svg").write_text(svg)

All values are ingested as a dict[state_name -> float]. States missing
from the dict are rendered in a neutral "no data" fill so it's obvious
which states the scenario didn't touch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..data.india_geo import STATES


# --------------------------- tilegram layout --------------------------- #
# (row, col) on an 8-row x 10-col grid. Row 0 is north. Column 0 is west.
# Laid out to keep rough geographic adjacency: NE cluster on the right,
# northern belt at top, peninsular states down the middle, UTs nearby.
# This isn't geographically exact -- it's a UI device -- but it is stable.

_TILE_LAYOUT: dict[str, tuple[int, int]] = {
    # --- far north ---
    "Ladakh":                                       (0, 3),
    "Jammu and Kashmir":                            (0, 2),
    "Himachal Pradesh":                             (1, 2),
    "Punjab":                                       (1, 1),
    "Chandigarh":                                   (1, 0),
    "Uttarakhand":                                  (1, 3),

    # --- northern plain ---
    "Haryana":                                      (2, 1),
    "Delhi":                                        (2, 2),
    "Uttar Pradesh":                                (2, 3),
    "Bihar":                                        (2, 5),
    "Sikkim":                                       (1, 6),
    "Arunachal Pradesh":                            (1, 8),

    # --- northeast cluster ---
    "Assam":                                        (2, 7),
    "Nagaland":                                     (2, 8),
    "Meghalaya":                                    (3, 6),
    "Manipur":                                      (3, 8),
    "Mizoram":                                      (4, 8),
    "Tripura":                                      (3, 7),

    # --- east ---
    "West Bengal":                                  (3, 5),
    "Jharkhand":                                    (3, 4),
    "Odisha":                                       (4, 5),

    # --- central / west ---
    "Rajasthan":                                    (3, 1),
    "Madhya Pradesh":                               (3, 2),
    "Chhattisgarh":                                 (3, 3),
    "Gujarat":                                      (4, 0),
    "Dadra and Nagar Haveli and Daman and Diu":     (4, 1),
    "Maharashtra":                                  (4, 2),
    "Telangana":                                    (4, 3),
    "Andhra Pradesh":                               (5, 3),

    # --- peninsula ---
    "Goa":                                          (5, 1),
    "Karnataka":                                    (5, 2),
    "Kerala":                                       (6, 2),
    "Tamil Nadu":                                   (6, 3),
    "Puducherry":                                   (6, 4),

    # --- islands / lakshadweep ---
    "Lakshadweep":                                  (7, 1),
    "Andaman and Nicobar Islands":                  (7, 5),
}

# Two-letter abbreviations so tiles stay legible at small sizes.
_STATE_ABBR: dict[str, str] = {
    "Andhra Pradesh": "AP", "Arunachal Pradesh": "AR", "Assam": "AS",
    "Bihar": "BR", "Chhattisgarh": "CG", "Goa": "GA", "Gujarat": "GJ",
    "Haryana": "HR", "Himachal Pradesh": "HP", "Jammu and Kashmir": "JK",
    "Jharkhand": "JH", "Karnataka": "KA", "Kerala": "KL", "Madhya Pradesh": "MP",
    "Maharashtra": "MH", "Manipur": "MN", "Meghalaya": "ML", "Mizoram": "MZ",
    "Nagaland": "NL", "Odisha": "OD", "Punjab": "PB", "Rajasthan": "RJ",
    "Sikkim": "SK", "Tamil Nadu": "TN", "Telangana": "TS", "Tripura": "TR",
    "Uttar Pradesh": "UP", "Uttarakhand": "UK", "West Bengal": "WB",
    "Andaman and Nicobar Islands": "AN", "Chandigarh": "CH",
    "Dadra and Nagar Haveli and Daman and Diu": "DD", "Delhi": "DL",
    "Ladakh": "LA", "Lakshadweep": "LD", "Puducherry": "PY",
}


# ----------------------------- color ramp ----------------------------- #

@dataclass
class ColorRamp:
    """Diverging ramp with a neutral midpoint. Safe for +/- deltas."""
    negative: tuple[int, int, int] = (49, 130, 189)     # muted blue
    neutral: tuple[int, int, int] = (247, 247, 247)     # near-white
    positive: tuple[int, int, int] = (215, 48, 39)      # muted red
    no_data: tuple[int, int, int] = (220, 220, 220)     # light grey

    def _lerp(self, a: tuple[int, int, int], b: tuple[int, int, int],
              t: float) -> tuple[int, int, int]:
        t = max(0.0, min(1.0, t))
        return (
            int(round(a[0] + (b[0] - a[0]) * t)),
            int(round(a[1] + (b[1] - a[1]) * t)),
            int(round(a[2] + (b[2] - a[2]) * t)),
        )

    def color(self, value: float | None, vmax: float) -> str:
        if value is None:
            r, g, b = self.no_data
            return f"rgb({r},{g},{b})"
        if vmax <= 0:
            vmax = 1.0
        t = max(-1.0, min(1.0, float(value) / float(vmax)))
        if t >= 0:
            r, g, b = self._lerp(self.neutral, self.positive, t)
        else:
            r, g, b = self._lerp(self.neutral, self.negative, -t)
        return f"rgb({r},{g},{b})"


# -------------------------- svg rendering ----------------------------- #

_TILE_W = 64
_TILE_H = 48
_PAD = 4
_LEFT = 24
_TOP = 64


def _text_color_for(rgb_str: str) -> str:
    """Pick black/white label for contrast against the tile fill."""
    try:
        inside = rgb_str[rgb_str.index("(") + 1: rgb_str.index(")")]
        r, g, b = (int(x) for x in inside.split(","))
    except Exception:                                              # noqa: BLE001
        return "#111"
    # Rec. 709 luma.
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#111" if y >= 150 else "#fff"


def _svg_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
         .replace('"', "&quot;")
    )


def _fmt_val(x: float) -> str:
    ax = abs(x)
    if ax >= 10:
        return f"{x:+.1f}"
    if ax >= 1:
        return f"{x:+.2f}"
    return f"{x:+.2f}"


def choropleth(values: dict[str, float],
               title: str = "Friction delta by state",
               subtitle: str | None = None,
               ramp: ColorRamp | None = None,
               show_values: bool = True,
               vmax: float | None = None,
               ) -> str:
    """Render the tilegram as a standalone SVG string.

    Parameters
    ----------
    values      : dict of {state_name: float}. Missing states render as "no data".
    title       : main heading above the map.
    subtitle    : optional second line.
    ramp        : color ramp. Uses a diverging default if None.
    show_values : if True, print a short numeric label below the abbr.
    vmax        : color-scale max; defaults to the max abs value in `values`.
    """
    ramp = ramp or ColorRamp()
    if vmax is None:
        abs_vals = [abs(float(v)) for v in values.values() if v is not None]
        vmax = max(abs_vals) if abs_vals else 1.0
    vmax = float(vmax) if vmax > 0 else 1.0

    rows = max(r for r, _ in _TILE_LAYOUT.values()) + 1
    cols = max(c for _, c in _TILE_LAYOUT.values()) + 1

    w = _LEFT * 2 + cols * (_TILE_W + _PAD)
    h = _TOP + rows * (_TILE_H + _PAD) + 80       # room for legend

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" '
        f'font-family="-apple-system, Segoe UI, Roboto, sans-serif">'
    )
    # Background
    parts.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff"/>')
    # Title
    parts.append(
        f'<text x="{_LEFT}" y="28" font-size="18" font-weight="600" '
        f'fill="#111">{_svg_escape(title)}</text>'
    )
    if subtitle:
        parts.append(
            f'<text x="{_LEFT}" y="48" font-size="12" fill="#555">'
            f'{_svg_escape(subtitle)}</text>'
        )

    # Tiles
    for state in STATES:
        rc = _TILE_LAYOUT.get(state)
        if rc is None:                                              # pragma: no cover
            continue
        r, c = rc
        x = _LEFT + c * (_TILE_W + _PAD)
        y = _TOP + r * (_TILE_H + _PAD)
        v = values.get(state)
        fill = ramp.color(v, vmax) if v is not None else \
            f"rgb({ramp.no_data[0]},{ramp.no_data[1]},{ramp.no_data[2]})"
        text_fill = _text_color_for(fill)
        abbr = _STATE_ABBR.get(state, state[:2].upper())
        tooltip = f"{state}" + (f": {_fmt_val(float(v))}" if v is not None else ": n/a")
        parts.append(
            f'<g><title>{_svg_escape(tooltip)}</title>'
            f'<rect x="{x}" y="{y}" width="{_TILE_W}" height="{_TILE_H}" '
            f'rx="4" ry="4" fill="{fill}" stroke="#ffffff" stroke-width="1"/>'
            f'<text x="{x + _TILE_W / 2:.1f}" y="{y + _TILE_H / 2 - 2:.1f}" '
            f'text-anchor="middle" font-size="12" font-weight="600" '
            f'fill="{text_fill}">{abbr}</text>'
        )
        if show_values and v is not None:
            parts.append(
                f'<text x="{x + _TILE_W / 2:.1f}" y="{y + _TILE_H - 6:.1f}" '
                f'text-anchor="middle" font-size="10" fill="{text_fill}">'
                f'{_fmt_val(float(v))}</text>'
            )
        parts.append("</g>")

    # Legend
    legend_y = _TOP + rows * (_TILE_H + _PAD) + 16
    parts.append(_legend_svg(ramp=ramp, vmax=vmax, x=_LEFT, y=legend_y,
                             width=min(360, w - 2 * _LEFT)))
    parts.append("</svg>")
    return "".join(parts)


def _legend_svg(ramp: ColorRamp, vmax: float, x: int, y: int,
                width: int = 320) -> str:
    steps = 40
    cell_w = width / steps
    parts: list[str] = []
    parts.append(
        f'<text x="{x}" y="{y - 6}" font-size="11" fill="#555">'
        f'-{vmax:.2f}</text>'
    )
    parts.append(
        f'<text x="{x + width}" y="{y - 6}" font-size="11" fill="#555" '
        f'text-anchor="end">+{vmax:.2f}</text>'
    )
    for i in range(steps):
        t = -1.0 + 2.0 * (i + 0.5) / steps
        color = ramp.color(t * vmax, vmax)
        parts.append(
            f'<rect x="{x + i * cell_w:.2f}" y="{y}" width="{cell_w + 0.5:.2f}" '
            f'height="14" fill="{color}"/>'
        )
    # Midpoint tick.
    parts.append(
        f'<line x1="{x + width / 2}" y1="{y - 2}" x2="{x + width / 2}" '
        f'y2="{y + 16}" stroke="#111" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{x + width / 2}" y="{y + 30}" font-size="11" fill="#555" '
        f'text-anchor="middle">0</text>'
    )
    # No-data swatch to the right of the bar.
    nd_x = x + width + 16
    r, g, b = ramp.no_data
    parts.append(
        f'<rect x="{nd_x}" y="{y}" width="14" height="14" '
        f'fill="rgb({r},{g},{b})" stroke="#999"/>'
    )
    parts.append(
        f'<text x="{nd_x + 20}" y="{y + 11}" font-size="11" fill="#555">no data</text>'
    )
    return "".join(parts)


# --------------------------- convenience APIs -------------------------- #

def choropleth_from_state_deltas(state_deltas: Iterable[tuple[str, float]],
                                 title: str = "Friction delta by state",
                                 subtitle: str | None = None,
                                 ) -> str:
    """Shortcut accepting the output of SimulationResult.state_delta_ranked()."""
    return choropleth(dict(state_deltas), title=title, subtitle=subtitle)
