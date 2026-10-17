"""Shared colours and chart conventions, so every page reads the same way."""
from __future__ import annotations

import plotly.graph_objects as go

# Pirelli sidewall colours. HARD is white on a real tyre; grey here so it
# survives a light background.
COMPOUND = {
    "SOFT": "#E10600",
    "MEDIUM": "#F5C400",
    "HARD": "#9AA0A6",
    "INTERMEDIATE": "#2BA84A",
    "WET": "#1F6FD1",
    "UNKNOWN": "#7F7F7F",
}

RUNG = {"L0": "#8C8C8C", "L1": "#5B8DEF", "L2": "#2BA84A",
        "L3": "#F5A623", "L4": "#E10600", "L5": "#9B59B6",
        "L6": "#8E44AD"}
BASELINE = "#B0B4BA"          # all protocol baselines share one muted grey

GATE = {"PASS": "🟢", "SHIP": "🟢", "HOLD": "🟡", "FAIL": "🔴", "-": "⚪"}

BAND = (0.85, 0.95)            # protocol section 5 coverage band at 90%
PHYSICS_FUEL = (0.055, 0.060)  # s/lap implied by beta_f 0.030-0.035 x kappa ~1.8
BAND_FILL = "rgba(43,168,74,0.12)"
WARN_FILL = "rgba(225,6,0,0.08)"


def tidy(fig: go.Figure, *, height: int = 420, legend: bool = True) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      showlegend=legend,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="left", x=0))
    return fig
