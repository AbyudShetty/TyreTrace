"""Tyre Trace dashboard.  Launch:  python -m tyretrace app

Reads what the pipeline wrote to out/ and never fits a model itself, so it
opens in seconds. Run `python -m tyretrace run all` first to populate it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:          # streamlit only adds the script's dir
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Tyre Trace", page_icon="🛞", layout="wide")

VIEWS = Path(__file__).resolve().parent / "views"


def page(name: str, title: str, icon: str, default: bool = False):
    return st.Page(str(VIEWS / f"{name}.py"), title=title, icon=icon,
                   default=default)


nav = st.navigation({
    "Tyre Trace": [
        page("overview", "Overview", "🏁", default=True),
        page("problem", "The problem", "❓"),
    ],
    "Data": [
        page("races", "Race explorer", "🏎️"),
        page("track", "Track evolution", "🛣️"),
    ],
    "Models": [
        page("ladder", "Model ladder", "🪜"),
        page("predictions", "Held-out predictions", "🎯"),
        page("compounds", "Compound contrasts", "⚖️"),
        page("backtest", "FP → race backtest", "🔁"),
    ],
    "Decide": [
        page("planner", "Run planner", "🧭"),
    ],
    "Ground truth": [
        page("rig", "Game rig", "🎮"),
    ],
})

with st.sidebar:
    st.caption("Causal tyre-degradation intelligence. Every number on these "
               "pages was produced by a gated audit in `out/`.")

nav.run()
