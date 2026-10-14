"""Cached readers for the dashboard. Read-only, by design.

Everything here reads what the pipeline already wrote to out/ -- the lap
parquets, results.json and out/artifacts/. Nothing fits a Bayesian model:
a page that ran NUTS on load would take minutes to open and could show a
number no audit ever gated. The only fitting done here is the closed-form
least squares the audits themselves use (track evolution, rig recovery),
which is milliseconds.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import streamlit as st

from tyretrace import artifacts, results

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out"

LAP_FILES = {
    "race": OUT / "audit_race.parquet",
    "quali": OUT / "audit_quali.parquet",
    "practice": OUT / "audit_practice.parquet",
}


def _stamp(p: Path) -> float:
    """mtime, so a cache entry is invalidated when the pipeline rewrites."""
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=False)
def _read(p: str, stamp: float) -> pl.DataFrame | None:
    path = Path(p)
    return pl.read_parquet(path) if path.exists() else None


def laps(kind: str) -> pl.DataFrame | None:
    p = LAP_FILES[kind]
    return _read(str(p), _stamp(p))


def artifact(task: str, name: str) -> pl.DataFrame | None:
    p = artifacts.path(task, name)
    return _read(str(p), _stamp(p))


def result(task: str) -> dict:
    return results.get(task)


def metric(task: str, key: str, default=None):
    return results.get(task).get("metrics", {}).get(key, default)


def all_results() -> dict:
    return results.load()["tasks"]


@st.cache_data(show_spinner=False)
def session_names(stamp: float) -> dict[str, str]:
    """session_uid -> readable label, e.g. '2023_04_R' -> '2023 R04 · Baku'."""
    out = {}
    for kind in LAP_FILES:
        df = laps(kind)
        if df is None:
            continue
        for uid, circ in df.select("session_uid", "circuit").unique().iter_rows():
            y, rnd, ses = uid.split("_")
            out[uid] = f"{y} R{rnd} · {circ} · {ses}"
    return out


def label(uid: str) -> str:
    return session_names(_stamp(LAP_FILES["race"])).get(uid, uid)


def weekend_label(wk: str) -> str:
    """'2023_04' -> '2023 R04 · Baku'."""
    full = label(f"{wk}_R")
    return full.rsplit(" · ", 1)[0] if full != f"{wk}_R" else wk


def need(obj, task: str, what: str = "this view") -> bool:
    """Show how to produce a missing input, and tell the caller to stop."""
    if obj is not None:
        return True
    st.info(f"No data yet for {what}. Generate it with\n\n"
            f"```\npython -m tyretrace run {task}\n```")
    return False
