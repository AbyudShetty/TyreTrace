"""Which lap tables an analysis reads.

    audit   out/audit_{race,quali,practice}.parquet -- the original 10
            weekends from FastF1. Fixed, so every gated audit stays
            reproducible exactly as recorded.
    lake    out/lake/{race,quali,practice}.parquet -- every weekend from 2023
            via OpenF1 (tyretrace.data.bulk). Used where more weekends buy
            statistical power: the backtest, the fuel-gap test, 2026.

Both carry the same canonical schema, and audit/d1_source_parity.py checks
the two sources agree on every session they share.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

OUT = Path(__file__).resolve().parents[2] / "out"
PATHS = {
    "audit": {k: OUT / f"audit_{k}.parquet" for k in ("race", "quali", "practice")},
    "lake": {k: OUT / "lake" / f"{k}.parquet" for k in ("race", "quali", "practice")},
}


def available(source: str) -> bool:
    return all(p.exists() for p in PATHS[source].values())


def load(source: str = "audit") -> dict[str, pl.DataFrame]:
    if source not in PATHS:
        raise ValueError(f"unknown source {source!r}; use one of {list(PATHS)}")
    if not available(source):
        missing = [str(p) for p in PATHS[source].values() if not p.exists()]
        hint = ("python -m tyretrace.data.bulk" if source == "lake"
                else "python -m tyretrace run pull")
        raise FileNotFoundError(f"{source} data missing: {missing}. Run `{hint}`.")
    return {k: pl.read_parquet(p) for k, p in PATHS[source].items()}
