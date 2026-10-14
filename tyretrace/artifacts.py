"""Model outputs the dashboard draws from, beside results.json.

results.json carries headline numbers -- one scalar per metric, enough for a
gate board. A plot needs more: per-lap predictive intervals, a calibration
curve at many levels, posterior draws. Those land here as parquet tables, one
directory per task:

    out/artifacts/<task>/<name>.parquet

Same rules as results.py. Writes are atomic, so a crashed audit never leaves a
half-written table for the dashboard to draw. A missing artifact is None, not
an exception -- the dashboard says which task to run instead of crashing.

The split of responsibilities is the point: audits fit and write, the
dashboard only reads. A dashboard that fits models is slow to open and, worse,
can show numbers no audit ever gated.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from tyretrace.eval import metrics

ARTIFACTS = Path(__file__).resolve().parents[1] / "out" / "artifacts"

# predictive quantiles kept per lap -- enough for 50% and 90% bands and a median
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
# calibration is reported as a whole curve, never one level (protocol section 3)
LEVELS = tuple(round(float(x), 2) for x in np.arange(0.05, 1.0, 0.05))


def qcol(q: float) -> str:
    """Column name for a quantile: 0.05 -> 'q05', 0.5 -> 'q50'."""
    return f"q{int(round(q * 100)):02d}"


def path(task: str, name: str, root: Path = ARTIFACTS) -> Path:
    return root / task / f"{name}.parquet"


def save(task: str, name: str, df: pl.DataFrame, root: Path = ARTIFACTS) -> Path:
    out = path(task, name, root)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(out.parent), suffix=".tmp")
    os.close(fd)
    try:
        df.write_parquet(tmp)
        os.replace(tmp, out)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return out


def load(task: str, name: str, root: Path = ARTIFACTS) -> pl.DataFrame | None:
    p = path(task, name, root)
    if not p.exists():
        return None
    try:
        return pl.read_parquet(p)
    except Exception:
        # a corrupt artifact must not take the dashboard down with it
        return None


# --------------------------------------------------------------------------
# table builders -- shared so every audit writes the same shapes
# --------------------------------------------------------------------------

def predictive(draws: np.ndarray, y: np.ndarray, **cols) -> pl.DataFrame:
    """One row per held-out lap: observed, predictive quantiles, CRPS.

    draws: (n_draws, n_obs). `cols` are per-lap labels (circuit, driver, ...)
    of length n_obs, carried through so the dashboard can slice by them.
    """
    draws = np.asarray(draws, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    qs = np.quantile(draws, QUANTILES, axis=0)
    data = {k: np.asarray(v) for k, v in cols.items()}
    data["y"] = y
    data["mean"] = draws.mean(axis=0)
    for q, row in zip(QUANTILES, qs):
        data[qcol(q)] = row
    data["crps"] = metrics.crps_ensemble(draws, y)
    return pl.DataFrame(data)


def calibration(draws: np.ndarray, y: np.ndarray,
                levels=LEVELS) -> pl.DataFrame:
    """Empirical coverage of the central interval at each nominal level."""
    cov = metrics.calibration_curve(draws, y, levels)
    return pl.DataFrame({"level": list(cov), "coverage": list(cov.values())})


def posterior(samples: dict, names, *, max_draws: int = 2000,
              derived: dict | None = None) -> pl.DataFrame:
    """Scalar posterior draws, one column per parameter, thinned evenly.

    Names absent from `samples` are skipped, so one call works for every
    rung. `derived` adds columns computed from the draws (e.g. fuel pace in
    s/lap from beta_f), already aligned to the same draws.
    """
    cols = {n: np.asarray(samples[n]).reshape(-1) for n in names
            if n in samples and np.asarray(samples[n]).size
            == np.asarray(samples[n]).shape[0]}
    if derived:
        cols.update({k: np.asarray(v).reshape(-1) for k, v in derived.items()})
    if not cols:
        return pl.DataFrame()
    n = min(len(v) for v in cols.values())
    keep = np.unique(np.linspace(0, n - 1, min(n, max_draws)).astype(int))
    return pl.DataFrame({k: v[keep] for k, v in cols.items()})
