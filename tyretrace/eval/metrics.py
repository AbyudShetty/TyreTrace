"""Scoring, as pre-registered in eval/protocol.md §3.

CRPS is primary because it scores the whole predictive distribution: point
accuracy alone cannot tell a confident wrong answer from an honest uncertain
one, and that distinction is the entire claim of this project.
"""
from __future__ import annotations

import numpy as np


def crps_ensemble(draws: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS per observation from posterior-predictive draws.

    Uses the energy form  E|X - y| - 0.5 * E|X - X'|,  computed exactly from
    the sorted sample rather than by resampling pairs:

        E|X - X'| = 2/m^2 * sum_i (2i - m + 1) * x_(i)

    draws: (n_draws, n_obs)   y: (n_obs,)   returns: (n_obs,)
    """
    draws = np.asarray(draws, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if draws.ndim != 2 or draws.shape[1] != y.shape[0]:
        raise ValueError(f"draws {draws.shape} incompatible with y {y.shape}")

    m = draws.shape[0]
    term1 = np.abs(draws - y[None, :]).mean(axis=0)

    s = np.sort(draws, axis=0)
    i = np.arange(1, m + 1, dtype=np.float64)[:, None]
    term2 = (2.0 * (2.0 * i - m - 1.0) * s).sum(axis=0) / (m * m)

    return term1 - 0.5 * term2


def coverage(draws: np.ndarray, y: np.ndarray, level: float = 0.90) -> float:
    """Fraction of observations inside the central `level` predictive interval."""
    lo = np.quantile(draws, (1 - level) / 2, axis=0)
    hi = np.quantile(draws, 1 - (1 - level) / 2, axis=0)
    return float(np.mean((y >= lo) & (y <= hi)))


def calibration_curve(draws: np.ndarray, y: np.ndarray,
                      levels=(0.5, 0.8, 0.9, 0.95)) -> dict[float, float]:
    """Report the whole curve, never a single level -- protocol §3."""
    return {lv: coverage(draws, y, lv) for lv in levels}


def mae(draws: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(np.median(draws, axis=0) - y)))


def score(draws: np.ndarray, y: np.ndarray) -> dict:
    """The full pre-registered metric set for one prediction task."""
    return {
        "crps": float(np.mean(crps_ensemble(draws, y))),
        "mae": mae(draws, y),
        "coverage_90": coverage(draws, y, 0.90),
        "calibration": calibration_curve(draws, y),
    }


def passes_ship_gate(new: dict, base: dict | None,
                     band: tuple[float, float] = (0.85, 0.95)) -> tuple[bool, str]:
    """Protocol §5: a rung ships only if CRPS improves AND coverage holds.

    A rung that sharpens the point estimate while breaking calibration is a
    regression, not an improvement.
    """
    cov = new["coverage_90"]
    if not (band[0] <= cov <= band[1]):
        return False, f"coverage {cov:.3f} outside {band}"
    if base is None:
        return True, "baseline rung, calibration in band"
    if new["crps"] >= base["crps"]:
        return False, f"CRPS {new['crps']:.5f} not better than {base['crps']:.5f}"
    gain = 100 * (1 - new["crps"] / base["crps"])
    return True, f"CRPS improved {gain:.1f}%, coverage {cov:.3f}"
