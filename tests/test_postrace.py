"""Post-race decomposition: exact, and it blames the right assumption."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tyretrace.backtest import postrace as P  # noqa: E402

SLOPE, DEG, WAKE = -0.055, {"MEDIUM": 0.05, "HARD": 0.03}, 0.4


def race(seed=0, sigma=0.05, n_drv=8, pit=22, n_laps=50):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_drv):
        off = rng.normal(0, 0.3)
        for lap in range(2, n_laps + 1):
            first = lap <= pit
            comp, age, stint = ("MEDIUM", lap, 0) if first else ("HARD", lap - pit, 1)
            da = float(np.exp(-rng.uniform(0.3, 6) / 2)) if lap < 12 else 0.0
            y = 90 + off + SLOPE * lap + DEG[comp] * age + WAKE * da + rng.normal(0, sigma)
            rows.append((f"D{d}", stint, lap, age, comp, y, da))
    return pl.DataFrame(rows, schema=["driver", "stint_id", "lap_in_session", "tyre_age",
                                      "compound_rel", "lap_time_s", "dirty_air"], orient="row")


def test_parts_sum_exactly_to_the_error():
    t = P.decompose(race(), slope_fc=-0.04, deg_fc={"MEDIUM": 0.07, "HARD": 0.02})
    total = sum(t[p].to_numpy() for p in P.PARTS)
    assert np.allclose(total, t["error"].to_numpy())
    assert np.allclose(t["forecast"] - t["actual"], t["error"])


def test_a_wrong_slope_is_blamed_on_slope_not_tyres():
    t = P.decompose(race(sigma=0.01), slope_fc=SLOPE + 0.02, deg_fc=DEG)
    assert abs(t["slope"].mean()) > 5 * abs(t["tyres"].mean())
    assert t["slope"].mean() > 0                     # shallower slope: forecast too slow


def test_wrong_degradation_is_blamed_on_tyres_not_slope():
    t = P.decompose(race(sigma=0.01), slope_fc=SLOPE, deg_fc={"MEDIUM": 0.05, "HARD": 0.06})
    after = t.filter(pl.col("after_stop"))
    assert abs(after["tyres"].mean()) > 5 * abs(after["slope"].mean())


def test_traffic_at_the_anchor_shows_up_as_traffic():
    """Dirty air in the early anchor laps makes every later lap's forecast
    too slow -- the decomposition must attribute that to traffic."""
    t = P.decompose(race(sigma=0.01), slope_fc=SLOPE, deg_fc=DEG)
    assert t["traffic"].mean() > 0.05
    assert abs(t["slope"].mean()) < 0.01 and abs(t["tyres"].mean()) < 0.01


def test_identity_check_raises_if_the_algebra_breaks(monkeypatch):
    """The internal sum check is real, not decorative."""
    orig = P.post_race_fit

    def broken(df):
        out = orig(df)
        out["wake"] += 1.0          # parts no longer match the forecast
        return out

    monkeypatch.setattr(P, "post_race_fit", broken)
    with pytest.raises(AssertionError, match="does not sum"):
        P.decompose(race(), slope_fc=-0.04, deg_fc=DEG)
