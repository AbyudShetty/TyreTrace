"""Traffic feature tests -- gap to car ahead and the wake index.

Frames are built directly rather than via the simulator: these test the
ordering logic, and a hand-written frame makes the expected gaps obvious.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tyretrace.features import traffic  # noqa: E402


def _frame(clocks, lap=1, uid="2024_01_R"):
    n = len(clocks)
    return pl.DataFrame({
        "session_uid": [uid] * n,
        "driver": [f"D{i:02d}" for i in range(n)],
        "lap_in_session": [lap] * n,
        "session_clock_s": [float(c) for c in clocks],
    })


def test_gap_is_distance_to_the_car_directly_ahead():
    out = traffic.add_gap_ahead(_frame([10.0, 11.5, 14.0, 30.0]))
    gaps = out.sort("session_clock_s")["gap_ahead_s"].to_list()
    assert gaps[0] is None, "the leader has no car ahead"
    assert gaps[1] == 1.5
    assert gaps[2] == 2.5
    assert gaps[3] == 16.0


def test_gaps_are_computed_within_a_lap_not_across_laps():
    df = pl.concat([_frame([10.0, 12.0], lap=1), _frame([100.0, 103.0], lap=2)])
    out = traffic.add_gap_ahead(df).sort("lap_in_session", "session_clock_s")
    gaps = out["gap_ahead_s"].to_list()
    assert gaps == [None, 2.0, None, 3.0], "each lap restarts the ordering"


def test_implausible_gaps_become_null_not_zero():
    """A gap beyond ~30 s means nobody is meaningfully ahead. Recording it as
    0 would read as 'right behind a car', inverting the covariate."""
    out = traffic.add_gap_ahead(_frame([10.0, 100.0])).sort("session_clock_s")
    assert out["gap_ahead_s"].to_list()[1] is None


def test_wake_index_is_bounded_and_decays_with_distance():
    out = traffic.dirty_air_index(
        traffic.add_gap_ahead(_frame([10.0, 10.2, 12.0, 20.0]))
    ).sort("session_clock_s")
    da = out["dirty_air"].to_numpy()
    assert np.all((da >= 0) & (da <= 1))
    assert da[0] == 0.0, "clear air for the leader"
    assert da[1] > da[2] > da[3], "wake must decay with distance"


def test_null_gap_is_clear_air_not_maximum_wake():
    out = traffic.dirty_air_index(traffic.add_gap_ahead(_frame([10.0, 200.0])))
    assert out.sort("session_clock_s")["dirty_air"].to_list() == [0.0, 0.0]


def test_dropping_a_car_changes_the_gaps_behind_it():
    """The running order depends on every car on track, including ones whose
    laps a long-run filter would later trim -- so the feature has to be
    computed before filtering, not after."""
    full = traffic.add_gap_ahead(_frame([10.0, 12.0, 15.0]))
    partial = traffic.add_gap_ahead(_frame([10.0, 15.0]))

    g_full = dict(zip(full["driver"], full["gap_ahead_s"]))
    g_part = dict(zip(partial["driver"], partial["gap_ahead_s"]))
    # D02 sits 3 s behind D01 with the field intact, 5 s behind D00 without it
    assert g_full["D02"] == 3.0
    assert g_part["D01"] == 5.0
