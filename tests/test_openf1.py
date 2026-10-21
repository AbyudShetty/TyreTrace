"""OpenF1 adapter -- track status rebuilt from race-control messages.

No network: these test the pure reconstruction logic on hand-written
messages shaped like the real API (see the 2023 Baku FP1 diagnosis in
tyretrace/data/openf1.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tyretrace.data import openf1 as O  # noqa: E402
from tyretrace.data import sources  # noqa: E402


def msg(t, flag=None, sector=None, lap=None, text=""):
    return {"date": f"2023-04-28T09:{t}+00:00", "flag": flag, "sector": sector,
            "lap_number": lap, "message": text, "category": "Flag" if flag else "Other"}


def test_practice_yellow_without_lap_number_is_found_by_time():
    """The bug: practice messages carry no lap number, so a lap-only reading
    marked every practice lap green."""
    rc = [msg("33:24", "YELLOW", 2), msg("33:27", "CLEAR", 2)]
    w = O._status_intervals(rc)
    t0 = O._ts("2023-04-28T09:33:00+00:00")
    assert O._track_status(5, t0, t0 + 90, {}, w) == "2"           # overlaps the yellow
    assert O._track_status(6, t0 + 120, t0 + 210, {}, w) == "1"    # after the CLEAR


def test_yellow_ends_at_its_own_sectors_clear_not_another_sectors():
    rc = [msg("40:31", "YELLOW", 9), msg("40:40", "CLEAR", 8), msg("41:30", "CLEAR", 9)]
    (w0, w1, code), = O._status_intervals(rc)
    assert code == "2" and w1 - w0 == pytest.approx(59.0)


def test_unclosed_yellow_is_capped_not_infinite():
    (w0, w1, _), = O._status_intervals([msg("10:00", "DOUBLE YELLOW", 4)])
    assert w1 - w0 == O.YELLOW_MAX_S


def test_red_flag_runs_until_green():
    rc = [msg("20:00", "RED"), msg("35:00", "GREEN")]
    (w0, w1, code), = O._status_intervals(rc)
    assert code == "5" and w1 - w0 == pytest.approx(900.0)


def test_race_safety_car_by_lap_number():
    rc = [msg("22:24", lap=11, text="SAFETY CAR DEPLOYED"),
          msg("28:42", lap=13, text="SAFETY CAR IN THIS LAP")]
    by_lap = O._lap_status(rc)
    assert {k: v for k, v in by_lap.items()} == {11: "4", 12: "4", 13: "4"}
    nan = float("nan")
    assert O._track_status(12, nan, nan, by_lap, []) == "4"
    assert O._track_status(14, nan, nan, by_lap, []) == "1"


def test_virtual_safety_car_is_not_mistaken_for_a_safety_car():
    rc = [msg("01:00", lap=30, text="VIRTUAL SAFETY CAR DEPLOYED"),
          msg("02:00", lap=31, text="VIRTUAL SAFETY CAR ENDING")]
    assert O._lap_status(rc) == {30: "6", 31: "6"}


def test_sources_reject_unknown_and_explain_missing():
    with pytest.raises(ValueError, match="unknown source"):
        sources.load("nope")


def test_race_stint_start_snaps_to_the_pit_out_lap():
    """2025 Montreal: OpenF1 started a second stint on the in-lap."""
    ss = [{"stint_number": 1, "lap_start": 1, "lap_end": 14},
          {"stint_number": 2, "lap_start": 15, "lap_end": 70}]
    O.align_stints(ss, 44, {(44, 16)})
    assert ss[1]["lap_start"] == 16 and ss[0]["lap_end"] == 15


def test_correct_or_unsupported_boundaries_are_left_alone():
    ok = [{"stint_number": 1, "lap_start": 1, "lap_end": 15},
          {"stint_number": 2, "lap_start": 16, "lap_end": 70}]
    O.align_stints(ok, 1, {(1, 16)})
    assert ok[1]["lap_start"] == 16
    far = [{"stint_number": 1, "lap_start": 1, "lap_end": 15},
           {"stint_number": 2, "lap_start": 16, "lap_end": 70}]
    O.align_stints(far, 1, {(1, 20)})                 # no pit-out within one lap
    assert far[1]["lap_start"] == 16
