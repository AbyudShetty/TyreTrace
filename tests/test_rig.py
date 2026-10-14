"""Rig tests -- decode path, defences, and end-to-end recovery.

These prove the decoder is SELF-CONSISTENT. They cannot prove the byte
offsets match a real game build; only `record.py --self-check` against a live
session does that, and the docstrings say so in both places.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit.a5_rig_recovery import build_rig_laps, estimate  # noqa: E402
from tyretrace import schema  # noqa: E402
from tyretrace.rig import fake, packets as P  # noqa: E402


def _hdr(pid, **kw):
    kw.setdefault("session_uid", 1)
    kw.setdefault("session_time", 0.0)
    kw.setdefault("frame", 1)
    return kw


def test_header_size_is_29_bytes():
    assert P.HEADER_SIZE == 29


def test_rejects_unsupported_packet_format():
    buf = bytearray(fake.build_car_status([], **_hdr(P.PACKET_CAR_STATUS)))
    struct.pack_into("<H", buf, 0, 2022)          # F1 22 header is 24 bytes
    with pytest.raises(P.PacketError, match="not supported"):
        P.parse_header(bytes(buf))


def test_rejects_truncated_packet():
    with pytest.raises(P.PacketError, match="too short"):
        P.parse_header(b"\x00" * 8)


def test_stride_is_derived_not_assumed():
    """A format year that appends fields must still decode."""
    pkt = fake.build_car_status([{"fuel": 50.0}], **_hdr(P.PACKET_CAR_STATUS))
    assert P.car_stride(pkt) == fake.STRIDE_CAR_STATUS

    # simulate a future year adding 6 bytes to every car block
    grown = bytearray(pkt[:P.HEADER_SIZE])
    for i in range(P.MAX_CARS):
        base = P.HEADER_SIZE + i * fake.STRIDE_CAR_STATUS
        grown += pkt[base:base + fake.STRIDE_CAR_STATUS] + b"\x00" * 6
    assert P.car_stride(bytes(grown)) == fake.STRIDE_CAR_STATUS + 6
    assert P.decode_car_status(bytes(grown))[0]["fuel_in_tank_kg"] == pytest.approx(50.0)


def test_rejects_stride_too_small_for_offsets():
    short = bytes(P.HEADER_SIZE + P.MAX_CARS * 8)
    short = fake.build_car_status([], **_hdr(P.PACKET_CAR_STATUS))[:P.HEADER_SIZE] \
        + bytes(P.MAX_CARS * 8)
    with pytest.raises(P.PacketError, match="smaller than"):
        P.decode_car_status(short)


def test_round_trip_fuel_wear_and_age():
    cars = [{"fuel": 87.5, "age": 12, "compound_id": 19}]
    st = P.decode_car_status(fake.build_car_status(cars, **_hdr(P.PACKET_CAR_STATUS)))
    assert st[0]["fuel_in_tank_kg"] == pytest.approx(87.5, abs=1e-3)
    assert st[0]["tyres_age_laps"] == 12
    assert st[0]["compound"] == "C2"

    dm = P.decode_car_damage(fake.build_car_damage([{"wear": 33.0}],
                                                   **_hdr(P.PACKET_CAR_DAMAGE)))
    assert dm[0]["wear_rl"] == pytest.approx(33.0, abs=1e-3)
    assert dm[0]["wear_fl"] == pytest.approx(29.7, abs=1e-3)


def test_validate_catches_implausible_values():
    """An offset error yields plausible-looking garbage, so validate() is the
    only thing standing between a wrong layout and a corrupt ground truth."""
    good = fake.build_car_status([{"fuel": 90.0}], **_hdr(P.PACKET_CAR_STATUS))
    assert P.validate(P.PACKET_CAR_STATUS, P.decode_car_status(good)) == []

    bad = bytearray(good)
    struct.pack_into("<f", bad, P.HEADER_SIZE + 5, 9999.0)      # absurd fuel
    problems = P.validate(P.PACKET_CAR_STATUS, P.decode_car_status(bytes(bad)))
    assert any("fuel outside" in p for p in problems)

    nofuel = fake.build_car_status([], **_hdr(P.PACKET_CAR_STATUS))
    assert any("no car reports fuel" in p
               for p in P.validate(P.PACKET_CAR_STATUS, P.decode_car_status(nofuel)))


def test_recorder_emits_canonical_schema_with_truth():
    laps = build_rig_laps(fake.FakeSession(n_cars=6, n_laps=12, seed=3))
    assert list(laps.columns) == list(schema.LAPS_SCHEMA)
    assert laps.height > 0
    assert laps["truth_fuel_kg"].null_count() == 0
    assert laps["truth_wear_fl"].null_count() == 0
    schema.validate(laps, source="test")


def test_recorder_splits_stints_on_tyre_age_reset():
    """Stint boundaries come from the physical age reset, not num_pit_stops,
    which is not reliably incremented across game builds."""
    laps = build_rig_laps(fake.FakeSession(n_cars=4, n_laps=40, seed=5))
    stints = laps.select(["driver", "stint_id"]).unique().height
    assert stints > laps["driver"].n_unique(), "expected at least one pit stop"
    # the schema contract already forbids age decreasing inside a stint
    schema.validate(laps, source="test")


def test_rig_recovers_hidden_state():
    """The Tier 2 gate: strip every truth column, recover what was hidden."""
    fs = fake.FakeSession(n_cars=20, n_laps=40, seed=11)
    laps = build_rig_laps(fs)
    observable = laps.drop([c for c in laps.columns if c.startswith("truth_")])
    assert not any(c.startswith("truth_") for c in observable.columns)

    est = estimate(observable)
    m, se = est["deg"]
    assert abs(m - fs.deg_per_lap) <= 3 * se
    m, se = est["lap_coef"]
    assert abs(-m - fs.beta_f * fs.kappa) <= 3 * se
