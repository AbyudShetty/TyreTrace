"""Spec-conformant fake UDP packets, so the rig is testable without the game.

The decoder in packets.py cannot be verified against a live game on every
machine, and an offset error there produces plausible garbage rather than a
crash. This module builds packets at the documented byte layout and drives a
simple physical model through them, so the decode path, the recorder and the
canonical-schema emit all get exercised in CI.

It proves the decoder is self-consistent. It does NOT prove the offsets match
a real game build -- only `record.py --self-check` against a live session can
do that.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from tyretrace.rig import packets as P


def _header(packet_id: int, session_uid: int, session_time: float,
            frame: int, fmt: int = 2024) -> bytes:
    return struct.pack(
        P.HEADER_FMT,
        fmt,            # m_packetFormat
        fmt % 100,      # m_gameYear
        1, 0,           # major, minor
        1,              # packet version
        packet_id,
        session_uid,
        session_time,
        frame, frame,   # frame, overall frame
        0, 255,         # player index, secondary player index
    )


# Per-car block sizes at the documented layout. The decoder derives stride
# from packet length rather than trusting these, but the fake has to pick
# something, so it picks the real ones.
STRIDE_CAR_STATUS = 55
STRIDE_CAR_DAMAGE = 42
STRIDE_LAP_DATA = 52


def _pack_block(stride: int, fields: dict, offsets: dict) -> bytearray:
    blk = bytearray(stride)
    for name, value in fields.items():
        off, fmt = offsets[name]
        struct.pack_into("<" + fmt, blk, off, value)
    return blk


def build_car_status(cars: list[dict], **hdr) -> bytes:
    buf = bytearray(_header(P.PACKET_CAR_STATUS, **hdr))
    for i in range(P.MAX_CARS):
        c = cars[i] if i < len(cars) else {}
        buf += _pack_block(STRIDE_CAR_STATUS, {
            "fuel_in_tank_kg": float(c.get("fuel", 0.0)),
            "fuel_capacity_kg": float(c.get("capacity", 110.0)),
            "fuel_remaining_laps": float(c.get("fuel_laps", 0.0)),
            "actual_compound": int(c.get("compound_id", 18)),
            "visual_compound": int(c.get("compound_id", 18)),
            "tyres_age_laps": int(c.get("age", 0)),
        }, P.CAR_STATUS_OFFSETS)
    return bytes(buf)


def build_car_damage(cars: list[dict], **hdr) -> bytes:
    buf = bytearray(_header(P.PACKET_CAR_DAMAGE, **hdr))
    for i in range(P.MAX_CARS):
        c = cars[i] if i < len(cars) else {}
        w = float(c.get("wear", 0.0))
        buf += _pack_block(STRIDE_CAR_DAMAGE, {
            "wear_rl": w, "wear_rr": w,
            "wear_fl": w * 0.9, "wear_fr": w * 0.9,
        }, P.CAR_DAMAGE_OFFSETS)
    return bytes(buf)


def build_lap_data(cars: list[dict], **hdr) -> bytes:
    buf = bytearray(_header(P.PACKET_LAP_DATA, **hdr))
    for i in range(P.MAX_CARS):
        c = cars[i] if i < len(cars) else {}
        buf += _pack_block(STRIDE_LAP_DATA, {
            "last_lap_time_ms": int(c.get("last_lap_ms", 0)),
            "current_lap_time_ms": int(c.get("cur_lap_ms", 0)),
            "lap_distance_m": float(c.get("lap_distance", 0.0)),
            "total_distance_m": float(c.get("total_distance", 0.0)),
            "car_position": int(c.get("position", i + 1)),
            "current_lap_num": int(c.get("lap_num", 1)),
            "pit_status": int(c.get("pit_status", 0)),
            "num_pit_stops": int(c.get("n_stops", 0)),
            "current_lap_invalid": 0,
            "driver_status": int(c.get("driver_status", 1)),
        }, P.LAP_DATA_OFFSETS)
    # LapData carries two trailing bytes after the car blocks
    buf += struct.pack("<BB", 255, 255)
    return bytes(buf)


@dataclass
class FakeSession:
    """A tiny physical model driving the fake stream.

    Fuel burns linearly, wear accumulates with tyre age, and lap time responds
    to both -- the same structure the estimator has to recover. Cars pit when
    wear crosses a threshold, which resets age but not fuel: the identifying
    discontinuity, present in the fake stream exactly as in a real race.
    """
    n_cars: int = 20
    n_laps: int = 30
    beta_f: float = 0.033
    kappa: float = 1.75
    deg_per_lap: float = 0.055
    wear_per_lap: float = 2.6
    base_lap_s: float = 90.0
    fuel_start: float = 100.0
    pit_wear_pct: float = 55.0
    seed: int = 0

    session_uid: int = 0xF1F1F1F1
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def stream(self):
        """Yield (packet_bytes, ...) triples lap by lap, as the game would."""
        fuel = np.full(self.n_cars, self.fuel_start)
        age = np.zeros(self.n_cars, dtype=int)
        wear = np.zeros(self.n_cars)
        stops = np.zeros(self.n_cars, dtype=int)
        offs = self._rng.normal(0, 0.25, self.n_cars)
        clock, frame = 0.0, 0

        for lap in range(1, self.n_laps + 1):
            pitting = wear > self.pit_wear_pct
            age = np.where(pitting, 0, age)
            wear = np.where(pitting, 0.0, wear)
            stops += pitting.astype(int)

            age += 1
            wear += self.wear_per_lap
            fuel = np.maximum(fuel - self.kappa, 0.0)

            lap_s = (self.base_lap_s + offs
                     + self.beta_f * fuel
                     + self.deg_per_lap * age
                     + self._rng.normal(0, 0.15, self.n_cars))
            clock += float(lap_s.mean())
            frame += 1

            cars_status = [{"fuel": fuel[i], "age": int(age[i]),
                            "compound_id": 18,
                            "fuel_laps": fuel[i] / self.kappa}
                           for i in range(self.n_cars)]
            cars_damage = [{"wear": wear[i]} for i in range(self.n_cars)]
            cars_lap = [{"lap_num": lap,
                         "last_lap_ms": int(lap_s[i] * 1000),
                         "pit_status": 1 if pitting[i] else 0,
                         "n_stops": int(stops[i]),
                         "lap_distance": 0.0,
                         "total_distance": float(lap * 5000),
                         "driver_status": 1}
                        for i in range(self.n_cars)]

            hdr = dict(session_uid=self.session_uid, session_time=clock, frame=frame)
            yield (build_lap_data(cars_lap, **hdr),
                   build_car_status(cars_status, **hdr),
                   build_car_damage(cars_damage, **hdr))
