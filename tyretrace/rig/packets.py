"""F1 game UDP telemetry decoder -- the ground-truth rig.

WHY THIS IS THE MOST VALUABLE THING IN THE PROJECT
--------------------------------------------------
The stream can carry fuel mass in kg and per-wheel tyre wear for multiple
cars. Privacy settings may redact those channels; the capture report measures
how many cars actually expose both instead of assuming twenty. It is
also the only place a tyre cliff is observable on demand (practice runs get
boxed before it, races rarely go deep enough), and the only place the run
planner's counterfactual can actually be executed.

DESIGN NOTE -- WHY THIS IS BUILT DEFENSIVELY
--------------------------------------------
These layouts are written from the published packet spec, not verified against
a running game on this machine. A wrong offset does not crash: it silently
yields plausible-looking garbage, which is the worst possible failure for a
ground-truth harness. Three defences:

  1. Per-car stride is DERIVED from the packet length, never assumed, so a
     format year that adds trailing fields still decodes.
  2. Version-specific lap offsets. F1 24 added minute fields in the middle
     of lap data; a stride change alone cannot keep F1 23 aligned.
  3. `validate()` checks physical plausibility (fuel 0-150 kg, wear 0-100%,
     lap numbers sane). Run `python -m tyretrace.rig.record --self-check`
     against a live session BEFORE trusting a single recorded lap.

If the self-check fails, the offsets are wrong for that game build. Fix the
table, do not loosen the check.
"""
from __future__ import annotations

import struct
import math
from dataclasses import dataclass

# --- header ---------------------------------------------------------------
# F1 23 introduced m_gameYear, giving a 29-byte header that F1 24 and F1 25
# both keep. F1 22 and earlier used 24 bytes and are not supported.
HEADER_FMT = "<HBBBBBQfIIBB"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 29, f"header should be 29 bytes, got {HEADER_SIZE}"

SUPPORTED_FORMATS = (2023, 2024, 2025)
MAX_CARS = 22

PACKET_LAP_DATA = 2
PACKET_SESSION = 1
PACKET_CAR_STATUS = 7
PACKET_CAR_DAMAGE = 10

# --- per-car field offsets ------------------------------------------------
# Offsets are from the start of each car's block. Only leading fields are
# listed; everything after the last named field is deliberately not read.

CAR_STATUS_OFFSETS = {
    "fuel_in_tank_kg":    (5,  "f"),   # <- the ground truth we came for
    "fuel_capacity_kg":   (9,  "f"),
    "fuel_remaining_laps": (13, "f"),
    "actual_compound":    (25, "B"),   # 16=C5 17=C4 18=C3 19=C2 20=C1 21=C0
    "visual_compound":    (26, "B"),
    "tyres_age_laps":     (27, "B"),
}

CAR_DAMAGE_OFFSETS = {
    # tyresWear[4] as float percent, RL RR FL FR in the game's wheel order
    "wear_rl": (0,  "f"),
    "wear_rr": (4,  "f"),
    "wear_fl": (8,  "f"),
    "wear_fr": (12, "f"),
}

LAP_DATA_OFFSETS = {
    "last_lap_time_ms":    (0,  "I"),
    "current_lap_time_ms": (4,  "I"),
    "lap_distance_m":      (20, "f"),
    "total_distance_m":    (24, "f"),
    "car_position":        (32, "B"),
    "current_lap_num":     (33, "B"),
    "pit_status":          (34, "B"),   # 0 none, 1 pitting, 2 in pit area
    "num_pit_stops":       (35, "B"),
    "current_lap_invalid": (37, "B"),
    "driver_status":       (44, "B"),   # 0 in garage, 1 flying lap, 2 in lap...
}

# EA's 2023 layout lacks two delta-minute fields in the MIDDLE of the block.
# Deriving stride alone cannot repair shifted offsets. See docs/GAME_RIG.md.
LAP_DATA_OFFSETS_BY_FORMAT = {
    2023: {name: (offset - 2 if offset >= 20 else offset, fmt)
           for name, (offset, fmt) in LAP_DATA_OFFSETS.items()},
    2024: LAP_DATA_OFFSETS,
    2025: LAP_DATA_OFFSETS,
}

# Actual tyre compound ids -> Pirelli construction, so the rig speaks the same
# language as the Pirelli C1-C5 map used for real-data pooling.
COMPOUND_BY_ID = {
    16: "C5", 17: "C4", 18: "C3", 19: "C2", 20: "C1", 21: "C0", 22: "C6",
    7: "INT", 8: "WET",
}
VISUAL_BY_ID = {16: 'SOFT', 17: 'MEDIUM', 18: 'HARD', 7: 'INTERMEDIATE', 8: 'WET'}


@dataclass(frozen=True)
class Header:
    packet_format: int
    game_year: int
    packet_id: int
    session_uid: int
    session_time: float
    frame: int
    player_car_index: int
    packet_version: int = 1


class PacketError(Exception):
    """Raised when a packet cannot be decoded safely. Never swallowed."""


def parse_header(buf: bytes) -> Header:
    if len(buf) < HEADER_SIZE:
        raise PacketError(f"packet too short for header: {len(buf)} bytes")
    (fmt, year, _maj, _min, _pv, pid, uid, stime, frame, _overall,
     player, _second) = struct.unpack_from(HEADER_FMT, buf, 0)
    if fmt not in SUPPORTED_FORMATS:
        raise PacketError(
            f"packet format {fmt} not supported (expected one of "
            f"{SUPPORTED_FORMATS}); F1 22 and earlier use a 24-byte header")
    if not math.isfinite(stime) or stime < 0:
        raise PacketError("invalid session clock")
    if _pv != 1:
        raise PacketError(f"unsupported packet version {_pv}")
    return Header(fmt, year, pid, uid, stime, frame, player, _pv)


def car_stride(buf: bytes) -> int:
    """Bytes per car, derived from the packet length rather than assumed.

    A format year that appends fields to each car block changes the stride but
    not the leading offsets, so deriving it here is what makes the decoder
    survive a game update.
    """
    hdr = parse_header(buf)
    trailing = 2 if hdr.packet_id == PACKET_LAP_DATA else 0
    body = len(buf) - HEADER_SIZE - trailing
    if body <= 0:
        raise PacketError("packet has no body")
    stride = body // MAX_CARS
    if stride <= 0:
        raise PacketError(f"implausible stride from {body}-byte body")
    if body % MAX_CARS:
        raise PacketError(
            f"body {body} does not look like {MAX_CARS} car blocks "
            f"(unexpected partial block of {body % MAX_CARS} bytes)")
    return stride


def _decode(buf: bytes, offsets: dict, stride: int) -> list[dict]:
    need = max(off + struct.calcsize(f) for off, f in offsets.values())
    if stride < need:
        raise PacketError(
            f"car stride {stride} is smaller than the {need} bytes the offset "
            "table needs -- layout does not match this game build")

    out = []
    for i in range(MAX_CARS):
        base = HEADER_SIZE + i * stride
        row = {}
        for name, (off, fmt) in offsets.items():
            row[name] = struct.unpack_from("<" + fmt, buf, base + off)[0]
        out.append(row)
    return out


def decode_car_status(buf: bytes) -> list[dict]:
    rows = _decode(buf, CAR_STATUS_OFFSETS, car_stride(buf))
    for r in rows:
        r["compound"] = COMPOUND_BY_ID.get(r["actual_compound"], "UNKNOWN")
        r["compound_rel"] = VISUAL_BY_ID.get(r["visual_compound"], "UNKNOWN")
    return rows


def decode_car_damage(buf: bytes) -> list[dict]:
    return _decode(buf, CAR_DAMAGE_OFFSETS, car_stride(buf))


def decode_lap_data(buf: bytes) -> list[dict]:
    hdr = parse_header(buf)
    return _decode(buf, LAP_DATA_OFFSETS_BY_FORMAT[hdr.packet_format], car_stride(buf))


DECODERS = {
    PACKET_CAR_STATUS: decode_car_status,
    PACKET_CAR_DAMAGE: decode_car_damage,
    PACKET_LAP_DATA: decode_lap_data,
}


def validate(kind: int, rows: list[dict]) -> list[str]:
    """Physical plausibility. An offset error shows up here, not as a crash.

    Returns a list of problems; empty means the layout looks right. Only cars
    that are actually running are checked -- empty slots read as zeros.
    """
    problems: list[str] = []

    if kind == PACKET_CAR_STATUS:
        fuel = [r["fuel_in_tank_kg"] for r in rows]
        live = [f for f in fuel if f > 0.01]
        if not live:
            problems.append("no car reports fuel above zero")
        if any(not (0.0 <= f <= 150.0) for f in fuel):
            problems.append(f"fuel outside 0-150 kg: {max(fuel):.1f}")
        caps = [r["fuel_capacity_kg"] for r in rows if r["fuel_capacity_kg"] > 0]
        if any(not math.isfinite(r['fuel_capacity_kg']) for r in rows):
            problems.append('non-finite fuel capacity')
        if caps and not all(50.0 <= c <= 150.0 for c in caps):
            problems.append(f"fuel capacity implausible: {caps[:3]}")
        ages = [r["tyres_age_laps"] for r in rows]
        if any(a > 200 for a in ages):
            problems.append(f"tyre age above 200 laps: {max(ages)}")
        if any(r['fuel_in_tank_kg']>.01 and r.get('compound')=='UNKNOWN' for r in rows):
            problems.append('active car reports unsupported tyre construction')

    elif kind == PACKET_CAR_DAMAGE:
        wear = [r[k] for r in rows for k in ("wear_fl", "wear_fr", "wear_rl", "wear_rr")]
        if any(not (0.0 <= w <= 100.0) for w in wear):
            problems.append(f"tyre wear outside 0-100%: min {min(wear):.1f} "
                            f"max {max(wear):.1f}")

    elif kind == PACKET_LAP_DATA:
        laps = [r["current_lap_num"] for r in rows]
        if any(l > 200 for l in laps):
            problems.append(f"lap number above 200: {max(laps)}")
        if any(r["pit_status"] > 2 for r in rows):
            problems.append("pit_status outside 0-2")
        dist = [r["lap_distance_m"] for r in rows]
        if any(not math.isfinite(d) or d < -2000 or d > 30000 for d in dist):
            problems.append(f"lap distance implausible: {min(dist):.0f} to "
                            f"{max(dist):.0f} m")
        if any(r['current_lap_invalid'] not in (0,1) for r in rows):
            problems.append('current_lap_invalid outside 0-1')
        if any(r['driver_status'] > 4 for r in rows):
            problems.append('driver_status outside 0-4')

    return problems
