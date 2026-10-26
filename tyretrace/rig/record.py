"""Listen to the game's UDP stream and emit canonical laps with ground truth.

The recorder keeps the latest fuel / wear / age state per car and commits one
canonical row the moment a car's lap number ticks over. Rows carry the same
37-column schema as FastF1 ingest and the simulator, with the truth_* columns
filled -- so the estimator sees ONLY the real-world-observable channels, and
the check is whether it recovers a hidden state it was never shown.

Usage
-----
  # verify the byte layout against a live session BEFORE trusting anything
  python -m tyretrace.rig.record --self-check

  # record a session to parquet
  python -m tyretrace.rig.record --out out/rig_session.parquet

  # exercise the whole path with no game running
  python -m tyretrace.rig.record --fake --out out/rig_fake.parquet

Game setup: Settings -> Telemetry -> UDP Telemetry ON, format 2024 (or 2023 /
2025), IP = this machine, port 20777, rate 20 Hz or higher. Turn on "UDP Send
All Cars" if the build exposes it -- without it only the player's car reports
and the twenty-cars-per-session advantage is lost.
"""
from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

import polars as pl

from tyretrace import schema
from tyretrace.rig import packets as P

DEFAULT_PORT = 20777
BUF = 4096


class LapAccumulator:
    """Latest state per car; commits a row when the lap number advances."""

    def __init__(self, session_uid: str, circuit: str = "RigCircuit",
                 year: int = 2024, rnd: int = 0):
        self.uid = session_uid
        self.circuit = circuit
        self.year = year
        self.rnd = rnd
        self.state: dict[int, dict] = {}
        self.rows: list[dict] = []
        self.stint: dict[int, int] = {}
        self.skipped_gaps = 0
        self.skipped_stale_truth = 0

    def _blank(self) -> dict:
        return {"lap_num": 0, "fuel": None, "age": None, "compound": "UNKNOWN",
                "wear": None, "wear_fl": None, "wear_fr": None,
                "wear_rl": None, "wear_rr": None, "pit": 0, "clock": 0.0,
                "n_stops": 0,'invalid':False,'status_clock':None,'wear_clock':None,
                'compound_rel':'UNKNOWN'}

    def update(self, kind: int, rows: list[dict], session_time: float) -> None:
        for idx, r in enumerate(rows):
            st = self.state.setdefault(idx, self._blank())
            st["clock"] = session_time

            if kind == P.PACKET_CAR_STATUS:
                if r["fuel_capacity_kg"] > 0.0:
                    st["fuel"] = float(r["fuel_in_tank_kg"])
                    st['status_clock']=session_time
                age = int(r["tyres_age_laps"])
                # A tyre-age RESET is the stint boundary, and it is the
                # physical signal rather than a reported one -- num_pit_stops
                # is not reliably incremented across game builds and session
                # types. This also keeps the schema's monotone-age contract.
                if st["age"] is not None and age < st["age"]:
                    self.stint[idx] = self.stint.get(idx, 0) + 1
                st["age"] = age
                st["compound"] = r["compound"]
                st['compound_rel']=r.get('compound_rel','UNKNOWN')

            elif kind == P.PACKET_CAR_DAMAGE:
                for w in ("fl", "fr", "rl", "rr"):
                    st[f"wear_{w}"] = float(r[f"wear_{w}"])
                st["wear"] = max(st[f"wear_{w}"] for w in ("fl", "fr", "rl", "rr"))
                st['wear_clock']=session_time

            elif kind == P.PACKET_LAP_DATA:
                lap = int(r["current_lap_num"])
                st["pit"] = int(r["pit_status"])
                # a completed lap: number advanced and we have a valid time
                if st['lap_num'] > 0 and lap > st['lap_num'] + 1:
                    self.skipped_gaps += 1
                elif st['lap_num'] > 0 and lap == st['lap_num'] + 1 and r["last_lap_time_ms"] > 0:
                    self._commit(idx, st, r)
                if lap>st['lap_num']:
                    st['invalid']=bool(r['current_lap_invalid'])
                else:
                    st['invalid'] |= bool(r['current_lap_invalid'])
                st["lap_num"] = lap

    def _commit(self, idx: int, st: dict, lap_row: dict) -> None:
        if st["fuel"] is None or st["age"] is None or st['wear'] is None:
            return                      # no status packet yet; skip the lap
        if (st['status_clock'] is None or st['wear_clock'] is None
                or st['clock']-st['status_clock']>2.0 or st['clock']-st['wear_clock']>2.0):
            self.skipped_stale_truth+=1
            return
        if st['compound']=='UNKNOWN':
            return

        sid = self.stint.get(idx, 0)
        age = st["age"]
        self.rows.append({
            "session_uid": self.uid, "year": self.year, "round": self.rnd,
            "circuit": self.circuit, "session_type": "RIG",
            "driver": f"C{idx:02d}", "team": f"T{idx // 2:02d}",
            "stint_id": sid,
            "lap_in_run": max(age, 1),
            # LastLapTime belongs to the lap that ENDED, not the current lap.
            "lap_in_session": int(lap_row["current_lap_num"]) - 1,
            "session_clock_s": float(st["clock"]),
            "tyre_age": age,
            "compound_rel": st['compound_rel'],
            "compound_abs": st["compound"],
            "is_fresh": age <= 1,
            "lap_time_s": lap_row["last_lap_time_ms"] / 1000.0,
            "sector_1_s": None, "sector_2_s": None, "sector_3_s": None,
            "fuel_load_kg_est": None, "fuel_load_kg_sd": None,
            "gap_ahead_s": None, "lift_coast_m": None,
            "track_temp_c": None, "air_temp_c": None,
            "track_status": "1",
            "pit_in": st["pit"] != 0, "pit_out": age <= 1,
            "is_valid": not st['invalid'],
            "corner_feats": None,
            # --- the whole point: truth the estimator never sees ------------
            "truth_fuel_kg": st["fuel"],
            "truth_wear_fl": st["wear_fl"], "truth_wear_fr": st["wear_fr"],
            "truth_wear_rl": st["wear_rl"], "truth_wear_rr": st["wear_rr"],
            "truth_deg_s": None, "truth_regime": None,
        })

    def frame(self) -> pl.DataFrame:
        if not self.rows:
            return schema.empty()
        df = pl.DataFrame(self.rows, schema_overrides=schema.LAPS_SCHEMA)
        # lap_in_run must restart at 1 per stint for the schema contract
        df = (df.sort("driver", "stint_id", "lap_in_session")
                .with_columns(
                    (pl.int_range(pl.len()).over(["driver", "stint_id"]) + 1)
                    .cast(pl.Int16).alias("lap_in_run")))
        return schema.validate(df, source="rig")


def self_check(port: int, n_packets: int = 400, timeout: float = 20.0) -> int:
    """Verify the byte layout against whatever is actually on the wire.

    This is the gate. Until it passes on the specific game build being used,
    no recorded lap should be trusted -- a wrong offset yields plausible
    numbers, not an error.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    sock.bind(("", port))
    print(f"listening on udp/{port} for {n_packets} packets ...")

    seen: dict[int, int] = {}
    problems: dict[int, set[str]] = {}
    fmt_seen: set[int] = set()

    try:
        for _ in range(n_packets):
            try:
                buf, _ = sock.recvfrom(BUF)
            except socket.timeout:
                print(f"\nno packets for {timeout:.0f}s.")
                print("check: UDP telemetry ON, correct IP/port, game unpaused.")
                return 2
            try:
                hdr = P.parse_header(buf)
            except P.PacketError as e:
                print(f"  header: {e}")
                continue
            fmt_seen.add(hdr.packet_format)
            if hdr.packet_id not in P.DECODERS:
                continue
            seen[hdr.packet_id] = seen.get(hdr.packet_id, 0) + 1
            try:
                rows = P.DECODERS[hdr.packet_id](buf)
            except P.PacketError as e:
                problems.setdefault(hdr.packet_id, set()).add(str(e))
                continue
            for p in P.validate(hdr.packet_id, rows):
                problems.setdefault(hdr.packet_id, set()).add(p)
    finally:
        sock.close()

    names = {P.PACKET_LAP_DATA: "lap_data",
             P.PACKET_CAR_STATUS: "car_status",
             P.PACKET_CAR_DAMAGE: "car_damage"}
    print(f"\npacket format(s) seen: {sorted(fmt_seen) or 'none'}")
    for pid, name in names.items():
        print(f"  {name:<12s} {seen.get(pid, 0):5d} packets")

    missing = [n for p, n in names.items() if not seen.get(p)]
    if missing:
        print(f"\nNEVER SAW: {', '.join(missing)}")
        print("enable all packet types in the game's telemetry settings.")
        return 2

    if problems:
        print("\nLAYOUT PROBLEMS -- do not record until these are fixed:")
        for pid, ps in problems.items():
            for p in sorted(ps):
                print(f"  {names[pid]}: {p}")
        print("\nthe offset table in tyretrace/rig/packets.py does not match")
        print("this game build. Fix the offsets; do not loosen validate().")
        return 1

    print("\nSELF-CHECK PASSED -- layout matches this build. Safe to record.")
    return 0


def record(port: int, out: Path, max_packets: int | None = None,
           timeout: float = 60.0) -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    sock.bind(("", port))
    acc: LapAccumulator | None = None
    n = 0
    print(f"recording from udp/{port} -> {out}   (ctrl-c to stop)")
    try:
        while max_packets is None or n < max_packets:
            try:
                buf, _ = sock.recvfrom(BUF)
            except socket.timeout:
                print(f"no packets for {timeout:.0f}s; stopping.")
                break
            n += 1
            try:
                hdr = P.parse_header(buf)
                if hdr.packet_id not in P.DECODERS:
                    continue
                rows = P.DECODERS[hdr.packet_id](buf)
            except P.PacketError:
                continue
            if acc is None:
                acc = LapAccumulator(f"rig_{hdr.session_uid:016x}")
            acc.update(hdr.packet_id, rows, hdr.session_time)
            if acc.rows and len(acc.rows) % 20 == 0:
                print(f"\r  {len(acc.rows)} laps captured", end="", flush=True)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        sock.close()

    if acc is None or not acc.rows:
        print("no laps captured.")
        return 1
    df = acc.frame()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    print(f"\nwrote {df.height} laps from {df['driver'].n_unique()} cars -> {out}")
    return 0


def record_fake(out: Path, laps: int = 30, cars: int = 20) -> int:
    """Drive the whole path with generated packets. No game, no socket."""
    from tyretrace.rig.fake import FakeSession

    fs = FakeSession(n_cars=cars, n_laps=laps)
    acc = LapAccumulator(f"rig_fake_{fs.session_uid:016x}")
    t = 0.0
    for lap_pkt, status_pkt, damage_pkt in fs.stream():
        hdr = P.parse_header(lap_pkt)
        t = hdr.session_time
        acc.update(P.PACKET_CAR_STATUS, P.decode_car_status(status_pkt), t)
        acc.update(P.PACKET_CAR_DAMAGE, P.decode_car_damage(damage_pkt), t)
        acc.update(P.PACKET_LAP_DATA, P.decode_lap_data(lap_pkt), t)

    df = acc.frame()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    print(f"wrote {df.height} laps from {df['driver'].n_unique()} cars -> {out}")
    print(f"truth fuel  {df['truth_fuel_kg'].min():.1f} - "
          f"{df['truth_fuel_kg'].max():.1f} kg")
    print(f"truth wear  {df['truth_wear_fl'].min():.1f} - "
          f"{df['truth_wear_fl'].max():.1f} %")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--self-check", action="store_true",
                    help="verify byte layout against a live session")
    ap.add_argument("--fake", action="store_true",
                    help="generate packets instead of listening")
    ap.add_argument("--out", type=Path, default=Path("out/rig_session.parquet"))
    ap.add_argument("--laps", type=int, default=30, help="fake mode only")
    args = ap.parse_args(argv)

    if args.self_check:
        return self_check(args.port)
    if args.fake:
        return record_fake(args.out, laps=args.laps)
    return record(args.port, args.out)


if __name__ == "__main__":
    sys.exit(main())
