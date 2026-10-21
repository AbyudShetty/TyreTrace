"""OpenF1 -> canonical lap table. The data lake's source from 2023 onward.

WHY THIS EXISTS
---------------
FastF1 reads F1's live-timing archive (livetiming.formula1.com/static),
which began answering 401 Unauthorized in 2026. The ten weekends pulled in
August are still in the FastF1 cache; nothing new can be fetched through it.
OpenF1 (api.openf1.org) serves the same timing for 2023 onward, free.

This module emits exactly tyretrace.schema.LAPS_SCHEMA, so no model can tell
which source a lap came from -- and audit/d1_source_parity.py checks that
claim on every session both sources have, lap by lap.

WHERE THE SOURCES DIFFER
------------------------
- track_status is rebuilt from race-control messages: safety car "4" and
  virtual safety car "6" by lap number; yellow "2" (until that sector's
  CLEAR) and red "5" (until green) by TIME, since practice and qualifying
  messages carry no lap number. Track-wide, not per sector. FastF1 reads a
  status feed directly; audit/d1_source_parity.py measures the difference.
- is_valid has no OpenF1 equivalent and is set True. It is a flag, never a
  filter, so nothing downstream depends on it.
- weather is not fetched.

Requests are throttled below OpenF1's free-tier limit and every response is
cached on disk, so a re-run is free and an interrupted pull resumes.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl

from tyretrace import schema

API = "https://api.openf1.org/v1"
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache" / "openf1"
MIN_INTERVAL_S = 2.1          # ~28 requests / minute, under the free-tier cap

CODES = {"Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3",
         "Qualifying": "Q", "Race": "R"}
_REL = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}

_last = [0.0]


def get(endpoint: str, **params) -> list[dict]:
    """GET with an on-disk cache, a throttle, and retry on 429 / 5xx."""
    q = urllib.parse.urlencode(sorted(params.items()))
    url = f"{API}/{endpoint}?{q}"
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{endpoint}_{hashlib.sha1(q.encode()).hexdigest()[:16]}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))

    for attempt in range(6):
        wait = MIN_INTERVAL_S - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:                       # OpenF1: "no results"
                data = []
                break
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(min(60, 5 * 2 ** attempt))
    else:
        raise RuntimeError(f"OpenF1 unreachable after retries: {url}")
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(f)
    return data


def _ts(s: str | None) -> float:
    return dt.datetime.fromisoformat(s).timestamp() if s else float("nan")


# --------------------------------------------------------------------------
# sessions -> official round numbers
# --------------------------------------------------------------------------

def sessions(year: int, before: dt.datetime) -> list[dict]:
    """Every pullable session in a year, keyed by the OFFICIAL round number.

    Round numbers come from FastF1's event schedule (served by Ergast, which
    still answers), matched to OpenF1 meetings by race date -- so a uid like
    2023_04_R means the same weekend in both sources.
    """
    import fastf1

    sched = fastf1.get_event_schedule(year, include_testing=False)
    rounds = [(int(r["RoundNumber"]), r["EventDate"].to_pydatetime().date(), str(r["Location"]))
              for _, r in sched.iterrows()]
    out = []
    for s in get("sessions", year=year):
        code = CODES.get(s.get("session_name"))
        if not code or s.get("is_cancelled"):
            continue
        start = dt.datetime.fromisoformat(s["date_start"])
        if start.replace(tzinfo=None) >= before:
            continue
        # the session belongs to the event whose race date is within 4 days after it
        match = [r for r in rounds if 0 <= (r[1] - start.date()).days <= 4]
        if not match:
            continue
        rnd, _, loc = match[0]
        out.append({"uid": f"{year}_{rnd:02d}_{code}", "year": year, "round": rnd,
                    "code": code, "session_key": s["session_key"], "location": loc})
    return sorted(out, key=lambda r: r["uid"])


# --------------------------------------------------------------------------
# one session -> canonical laps
# --------------------------------------------------------------------------

YELLOW_MAX_S = 120.0     # a yellow with no CLEAR found lasts at most this long
RED_MAX_S = 1800.0       # a red flag with no restart found


def _status_intervals(rc: list[dict]) -> list[tuple[float, float, str]]:
    """Time windows of non-green running, from race-control messages.

    Practice and qualifying messages carry NO lap number -- only a time and,
    for yellows, a sector -- so status must be rebuilt by time: a sector
    yellow runs until that sector's CLEAR, a red flag until the next green.
    Measured on 2023 Baku FP1, the lap-number-only reading marked all 354
    laps green against FastF1's 140 non-green.
    """
    msgs = sorted((m for m in rc if m.get("date")), key=lambda m: m["date"])
    out = []
    for i, m in enumerate(msgs):
        flag = (m.get("flag") or "").upper()
        t0 = _ts(m["date"])
        if flag in ("YELLOW", "DOUBLE YELLOW"):
            sector = m.get("sector")
            end = t0 + YELLOW_MAX_S
            for n in msgs[i + 1:]:
                nf = (n.get("flag") or "").upper()
                if nf == "CLEAR" and (sector is None or n.get("sector") in (sector, None)):
                    end = _ts(n["date"])
                    break
            out.append((t0, max(end, t0 + 1.0), "2"))
        elif flag == "RED":
            end = t0 + RED_MAX_S
            for n in msgs[i + 1:]:
                if (n.get("flag") or "").upper() == "GREEN":
                    end = _ts(n["date"])
                    break
            out.append((t0, end, "5"))
    return out


def _lap_status(rc: list[dict]) -> dict[int, str]:
    """Lap -> code for messages that DO carry a lap number: the race's
    safety car and virtual safety car, deployed to recalled."""
    status: dict[int, set[str]] = {}
    sc_from, vsc_from = None, None
    for m in sorted(rc, key=lambda m: m.get("date") or ""):
        lap = m.get("lap_number")
        if lap is None:
            continue
        msg = (m.get("message") or "").upper()
        if "VIRTUAL SAFETY CAR DEPLOYED" in msg:
            vsc_from = lap
        elif "VIRTUAL SAFETY CAR ENDING" in msg and vsc_from is not None:
            for k in range(vsc_from, lap + 1):
                status.setdefault(k, set()).add("6")
            vsc_from = None
        elif "SAFETY CAR DEPLOYED" in msg:
            sc_from = lap
        elif "SAFETY CAR IN THIS LAP" in msg and sc_from is not None:
            for k in range(sc_from, lap + 1):
                status.setdefault(k, set()).add("4")
            sc_from = None
    for start, code in ((sc_from, "4"), (vsc_from, "6")):     # never recalled
        if start is not None:
            status.setdefault(start, set()).add(code)
    return {k: "".join(sorted(v)) for k, v in status.items()}


def _track_status(lap_n: int, t_start: float, t_end: float,
                  by_lap: dict[int, str], windows: list[tuple[float, float, str]]) -> str:
    """FastF1-style code for one lap: lap-numbered events plus any timed
    window the lap overlaps. "1" only when nothing applies."""
    codes = set(by_lap.get(lap_n, ""))
    if t_start == t_start and t_end == t_end:            # both finite
        for w0, w1, c in windows:
            if t_start < w1 and t_end > w0:
                codes.add(c)
    return "".join(sorted(codes)) if codes else "1"


def align_stints(ss: list[dict], dn: int, pit_out: set[tuple[int, int]]) -> None:
    """Snap race stint boundaries onto pit-out laps, in place.

    A race stint after the first begins on the lap the car leaves the pits.
    OpenF1 sometimes starts it one lap early or late -- measured on 2025
    Montreal, where it put HAM's second stint on lap 15, his in-lap, and
    every later tyre age in the stint came out one off. When a start is
    exactly one lap from a pit-out lap and is not one itself, it moves there
    and the previous stint's end follows. Anything else is left alone.
    """
    for prev, cur in zip(ss, ss[1:]):
        start = cur["lap_start"]
        if (dn, start) in pit_out:
            continue
        for shift in (1, -1):
            if (dn, start + shift) in pit_out:
                cur["lap_start"] = start + shift
                prev["lap_end"] = start + shift - 1
                break


def load_session(meta: dict) -> pl.DataFrame:
    """One session as canonical laps; an empty frame if OpenF1 has nothing."""
    sk = meta["session_key"]
    laps = get("laps", session_key=sk)
    if not laps:
        return schema.empty()
    stints = get("stints", session_key=sk)
    drivers = {d["driver_number"]: d for d in get("drivers", session_key=sk)}
    rc = get("race_control", session_key=sk)
    by_lap, windows = _lap_status(rc), _status_intervals(rc)

    by_drv: dict[int, list[dict]] = {}
    for s in stints:
        if s.get("lap_start") is not None:
            by_drv.setdefault(s["driver_number"], []).append(dict(s))
    if meta["code"] == "R":
        pit_out = {(lp["driver_number"], lp["lap_number"]) for lp in laps if lp.get("is_pit_out_lap")}
        for dn, ss in by_drv.items():
            align_stints(sorted(ss, key=lambda x: x["stint_number"]), dn, pit_out)

    rows = []
    for lap in laps:
        dn, n = lap["driver_number"], lap["lap_number"]
        st = next((s for s in by_drv.get(dn, [])
                   if s["lap_start"] <= n <= (s.get("lap_end") or 10_000)), None)
        if st is None or lap.get("lap_duration") is None:
            continue
        comp = (st.get("compound") or "UNKNOWN").upper()
        later = [s for s in by_drv[dn] if s["stint_number"] > st["stint_number"]]
        d = drivers.get(dn, {})
        rows.append({
            "driver": d.get("name_acronym") or str(dn),
            "team": d.get("team_name") or "?",
            "stint_id": int(st["stint_number"]),
            "lap_in_session": float(n),
            "_start": _ts(lap.get("date_start")),
            # FastF1's TyreLife counts the current lap: 1 on a new set's first lap
            "tyre_age": float((st.get("tyre_age_at_start") or 0) + (n - st["lap_start"]) + 1),
            "compound_rel": comp if comp in _REL else "UNKNOWN",
            "is_fresh": (st.get("tyre_age_at_start") or 0) == 0,
            "lap_time_s": float(lap["lap_duration"]),
            "sector_1_s": lap.get("duration_sector_1"),
            "sector_2_s": lap.get("duration_sector_2"),
            "sector_3_s": lap.get("duration_sector_3"),
            "pit_out": bool(lap.get("is_pit_out_lap")),
            # an in-lap ends a stint that another stint follows (or ends the run in practice)
            "pit_in": n == st.get("lap_end") and (bool(later) or meta["code"] != "R"),
            "track_status": _track_status(
                n, _ts(lap.get("date_start")),
                _ts(lap.get("date_start")) + float(lap["lap_duration"]), by_lap, windows),
        })
    if not rows:
        return schema.empty()

    df = pl.DataFrame(rows).with_columns(pl.col("_start").fill_nan(None))
    t0 = df["_start"].min()
    df = df.with_columns(
        pl.lit(meta["uid"]).alias("session_uid"),
        pl.lit(meta["year"]).cast(pl.Int16).alias("year"),
        pl.lit(meta["round"]).cast(pl.Int8).alias("round"),
        pl.lit(meta["location"]).alias("circuit"),
        pl.lit(meta["code"]).alias("session_type"),
        pl.lit("C3").alias("compound_abs"),            # placeholder, as FastF1 path
        (pl.col("_start") - t0).alias("session_clock_s"),
        pl.lit(np.nan).alias("fuel_load_kg_est"), pl.lit(np.nan).alias("fuel_load_kg_sd"),
        pl.lit(np.nan).alias("gap_ahead_s"), pl.lit(np.nan).alias("lift_coast_m"),
        pl.lit(np.nan).alias("track_temp_c"), pl.lit(np.nan).alias("air_temp_c"),
        pl.lit(True).alias("is_valid"),
        pl.lit(1).cast(pl.Int16).alias("lap_in_run"),
    ).drop("_start")

    # same clean-up as the FastF1 path: no clockless laps, 1-based lap_in_run
    df = df.filter(pl.col("lap_time_s").is_finite() & pl.col("session_clock_s").is_finite())
    df = (df.sort("driver", "stint_id", "lap_in_session")
            .with_columns((pl.int_range(pl.len()).over(["driver", "stint_id"]) + 1)
                          .cast(pl.Int16).alias("lap_in_run")))
    return schema.validate(df, source=f"openf1:{meta['uid']}")
