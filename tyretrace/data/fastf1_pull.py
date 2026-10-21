"""FastF1 -> canonical lap table.

Timing only. Telemetry stays out of this path on purpose: the raw traces are
plausibly 200-600 GB across the full history, and every model consumes only
the extracted per-corner features. Features get computed at ingest by
tyretrace.features and joined in; the traces are never stored.

Emits exactly tyretrace.schema.LAPS_SCHEMA, so anything downstream cannot
tell a real session from a simulated one.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl

from tyretrace import schema

CACHE = Path(__file__).resolve().parents[2] / "cache" / "fastf1"

_REL = {
    "SOFT": "SOFT", "MEDIUM": "MEDIUM", "HARD": "HARD",
    "INTERMEDIATE": "INTERMEDIATE", "WET": "WET",
}


def enable_cache() -> None:
    import fastf1
    CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))


def _td_seconds(s):
    """Timedelta column -> float seconds, NaT-safe."""
    return s.dt.total_seconds().to_numpy(dtype=float)


def load_session(year: int, rnd: int, session: str) -> pl.DataFrame:
    """One session as canonical laps. Returns an empty frame on failure."""
    import fastf1

    enable_cache()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ses = fastf1.get_session(year, rnd, session)
            ses.load(telemetry=False, weather=True, messages=False)
        except Exception as exc:                      # noqa: BLE001
            print(f"    ! {year} r{rnd} {session}: {type(exc).__name__}: {exc}")
            return schema.empty()

    # FastF1 logs, rather than raises, when lap data fails to load -- the
    # error only surfaces here. One bad session must not abort a bulk pull.
    try:
        laps = ses.laps
    except Exception as exc:                          # noqa: BLE001
        print(f"    ! {year} r{rnd} {session}: laps not loaded ({type(exc).__name__})")
        return schema.empty()
    if laps is None or len(laps) == 0:
        return schema.empty()

    laps = laps.copy()
    uid = f"{year}_{rnd:02d}_{session}"

    lap_time = _td_seconds(laps["LapTime"])
    start = _td_seconds(laps["LapStartTime"])
    # session clock measured from the first green lap start in the session
    t0 = np.nanmin(start) if np.isfinite(start).any() else 0.0
    clock = start - t0

    stint = laps["Stint"].to_numpy(dtype=float)
    stint = np.where(np.isfinite(stint), stint, -1).astype(int)

    tyre_life = laps["TyreLife"].to_numpy(dtype=float)
    comp = laps["Compound"].astype(str).str.upper().to_numpy()
    comp_rel = np.array([_REL.get(c, "UNKNOWN") for c in comp])

    try:
        circuit = ses.event["Location"]
    except Exception:                                 # noqa: BLE001
        circuit = "?"

    df = pl.DataFrame({
        "session_uid": [uid] * len(laps),
        "year": np.full(len(laps), year, dtype=np.int16),
        "round": np.full(len(laps), rnd, dtype=np.int8),
        "circuit": [str(circuit)] * len(laps),
        "session_type": [session] * len(laps),
        "driver": laps["Driver"].astype(str).to_numpy(),
        "team": laps["Team"].astype(str).to_numpy(),
        "stint_id": stint,
        "lap_in_run": np.ones(len(laps), dtype=np.int16),      # filled below
        "lap_in_session": laps["LapNumber"].to_numpy(dtype=float),
        "session_clock_s": clock,
        "tyre_age": np.where(np.isfinite(tyre_life), tyre_life, 0),
        "compound_rel": comp_rel,
        "compound_abs": ["C3"] * len(laps),   # placeholder: needs Pirelli map
        "is_fresh": laps["FreshTyre"].fillna(False).astype(bool).to_numpy(),
        "lap_time_s": lap_time,
        "sector_1_s": _td_seconds(laps["Sector1Time"]),
        "sector_2_s": _td_seconds(laps["Sector2Time"]),
        "sector_3_s": _td_seconds(laps["Sector3Time"]),
        "fuel_load_kg_est": np.full(len(laps), np.nan),
        "fuel_load_kg_sd": np.full(len(laps), np.nan),
        "gap_ahead_s": np.full(len(laps), np.nan),
        "lift_coast_m": np.full(len(laps), np.nan),
        "track_temp_c": np.full(len(laps), np.nan),
        "air_temp_c": np.full(len(laps), np.nan),
        "track_status": laps["TrackStatus"].astype(str).to_numpy(),
        "pit_in": laps["PitInTime"].notna().to_numpy(),
        "pit_out": laps["PitOutTime"].notna().to_numpy(),
        "is_valid": laps["IsAccurate"].fillna(False).astype(bool).to_numpy(),
    })

    # drop laps with no usable time or clock -- they carry no information and
    # would break the monotonicity contract
    df = df.filter(pl.col("lap_time_s").is_finite()
                   & pl.col("session_clock_s").is_finite()
                   & (pl.col("stint_id") >= 0))
    if df.height == 0:
        return schema.empty()

    # rebuild lap_in_run as a 1-based rank inside each stint
    df = (df.sort("driver", "stint_id", "lap_in_session")
            .with_columns(
                (pl.int_range(pl.len()).over(["driver", "stint_id"]) + 1)
                .cast(pl.Int16).alias("lap_in_run")))

    # session clock must be non-negative
    df = df.with_columns(
        pl.when(pl.col("session_clock_s") < 0).then(0.0)
          .otherwise(pl.col("session_clock_s")).alias("session_clock_s"))

    return schema.validate(df, source=f"fastf1:{uid}")


def long_runs(df: pl.DataFrame, min_laps: int = 4,
              green_only: bool = True, trim: bool = True) -> pl.DataFrame:
    """Stints with enough timed laps to carry a slope.

    Note this filters STINTS, not laps -- individual slow laps stay in and
    become covariates. Dropping laps is what the whole approach argues
    against; dropping two-lap stints is just requiring a slope to exist.
    """
    out = df
    if green_only:
        # exact "1" only. TrackStatus concatenates codes seen during the lap,
        # so contains("1") also matches "12" (yellow) and "15" (red).
        out = out.filter(pl.col("track_status") == "1")
    out = out.filter(~pl.col("pit_in") & ~pl.col("pit_out"))

    grp = ["session_uid", "driver", "stint_id"]

    # Robust trim against a LOW quantile, not the median: a run that is mostly
    # cool-down laps has a high median, and trimming to +7% of it keeps the
    # cool-downs and throws away the representative laps.
    #
    # The window has to stay wide enough to preserve genuine degradation --
    # 0.06 s/lap over 15 laps is ~0.9 s, about 1% of a 90 s lap -- while
    # cutting cool-downs, which are +10 s or more. 3.5% (or 3 s) does both.
    if trim:
        out = out.with_columns(
            pl.col("lap_time_s").quantile(0.20).over(grp).alias("_base"))
        out = out.filter(
            pl.col("lap_time_s")
            <= pl.col("_base") + pl.max_horizontal(3.0, 0.035 * pl.col("_base"))
        ).drop("_base")

    # Runs still wildly dispersed after that are not long runs at all
    # (red-flag splits, mixed programmes sharing one stint id). Drop the run.
    stats = (out.group_by(grp)
                .agg([pl.len().alias("n"),
                      pl.col("lap_time_s").std().alias("sd")]))
    keep = (stats.filter((pl.col("n") >= min_laps)
                         & (pl.col("sd") <= 2.0))
                 .select(grp))
    out = out.join(keep, on=grp, how="inner")

    return (out.sort("session_uid", "driver", "stint_id", "lap_in_session")
               .with_columns(
                   (pl.int_range(pl.len()).over(["session_uid", "driver", "stint_id"]) + 1)
                   .cast(pl.Int16).alias("lap_in_run")))
