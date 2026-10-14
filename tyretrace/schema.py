"""Canonical lap table. The contract between ingest, simulator and every model.

The point of this module: the synthetic twin and the real FastF1 ingest emit
EXACTLY this schema, so no model can tell them apart. That is what makes the
recovery tests in tests/ meaningful, and what lets modelling start before
ingest is finished.

Index variables are kept deliberately distinct -- the whole identification
argument lives in the difference between them:

    lap_in_run      fuel burn and tyre wear both index on this   (confounded)
    session_clock_s track evolution indexes on this              (decoupling)
    lap_in_session  race lap; fuel indexes on this across stops  (pit reset)
"""
from __future__ import annotations

import polars as pl

# --- enumerations ---------------------------------------------------------

SESSION_TYPES = ("FP1", "FP2", "FP3", "Q", "SQ", "R", "S", "SIM", "RIG")

# Absolute Pirelli construction. NEVER pool on the relative label: "SOFT" is
# three different rubbers across a season, which silently invalidates any
# cross-race pooling.
COMPOUNDS_ABS = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "INT", "WET")
COMPOUNDS_REL = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN")

# --- schema ---------------------------------------------------------------

LAPS_SCHEMA: dict[str, pl.DataType] = {
    # keys ---------------------------------------------------------------
    "session_uid":     pl.Utf8,     # "2024_10_R"  year_round_session
    "year":            pl.Int16,
    "round":           pl.Int8,
    "circuit":         pl.Utf8,
    "session_type":    pl.Enum(SESSION_TYPES),
    "driver":          pl.Utf8,     # 3-letter code
    "team":            pl.Utf8,

    # index variables -- see module docstring -----------------------------
    "stint_id":        pl.Int32,    # unique within session_uid + driver
    "lap_in_run":      pl.Int16,    # 1-based, resets each stint
    "lap_in_session":  pl.Int16,    # 1-based, monotone across the session
    "session_clock_s": pl.Float64,  # seconds from session green

    # tyre state ---------------------------------------------------------
    "tyre_age":        pl.Int16,    # laps on this set INCLUDING prior scrub
    "compound_rel":    pl.Enum(COMPOUNDS_REL),
    "compound_abs":    pl.Enum(COMPOUNDS_ABS),
    "is_fresh":        pl.Boolean,  # set was new at stint start

    # response -----------------------------------------------------------
    "lap_time_s":      pl.Float64,
    "sector_1_s":      pl.Float64,
    "sector_2_s":      pl.Float64,
    "sector_3_s":      pl.Float64,

    # fuel -- LATENT. Never observed in public data; prior only. ----------
    "fuel_load_kg_est": pl.Float64,
    "fuel_load_kg_sd":  pl.Float64,

    # covariates -- these are MODELLED, never used to filter laps ---------
    "gap_ahead_s":     pl.Float64,
    "lift_coast_m":    pl.Float64,  # throttle-off distance before braking
    "track_temp_c":    pl.Float64,
    "air_temp_c":      pl.Float64,
    "track_status":    pl.Utf8,     # green / SC / VSC / yellow
    "pit_in":          pl.Boolean,
    "pit_out":         pl.Boolean,
    "is_valid":        pl.Boolean,  # deleted/invalid lap flag, NOT a filter

    # per-corner features, nullable (telemetry tier only) -----------------
    # struct[] one entry per corner: apex_v, apex_to_throttle_s,
    # brake_point_m, mid_corner_v_var, plus a measurement SD for each.
    "corner_feats":    pl.List(pl.Struct({
        "corner_id":            pl.Int16,
        "apex_v_kph":           pl.Float32,
        "apex_v_sd":            pl.Float32,
        "apex_to_throttle_s":   pl.Float32,
        "apex_to_throttle_sd":  pl.Float32,
        "brake_point_m":        pl.Float32,
        "brake_point_sd":       pl.Float32,
        "mid_corner_v_var":     pl.Float32,
    })),

    # GROUND TRUTH -- populated by sim/ and rig/ only. Always null on real
    # data. Any model that reads these outside a recovery test is cheating.
    "truth_fuel_kg":   pl.Float64,
    "truth_wear_fl":   pl.Float64,
    "truth_wear_fr":   pl.Float64,
    "truth_wear_rl":   pl.Float64,
    "truth_wear_rr":   pl.Float64,
    "truth_deg_s":     pl.Float64,  # true cumulative deg contribution, s
    "truth_regime":    pl.Int8,     # true latent regime index
}

TRUTH_COLS = tuple(c for c in LAPS_SCHEMA if c.startswith("truth_"))
REQUIRED = tuple(c for c in LAPS_SCHEMA if not c.startswith("truth_")
                 and c != "corner_feats")


def empty() -> pl.DataFrame:
    """An empty frame carrying the full canonical schema."""
    return pl.DataFrame(schema=LAPS_SCHEMA)


def conform(df: pl.DataFrame) -> pl.DataFrame:
    """Add missing optional columns as nulls, cast, and order canonically.

    Raises if a required column is absent -- that is a producer bug, and
    silent schema drift between sim/ and data/ would invalidate every
    recovery test at once without anyone noticing.
    """
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"producer is missing required columns: {missing}")

    for col, dtype in LAPS_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(col))

    return df.select([pl.col(c).cast(dt) for c, dt in LAPS_SCHEMA.items()])


def validate(df: pl.DataFrame, *, source: str = "?") -> pl.DataFrame:
    """Contract check at the boundary. Call on every read and every emit."""
    df = conform(df)

    problems: list[str] = []

    if df.height == 0:
        problems.append("empty frame")

    # index invariants -- these encode the identification argument, so a
    # violation here means the decorrelation logic is reading garbage.
    if df["lap_in_run"].min() is not None and df["lap_in_run"].min() < 1:
        problems.append("lap_in_run must be 1-based")

    bad_clock = df.filter(pl.col("session_clock_s") < 0).height
    if bad_clock:
        problems.append(f"{bad_clock} rows with negative session_clock_s")

    # tyre_age must never decrease inside a stint
    non_mono = (
        df.sort("session_uid", "driver", "stint_id", "lap_in_run")
          .with_columns(
              (pl.col("tyre_age").diff().over(["session_uid", "driver", "stint_id"]) < 0)
              .alias("_back")
          )["_back"].sum()
    )
    if non_mono:
        problems.append(f"{non_mono} rows where tyre_age decreases within a stint")

    # lap_in_run must restart at 1 for each stint
    starts = (
        df.group_by(["session_uid", "driver", "stint_id"])
          .agg(pl.col("lap_in_run").min().alias("m"))
          .filter(pl.col("m") != 1).height
    )
    if starts:
        problems.append(f"{starts} stints whose lap_in_run does not start at 1")

    # track_status must use the FastF1 encoding, not prose. The simulator once
    # emitted "green" here while the real ingest emitted "1", so every
    # green-only filter silently dropped all simulated laps -- which is
    # exactly the schema drift that would invalidate every recovery test at
    # once without failing loudly. Codes: 1 green, 2 yellow, 4 SC, 5 red,
    # 6/7 VSC; a lap carries every code seen during it, so "12" is legal.
    bad_status = (
        df.filter(pl.col("track_status").is_not_null()
                  & ~pl.col("track_status").str.contains(r"^[1-7]+$")).height
    )
    if bad_status:
        got = (df.filter(~pl.col("track_status").str.contains(r"^[1-7]+$"))
                 ["track_status"].unique().head(3).to_list())
        problems.append(
            f"{bad_status} rows with non-FastF1 track_status (e.g. {got}); "
            "expected digit codes like '1', not prose")

    if problems:
        raise ValueError(f"[{source}] schema contract violated:\n  - "
                         + "\n  - ".join(problems))
    return df
