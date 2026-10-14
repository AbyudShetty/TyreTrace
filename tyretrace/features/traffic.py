"""Traffic features: gap to the car ahead, from timing data alone.

The submission's position is that dirty air is a COVARIATE, not a reason to
bin a lap -- "recovering those laps beats any change of model architecture".
This module supplies the covariate.

No telemetry needed. Every lap already carries the session clock at which it
started, so ordering cars by that clock within a lap number gives the running
order and the time gaps directly.

WHAT THIS IS FOR
----------------
L2 left an open puzzle: the identified fuel pace on real races is ~0.032 s/lap
against ~0.055-0.06 implied by fuel physics. Dirty air is a candidate
explanation with an observable proxy, and it makes a FALSIFIABLE prediction --
the field spreads out over a race (measured corr(race lap, gap) = +0.37), so
early laps are dirtier and slower, which is another NEGATIVE contribution to
the lap coefficient. Controlling for it should therefore make the puzzle
worse, not better. If it does, dirty air is ruled out as the explanation.
"""
from __future__ import annotations

import polars as pl

MAX_MEANINGFUL_GAP_S = 30.0     # beyond this the car ahead is irrelevant
DIRTY_AIR_S = 2.0               # conventional threshold for meaningful wake


def add_gap_ahead(df: pl.DataFrame) -> pl.DataFrame:
    """Populate `gap_ahead_s`: seconds to the car directly ahead at lap start.

    Computed on ALL laps, before any long-run filtering -- the running order
    depends on every car on track, including ones whose laps get trimmed
    later. Filtering first would corrupt the ordering.

    The leader of each lap has no car ahead and keeps a null.
    """
    out = (df.sort("session_uid", "lap_in_session", "session_clock_s")
             .with_columns(
                 (pl.col("session_clock_s")
                  - pl.col("session_clock_s").shift(1)
                                             .over(["session_uid", "lap_in_session"]))
                 .alias("_gap")))

    # negative gaps mean the ordering is unreliable (lapped traffic, timing
    # glitches); implausibly large ones mean nobody is meaningfully ahead
    out = out.with_columns(
        pl.when((pl.col("_gap") > 0) & (pl.col("_gap") <= MAX_MEANINGFUL_GAP_S))
          .then(pl.col("_gap"))
          .otherwise(None)
          .alias("gap_ahead_s")
    ).drop("_gap")

    return out


def dirty_air_index(df: pl.DataFrame, scale_s: float = DIRTY_AIR_S) -> pl.DataFrame:
    """A bounded 0-1 wake exposure, added as `dirty_air`.

    Aerodynamic wake decays with distance, so lap-time loss is nonlinear in
    the raw gap: the difference between 0.5 s and 1.5 s behind matters far
    more than between 10 s and 11 s. `exp(-gap / scale)` captures that with a
    single interpretable parameter and stays bounded, which keeps the
    regressor well conditioned.

    A null gap (the leader, or no car within range) means clear air, so 0.
    """
    return df.with_columns(
        pl.when(pl.col("gap_ahead_s").is_null())
          .then(0.0)
          .otherwise((-pl.col("gap_ahead_s") / scale_s).exp())
          .alias("dirty_air")
    )
