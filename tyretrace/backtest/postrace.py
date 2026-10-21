"""Post-race validation: why was each lap's forecast wrong?

The submission promised "the residual decomposed into fuel error, traffic,
track evolution and genuine model miss -- so a wrong call can be traced to
which assumption failed". This is that, and it is EXACT: the parts sum to the
forecast error lap by lap, with no unexplained remainder hidden anywhere.

After the race, fit what the race itself says, per weekend:

    y = a_driver + slope * lap + sum_c deg_c * age + wake * dirty_air + r

The pre-race forecast for a lap anchored on the driver's first laps was

    f = L_driver + slope_fc * lap + deg_fc[c] * age,
    L_driver = mean over anchor laps of (y - slope_fc * lap - deg_fc * age)

Subtracting, the error  e = f - y  splits into four terms:

    slope      (slope_fc - slope) * (lap - mean anchor lap)
    tyres      (deg_fc - deg) * age, relative to the anchor laps
    traffic    -wake * (dirty_air - mean anchor dirty_air)
    noise      mean anchor residual - this lap's residual

"slope" is fuel and race track evolution together: a race cannot separate
them (that is the whole identification argument), so the decomposition does
not pretend to.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from tyretrace.backtest.fp_race import ANCHOR_LAPS, _dummies
from tyretrace.data import fastf1_pull as fp
from tyretrace.features import traffic

PARTS = ("slope", "tyres", "traffic", "noise")


def post_race_fit(race_lr: pl.DataFrame) -> dict:
    """What the race itself says: driver levels, slope, degradation, wake."""
    drv = race_lr["driver"].to_numpy().astype(str)
    comp = race_lr["compound_rel"].cast(pl.Utf8).to_numpy().astype(str)
    age = race_lr["tyre_age"].to_numpy().astype(float)
    lap = race_lr["lap_in_session"].to_numpy().astype(float)
    da = race_lr["dirty_air"].to_numpy().astype(float)
    dl = sorted(set(drv.tolist()))
    cs = sorted(set(comp.tolist()))
    X = np.column_stack([_dummies(drv)] + [(comp == c) * age for c in cs] + [lap, da])
    y = race_lr["lap_time_s"].to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    nd = len(dl)
    return {"level": dict(zip(dl, coef[:nd])), "deg": dict(zip(cs, coef[nd:nd + len(cs)])),
            "slope": float(coef[-2]), "wake": float(coef[-1]), "resid": y - X @ coef}


def prepare_race(race_raw: pl.DataFrame) -> pl.DataFrame:
    """Gap to the car ahead needs every car on track, so it is computed
    before the long-run filter -- the same order the ladder uses."""
    with_gap = traffic.dirty_air_index(traffic.add_gap_ahead(race_raw))
    return fp.long_runs(with_gap, min_laps=4)


def decompose(race_lr: pl.DataFrame, slope_fc: float, deg_fc: dict[str, float]) -> pl.DataFrame:
    """Per target lap: actual, forecast, and the four parts of the error."""
    post = post_race_fit(race_lr)
    df = (race_lr.with_columns(pl.Series("_resid", post["resid"]))
                 .sort("driver", "lap_in_session")
                 .with_columns(pl.int_range(pl.len()).over("driver").alias("_k")))
    comp = df["compound_rel"].cast(pl.Utf8).to_numpy().astype(str)
    age = df["tyre_age"].to_numpy().astype(float)
    lap = df["lap_in_session"].to_numpy().astype(float)
    dd = np.array([deg_fc.get(c, np.nan) - post["deg"].get(c, np.nan) for c in comp]) * age
    df = df.with_columns(pl.Series("_dd", dd))

    anc = (df.filter(pl.col("_k") < ANCHOR_LAPS).group_by("driver")
             .agg(pl.col("lap_in_session").mean().alias("_lap_a"),
                  pl.col("_dd").mean().alias("_dd_a"),
                  pl.col("dirty_air").mean().alias("_da_a"),
                  pl.col("_resid").mean().alias("_r_a"),
                  pl.col("stint_id").first().alias("_stint_a")))
    t = df.filter(pl.col("_k") >= ANCHOR_LAPS).join(anc, on="driver")

    lap_t = t["lap_in_session"].to_numpy().astype(float)
    parts = {
        "slope": (slope_fc - post["slope"]) * (lap_t - t["_lap_a"].to_numpy()),
        "tyres": t["_dd"].to_numpy() - t["_dd_a"].to_numpy(),
        "traffic": -post["wake"] * (t["dirty_air"].to_numpy() - t["_da_a"].to_numpy()),
        "noise": t["_r_a"].to_numpy() - t["_resid"].to_numpy(),
    }
    error = sum(parts.values())

    # The identity must hold: rebuild the forecast from its own definition
    # and check it equals actual + the parts. A decomposition that only
    # "sums by construction" proves nothing.
    deg_t = np.array([deg_fc.get(c, np.nan) for c in t["compound_rel"].cast(pl.Utf8).to_list()])
    a = df.filter(pl.col("_k") < ANCHOR_LAPS)
    deg_a = np.array([deg_fc.get(c, np.nan) for c in a["compound_rel"].cast(pl.Utf8).to_list()])
    lvl = (a.with_columns(pl.Series("_l", a["lap_time_s"].to_numpy()
                                    - slope_fc * a["lap_in_session"].to_numpy()
                                    - deg_a * a["tyre_age"].to_numpy()))
            .group_by("driver").agg(pl.col("_l").mean()))
    lv = dict(lvl.iter_rows())
    direct = (np.array([lv[d] for d in t["driver"].to_list()]) + slope_fc * lap_t
              + deg_t * t["tyre_age"].to_numpy())
    gap = float(np.nanmax(np.abs(direct - (t["lap_time_s"].to_numpy() + error)))) if t.height else 0.0
    if gap > 1e-6:
        raise AssertionError(f"post-race decomposition does not sum to the error (gap {gap:.2e})")

    return pl.DataFrame({
        "driver": t["driver"], "stint": t["stint_id"], "lap_in_session": t["lap_in_session"],
        "tyre_age": t["tyre_age"], "compound": t["compound_rel"].cast(pl.Utf8),
        "after_stop": t["stint_id"] != t["_stint_a"],
        "actual": t["lap_time_s"], "forecast": t["lap_time_s"].to_numpy() + error,
        "error": error, **parts,
        "post_slope": np.full(t.height, post["slope"]), "post_wake": np.full(t.height, post["wake"]),
    })
