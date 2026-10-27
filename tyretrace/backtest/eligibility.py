"""Race eligibility from observable flags; future pace never selects anchors.

Unlike the historical long-run audit, this does not trim by a whole stint's
lap-time distribution, variance or eventual length. Race strategy and distance
remain conditioned inputs, and dry-race selection remains retrospective.
"""
import polars as pl
from tyretrace.features import traffic


def race_laps(raw: pl.DataFrame):
    d=traffic.dirty_air_index(traffic.add_gap_ahead(raw))
    return d.filter((pl.col('track_status')=='1') & ~pl.col('pit_in') & ~pl.col('pit_out')
        & pl.col('is_valid') & pl.col('lap_time_s').is_finite()
        & pl.col('compound_rel').cast(pl.Utf8).is_in(['SOFT','MEDIUM','HARD'])
        & (pl.col('lap_in_session')>1)).with_columns(pl.col('session_uid').str.slice(0,7).alias('weekend'))
