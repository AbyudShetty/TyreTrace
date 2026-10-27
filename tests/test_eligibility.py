"""Future target pace cannot decide which opening laps anchor the forecast."""
import sys
from pathlib import Path
import polars as pl
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tyretrace.backtest.eligibility import race_laps
from tyretrace.sim.minimal import simulate_race


def test_flag_eligibility_preserves_all_future_pace_and_anchor_identities():
    raw=simulate_race(np.random.default_rng(14))
    # Use a real canonical synthetic frame, with deliberately wild future pace.
    raw=raw.with_columns(pl.lit('1').alias('track_status'),pl.lit(True).alias('is_valid'),pl.lit('MEDIUM').alias('compound_rel'))
    poison=raw.with_columns(pl.when(pl.col('lap_in_session')>8).then(pl.col('lap_time_s')+500).otherwise(pl.col('lap_time_s')).alias('lap_time_s'))
    a,b=race_laps(raw),race_laps(poison)
    keys=['session_uid','driver','stint_id','lap_in_session']
    assert a.select(keys).sort(keys).to_dicts()==b.select(keys).sort(keys).to_dicts()
    assert a.height>20
