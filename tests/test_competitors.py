"""Target outcomes cannot tune a rolling competitor or its noise estimate."""
import sys
from pathlib import Path
from dataclasses import replace
import numpy as np
import polars as pl
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tyretrace.backtest import competitors as C,fp_race as B
from test_backtest import practice,race,WK


def inputs(key,seed=0):
    w=replace(WK,key=key)
    p=practice(seed=seed).with_columns(pl.lit('Monza').alias('circuit'),pl.col('lap_in_run').alias('tyre_age'),pl.col('lap_in_run').alias('lap_in_session'))
    r=race(seed=seed,n_drv=3,n_laps=30).with_columns(pl.lit('Monza').alias('circuit'))
    return w,p,r,C.frame(w,p,r,(.04,.02),(0.,0.))


def test_target_features_and_predictions_ignore_future_lap_times():
    history=[inputs(f'2024_{i:02d}',i)[3] for i in range(1,5)]
    w,p,r,target=inputs('2024_05',5)
    poison=r.with_columns(pl.when(pl.col('lap_in_session')>6).then(pl.col('lap_time_s')+500).otherwise(pl.col('lap_time_s')).alias('lap_time_s'))
    other=C.frame(w,p,poison,(.04,.02),(0.,0.))
    assert np.allclose(target.features,other.features,atol=1e-12,rtol=0)
    assert np.array_equal(target.level,other.level)
    for fn in (C.ml,C.published):
        args=() if fn==C.ml else (p,w,(.04,.02))
        a=fn(history,target,*args);b=fn(history,other,*args)
        assert np.allclose(a[0],b[0],atol=1e-12,rtol=0) and np.allclose(a[1],b[1],atol=1e-12,rtol=0)
        assert np.all(np.isfinite(a[0])) and np.all(a[1]>0)


def test_frame_targets_match_original_forecast_identities():
    w,p,r,frame=inputs('2024_05')
    fc=B.forecast(r,{'HARD':(.03,.001,'test'),'MEDIUM':(.05,.001,'test')},(-.04,.001),.1)
    keys=['driver','stint','lap_in_session']
    a=frame.labels.select(keys).sort(keys)
    b=fc.laps.rename({'stint_id':'stint'}).select(keys).sort(keys)
    assert a.to_dicts()==b.to_dicts()


def test_quadratic_conventional_recovers_known_curvature_without_future_outcomes():
    w,p,r,target=inputs('2024_05',5)
    # A known quadratic in tyre age, arbitrary run levels, and fixed fuel.
    p=p.with_columns((90+.04*pl.col('tyre_age')+.001*pl.col('tyre_age')**2-.030*w.kappa*pl.col('lap_in_session')+pl.col('stint_id')).alias('lap_time_s'))
    a=C.conventional_quadratic(target,p,r,w,(.04,.02),.02)
    poison=r.with_columns(pl.when(pl.col('lap_in_session')>6).then(pl.col('lap_time_s')+500).otherwise(pl.col('lap_time_s')).alias('lap_time_s'))
    other=C.frame(w,p,poison,(.04,.02),(0.,0.))
    b=C.conventional_quadratic(other,p,poison,w,(.04,.02),.02)
    assert np.allclose(a[0],b[0],atol=1e-12,rtol=0)
    assert np.all(a[1]>0)
    rows=target.labels;driver=rows['driver'][0]
    anchor=r.filter(pl.col('driver')==driver).sort('lap_in_session').head(5)['tyre_age'].to_numpy()
    age=float(rows['tyre_age'][0]);dt=float(rows['lap_in_session'][0])-float(anchor.mean())
    expected=target.level[0]-.030*w.kappa*dt+.04*(age-anchor.mean())+.001*(age**2-np.mean(anchor**2))
    assert abs(a[0][0]-expected)<.01
    assert a[2]['curve']=='within-weekend quadratic'


@pytest.mark.parametrize('fn',[C.ml,C.published])
def test_same_or_later_weekend_history_is_rejected(fn):
    w,p,r,target=inputs('2024_05')
    hist=[inputs(k)[3] for k in ('2024_01','2024_03','2024_05')]
    with pytest.raises(ValueError,match='earlier'):
        fn(hist,target,*(() if fn==C.ml else (p,w,(.04,.02))))
