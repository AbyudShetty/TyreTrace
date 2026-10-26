"""Conditional optimal-stop error against a post-race fitted reference.

This is a decision diagnostic, not observed counterfactual ground truth. Only
drivers with exactly two observed dry-compound stints are evaluated, conditioned
on that compound order and race distance. Forecast parameters remain those
saved before target outcomes were used by the rolling backtest.
"""
from pathlib import Path
import sys
import numpy as np
import polars as pl

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tyretrace import artifacts,results
from tyretrace.backtest import fp_race as B
from tyretrace.backtest.postrace import post_race_fit
from tyretrace.data import sources
from tyretrace.eval.strategy import decision_error


def main():
    d=sources.load('lake')
    _,race=B.prepare(d['practice'],d['race'])
    params=artifacts.load('backtest_lake','params')
    if params is None:raise FileNotFoundError('run backtest_lake first')
    rows=[]
    for wk in sorted(params['weekend'].unique().to_list()):
        r=race.filter(pl.col('weekend')==wk)
        ref=post_race_fit(r)
        race_length=int(d['race'].filter(pl.col('session_uid')==wk+'_R')['lap_in_session'].max())
        curves=params.filter(pl.col('weekend')==wk)
        for (driver,),g in r.group_by('driver'):
            if g['lap_in_session'].max()<.9*race_length:continue
            stints=(g.group_by('stint_id').agg(pl.col('compound_rel').cast(pl.Utf8).first().alias('compound'),
                pl.col('lap_in_session').min().alias('start'),pl.col('tyre_age').min().alias('age'))
                .sort('start'))
            if stints.height!=2:continue
            first,second=stints['compound'].to_list()
            if first==second or first not in ref['deg'] or second not in ref['deg']:continue
            for (method,),m in curves.group_by('method'):
                deg=dict(m.select('compound','deg').iter_rows())
                if first not in deg or second not in deg:continue
                error=decision_error(race_length,(deg[first],deg[second]),(ref['deg'][first],ref['deg'][second]))
                rows.append({'weekend':wk,'driver':driver,'method':method,'first':first,
                    'second':second,'race_laps':race_length,**error})
    if not rows:raise ValueError('no eligible one-stop reference comparisons')
    table=pl.DataFrame(rows)
    summary=table.group_by('method').agg(pl.len().alias('drivers'),pl.col('stop_lap_error').mean().alias('stop_mae'),
        pl.col('regret_s').mean().alias('mean_regret_s'),pl.col('regret_s').median().alias('median_regret_s'))
    artifacts.save('strategy_reference','decisions',table)
    artifacts.save('strategy_reference','summary',summary)
    results.record('strategy_reference',{'scope':'conditional one-stop decisions versus fitted post-race reference; not observed optimal strategy',
        'weekends':table['weekend'].n_unique(),'comparisons':table.height,'methods':summary.to_dicts()},gate='PASS')
    print(summary)
    print('DESCRIPTIVE REFERENCE ONLY: compound order/distance conditioned, fresh sets assumed; no superiority ship gate.')
    return 0


if __name__=='__main__':raise SystemExit(main())
