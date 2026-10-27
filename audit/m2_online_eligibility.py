"""Flag-only anchor selection sensitivity; retrospective, conditioned strategy.

This tests a new software eligibility rule on previously inspected data. It is
not prospective validation. Compare ML, identified and quadratic conventional
on the same expanded race population, with separately calibrated forecasts.
"""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import polars as pl
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tyretrace import artifacts,results
from tyretrace.backtest import fp_race as B,competitors as C
from tyretrace.backtest.eligibility import race_laps
from tyretrace.data import sources
from tyretrace.data.compounds import nominations
from tyretrace.eval import metrics
from tyretrace.infer import uncertainty as U


def main():
    d=sources.load('lake');practice,_=B.prepare(d['practice'],d['race'])
    race=race_laps(d['race']);wk=B.weekend_inputs(d['quali'],d['race'])
    dry=B.dry_weekends(d['race']);keys=sorted(k for k in wk if k in dry)
    rc={k:B.race_lap_coef(race.filter(pl.col('weekend')==k)) for k in keys}
    rd={k:B.race_degradation(race.filter(pl.col('weekend')==k)) for k in keys}
    tf=B.traffic_history(race,keys);reg=nominations();history=[];tabs=[]
    for i,k in enumerate(keys):
        earlier=keys[:i];prior=B.degradation_prior([rd[e] for e in earlier]);traffic=B.traffic_prior([tf[e] for e in earlier])
        p=practice.filter(pl.col('weekend')==k);r=race.filter(pl.col('weekend')==k)
        target=C.frame(wk[k],p,r,prior,traffic,reg)
        if i>=B.MIN_EARLIER:
            sigma=float(np.median([rc[e][2] for e in earlier]))
            ft=B.fit_fuel_track([rc[e][0] for e in earlier],[rc[e][1] for e in earlier],[wk[e] for e in earlier])
            forecasts=B.run_weekend(wk[k],p,r,ft,sigma,seed=i,prior=prior,traffic=traffic,methods=('identified','naive'))
            for method,fc in forecasts.items():
                t=U.from_forecast(fc,weekend=k,method=method)
                tabs.append(t.with_columns(pl.Series('crps',metrics.crps_ensemble(fc.draws,t['y'].to_numpy()))))
            for method in ('ml','conventional_quadratic','published'):
                if method=='ml':mean,sd,_=C.ml(history,target)
                elif method=='published':mean,sd,_=C.published(history,target,p,wk[k],prior,reg)
                else:mean,sd,_=C.conventional_quadratic(target,p,r,wk[k],prior,sigma)
                draws=mean[None,:]+sd[None,:]*np.random.default_rng(i).standard_normal((1000,len(mean)))
                t=artifacts.predictive(draws,target.labels['y'].to_numpy(),**{c:target.labels[c].to_numpy() for c in target.labels.columns if c!='y'})
                tabs.append(t.with_columns(pl.lit(method).alias('method')))
            print(f'{k}: flag-only forecasts complete',flush=True)
        history.append(target)
    cols=['weekend','driver','stint','lap_in_session','tyre_age','after_stop','y','q05','q50','q95','crps','method']
    raw=pl.concat([t.select(cols) for t in tabs],how='vertical_relaxed')
    artifacts.save('online_eligibility','predictive',raw)
    calibrated=[];rows=[]
    for method in ('identified','ml','conventional_quadratic','published','naive'):
        t,_=U.rolling(raw.filter(pl.col('method')==method),score_from='2025');calibrated.append(t)
        for year in ('2025','2026'):
            for subset in ('all','after_stop'):
                g=t.filter(pl.col('weekend').str.starts_with(year))
                if subset=='after_stop':g=g.filter(pl.col('after_stop'))
                score=U.score(g);rows.append({'method':method,'year':year,'subset':subset,**{k:v for k,v in score.items() if k!='calibration'}})
    artifacts.save('online_eligibility','calibrated',pl.concat(calibrated,how='diagonal_relaxed'))
    artifacts.save('online_eligibility','summary',pl.DataFrame(rows))
    results.record('online_eligibility',{'summary':rows,
        'input_sha256':{name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in sources.PATHS['lake'].items()},
        'scope':'retrospective flag-only race eligibility; known future tyre strategy/distance; qualifying and five green race anchors; no future pace used in selection'},gate='PASS')
    print(json.dumps(rows,indent=2),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
