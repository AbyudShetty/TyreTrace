"""All baseline families and practice/regime candidates on identical rolling targets."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import polars as pl

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tyretrace import artifacts,results
from tyretrace.backtest import fp_race as B,competitors as C
from tyretrace.data import sources
from tyretrace.data.compounds import nominations,REGISTRY
from tyretrace.infer import uncertainty as U
from tyretrace.models import practice as P
from tyretrace.eval import metrics


def main():
    registry=nominations()
    registry_sha=hashlib.sha256(REGISTRY.read_bytes()).hexdigest()
    d=sources.load('lake');prac,race=B.prepare(d['practice'],d['race'])
    wk=B.weekend_inputs(d['quali'],d['race']);dry=B.dry_weekends(d['race'])
    keys=sorted(k for k in wk if k in dry)
    rc={k:B.race_lap_coef(race.filter(pl.col('weekend')==k)) for k in keys}
    rd={k:B.race_degradation(race.filter(pl.col('weekend')==k)) for k in keys}
    tf=B.traffic_history(race,keys)
    history=[];tabs=[];diagnostics=[]
    for i,k in enumerate(keys):
        prior=B.degradation_prior([rd[e] for e in keys[:i]])
        traffic=B.traffic_prior([tf[e] for e in keys[:i]])
        p=prac.filter(pl.col('weekend')==k);r=race.filter(pl.col('weekend')==k)
        frame=C.frame(wk[k],p,r,prior,traffic,registry)
        if i>=B.MIN_EARLIER:
            ft=B.fit_fuel_track([rc[e][0] for e in keys[:i]],[rc[e][1] for e in keys[:i]],[wk[e] for e in keys[:i]])
            sigma=float(np.median([rc[e][2] for e in keys[:i]]))
            for method,fn in [('ml',C.ml),('published',C.published),('conventional_quadratic',C.conventional_quadratic)]:
                if method=='ml':mean,sd,diag=fn(history,frame)
                elif method=='published':mean,sd,diag=fn(history,frame,p,wk[k],prior,registry)
                else:mean,sd,diag=fn(frame,p,r,wk[k],prior,sigma)
                rng=np.random.default_rng(i)
                draws=mean[None,:]+sd[None,:]*rng.standard_normal((1000,len(mean)))
                t=artifacts.predictive(draws,frame.labels['y'].to_numpy(),**{c:frame.labels[c].to_numpy() for c in frame.labels.columns if c!='y'})
                tabs.append(t.with_columns(pl.lit(method).alias('method')))
                diagnostics.append({'weekend':k,'method':method,'detail':json.dumps(diag)})
            for method,regimes in [('robust_practice',False),('regime_practice',True)]:
                slopes,diag=P.slopes(p,regimes=regimes)
                deg=B.degradation(slopes,ft.fuel_pace(wk[k]),wk[k].track_per_lap,
                    sorted(r['compound_rel'].cast(pl.Utf8).unique().to_list()),prior)
                forecast=B.forecast(r,deg,ft.lap_coef(wk[k]),sigma,seed=i,traffic=traffic)
                t=U.from_forecast(forecast,weekend=k,method=method)
                t=t.with_columns(pl.Series('crps',metrics.crps_ensemble(forecast.draws,t['y'].to_numpy())))
                tabs.append(t);diagnostics.append({'weekend':k,'method':method,'detail':json.dumps(diag)})
            print(f'{k}: five competitors complete ({i} earlier races)',flush=True)
        history.append(frame)
    original=artifacts.load('backtest_lake','predictive')
    if original is None:raise ValueError('run backtest_lake first')
    keep=['weekend','driver','stint','lap_in_session','tyre_age','after_stop','y','q05','q50','q95','crps','method']
    tabs.append(original.filter(pl.col('method').is_in(['naive','conventional','identified'])).select(keep))
    predictions=pl.concat([t.select(keep) for t in tabs],how='vertical_relaxed')
    artifacts.save('rolling_models','predictive',predictions)
    artifacts.save('rolling_models','diagnostics',pl.DataFrame(diagnostics))
    rows=[];cal=[];fit_tables=[];curves=[]
    for method in sorted(predictions['method'].unique().to_list()):
        table=predictions.filter(pl.col('method')==method)
        for year in ('2025','2026'):
            calibrated,fits=U.rolling(table,score_from=year,score_until=str(int(year)+1))
            cal.append(calibrated)
            fit_tables.append(fits.with_columns(pl.lit(method).alias('method')))
            for label,subset in [('all',calibrated),('after_stop',calibrated.filter(pl.col('after_stop')))]:
                score=U.score(subset)
                curves.extend({'method':method,'year':year,'subset':label,'level':level,'coverage':value} for level,value in score['calibration'].items())
                rows.append({'method':method,'year':year,'subset':label,**{k:v for k,v in score.items() if k!='calibration'}})
    artifacts.save('rolling_models','calibrated',pl.concat(cal,how='diagonal_relaxed'))
    artifacts.save('rolling_models','summary',pl.DataFrame(rows))
    artifacts.save('rolling_models','calibration',pl.DataFrame(curves))
    artifacts.save('rolling_models','fits',pl.concat(fit_tables,how='diagonal_relaxed'))
    # Select on 2025 only. 2026 is already inspected research data, so this is
    # a retrospective check, not a newly independent confirmation experiment.
    eligible=[r for r in rows if r['year']=='2025' and r['subset']=='all' and .85<=r['coverage_90']<=.95]
    selected=min(eligible,key=lambda r:r['crps'])['method'] if eligible else None
    comparison=[]
    calibrated=pl.concat(cal,how='diagonal_relaxed')
    identity=['weekend','driver','stint','lap_in_session']
    for method in sorted(predictions['method'].unique().to_list()):
        if method=='identified':continue
        for year in ('2025','2026'):
            for subset in ('all','after_stop'):
                a=calibrated.filter((pl.col('method')==method)&pl.col('weekend').str.starts_with(year))
                b=calibrated.filter((pl.col('method')=='identified')&pl.col('weekend').str.starts_with(year))
                if subset=='after_stop':a=a.filter(pl.col('after_stop'));b=b.filter(pl.col('after_stop'))
                a=a.sort(identity);b=b.sort(identity)
                if a.select(identity).to_dicts()!=b.select(identity).to_dicts() or not np.array_equal(a['y'],b['y']):
                    raise ValueError('rolling methods must score exactly the same target observations')
                sa,sb=U.score(a),U.score(b)
                deltas=(a.select('weekend','crps').group_by('weekend').mean().sort('weekend')['crps'].to_numpy()
                    -b.select('weekend','crps').group_by('weekend').mean().sort('weekend')['crps'].to_numpy())
                boot=np.random.default_rng(27).choice(deltas,(10000,len(deltas)),replace=True).mean(axis=1)
                comparison.append({'method':method,'year':year,'subset':subset,'crps_delta':sa['crps']-sb['crps'],
                    'weekend_mean_delta':float(deltas.mean()),'weekend_ci95':np.quantile(boot,[.025,.975]).tolist(),
                    'gate':'SHIP' if sa['crps']<sb['crps'] and .85<=sa['coverage_90']<=.95 else 'HOLD'})
    artifacts.save('rolling_models','comparison',pl.DataFrame(comparison))
    results.record('rolling_models',{'summary':rows,'comparison':comparison,'selected_on_2025':selected,
        'nomination_sha256':registry_sha,
        'input_sha256':{name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in sources.PATHS['lake'].items()},
        'scope':'retrospective rolling competitors; same outcome-filtered eligible laps and known tyre strategy/distance; five race anchors and qualifying; no target-outcome tuning of predictor',
        'weekends':predictions['weekend'].n_unique(),'methods':sorted(predictions['method'].unique().to_list())},gate='PASS')
    print(pl.DataFrame(rows),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
