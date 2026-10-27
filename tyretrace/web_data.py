"""Small, read-only dashboard datasets exported from canonical laps and audits."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import polars as pl
from tyretrace import artifacts,results
from tyretrace.data import sources
from tyretrace.data.compounds import nominations,resolve
from tyretrace.models import track_evolution as TE
from tyretrace.features import traffic
from tyretrace.infer import uncertainty as U

ROOT=Path(__file__).resolve().parents[1]


def _table(task,name,limit=None):
    t=artifacts.load(task,name)
    if t is None:return []
    return (t.head(limit) if limit else t).to_dicts()


def _calibration_rows(task):
    """Score saved distributions only; exporting never refits a model."""
    t=artifacts.load(task,'predictive' if task=='fresh_holdout' else 'calibrated')
    if t is None:return []
    rows=[]
    for (method,),g in t.group_by('method'):
        for (year,),h in g.with_columns(pl.col('weekend').str.slice(0,4).alias('_year')).group_by('_year'):
            for subset,p in [('all',h),('after_stop',h.filter(pl.col('after_stop')))]:
                if not p.height:continue
                rows.extend({'method':method,'year':year,'subset':subset,'level':level,'coverage':value}
                    for level,value in U.score(p)['calibration'].items())
    return rows


def export(write):
    """Writes static JSON only. No model sampling occurs in the browser."""
    source='lake' if sources.available('lake') else 'audit'
    d=sources.load(source);reg=nominations();race=d['race']
    with_gap=traffic.dirty_air_index(traffic.add_gap_ahead(race))
    index=[]
    for (uid,),g in with_gap.group_by('session_uid'):
        wk=str(uid)[:7];year=int(g['year'][0]);circuit=str(g['circuit'][0])
        label=f'{year} · R{int(g["round"][0]):02d} · {circuit}'
        nomination={c:resolve(year,circuit,c,reg) for c in ('HARD','MEDIUM','SOFT')}
        dry=not g['compound_rel'].cast(pl.Utf8).is_in(['WET','INTERMEDIATE']).any()
        index.append({'weekend':wk,'label':label,'year':year,'circuit':circuit,
                      'laps':g.height,'drivers':g['driver'].n_unique(),'dry':dry,'nomination':nomination})
        cols=['driver','lap_in_session','lap_time_s','tyre_age','compound_rel','stint_id',
              'gap_ahead_s','dirty_air','pit_in','pit_out','is_valid','track_status']
        write(f'races/{wk}.json',{'weekend':wk,'label':label,'nomination':nomination,
              'laps':g.select([c for c in cols if c in g.columns]).sort('driver','lap_in_session').to_dicts()})
    index=sorted(index,key=lambda x:x['weekend'],reverse=True)
    write('races.json',{'source':source,'weekends':index,'race_laps':race.height,
                       'mapped_weekends':sum(all(v for v in x['nomination'].values()) for x in index)})
    track=[]
    for fit in TE.fit(d['quali']):
        row=asdict(fit);row['weekend']=fit.session_uid[:7]
        row['z']=fit.tau_per_min/max(fit.se,1e-12);row['total_s']=fit.total_s
        track.append(row)
    write('track.json',{'source':source,'sessions':track,'nominations':list(reg.values()),
          'fuel_gap':{k:results.get(k).get('metrics',{}) for k in ('fuel_gap','fuel_gap_lake')}})
    posterior=artifacts.load('l4','posterior')
    histograms=[]
    if posterior is not None and posterior.height:
        t=posterior
        for rung,g in t.group_by('rung'):
            for name in ('beta_f','tau_scale','wake_s','sigma','nu','sd_stint'):
                if name not in g.columns:continue
                values=g[name].drop_nulls().to_numpy();values=values[np.isfinite(values)]
                if not len(values):continue
                counts,edges=np.histogram(values,bins=24)
                histograms.append({'rung':rung[0],'parameter':name,'mean':float(values.mean()),
                    'q05':float(np.quantile(values,.05)),'q95':float(np.quantile(values,.95)),
                    'bins':[[float((a+b)/2),int(n)] for a,b,n in zip(edges[:-1],edges[1:],counts)]})
    write('models.json',{'calibration':_table('l4','calibration'),
        'posterior':histograms,'contrasts':_table('l2','degradation'),
        'rolling':results.get('rolling_models').get('metrics',{}),
        'rolling_curves':_table('rolling_models','calibration'),
        'online_eligibility':results.get('online_eligibility').get('metrics',{}),
        'online_curves':_calibration_rows('online_eligibility'),
        'diagnostics':_table('rolling_models','diagnostics')})
    raw=artifacts.load('backtest_lake','predictive')
    if raw is None:raw=artifacts.load('backtest','predictive')
    calibrated=artifacts.load('uncertainty_t','predictive')
    candidate=artifacts.load('rolling_models','calibrated')
    prediction_index=[]
    def prediction_rows(t):
        if t is None:return []
        cols=['driver','lap_in_session','tyre_age','method','compound','stint','y','q05','q50','q95','after_stop']
        return t.select([pl.col(c).round(6) if t.schema[c] in (pl.Float32,pl.Float64) else pl.col(c)
            for c in cols if c in t.columns]).to_dicts()
    if raw is not None:
        for wk in sorted(raw['weekend'].unique().to_list(),reverse=True):
            original=raw.filter((pl.col('weekend')==wk)&(pl.col('method')=='identified'))
            corrected=calibrated.filter(pl.col('weekend')==wk) if calibrated is not None else None
            other=candidate.filter((pl.col('weekend')==wk)&(pl.col('method')!='identified')) if candidate is not None else None
            prediction_index.append({'weekend':wk,'calibrated':corrected is not None and corrected.height>0})
            write(f'predictions/{wk}.json',{'weekend':wk,'original':prediction_rows(original),
                'calibrated':prediction_rows(corrected),'competitors':prediction_rows(other)})
    fresh=artifacts.load('fresh_holdout','predictive');fresh_raw=artifacts.load('fresh_holdout','original')
    if fresh is not None and fresh_raw is not None:
        wk=str(fresh['weekend'][0])
        prediction_index.append({'weekend':wk,'calibrated':True,'fresh':True})
        write(f'predictions/{wk}.json',{'weekend':wk,
            'original':prediction_rows(fresh_raw.filter(pl.col('method')=='identified')),
            'calibrated':prediction_rows(fresh.filter(pl.col('method')=='identified')),
            'competitors':prediction_rows(fresh.filter(pl.col('method')!='identified'))})
    write('predictions.json',{'weekends':sorted(prediction_index,key=lambda r:r['weekend'],reverse=True)})
    write('evaluation.json',{'backtest':results.get('backtest_lake').get('metrics',{}),
        'weekends':_table('backtest_lake','weekends'),'ablations':_table('ablations','table'),
        'postrace':results.get('postrace_lake').get('metrics',{}),
        'strategy':results.get('strategy_reference').get('metrics',{}),
        'rolling':results.get('rolling_models').get('metrics',{}),
        'fresh_holdout':{**results.get('fresh_holdout').get('metrics',{}),
                         'audit_gate':results.get('fresh_holdout').get('gate')},
        'fresh_curves':_calibration_rows('fresh_holdout'),
        'headline':results.get('headline_2026').get('metrics',{})})
    reports=[]
    for p in sorted((ROOT/'out/rig').glob('*.report.json')):
        try:
            value=json.loads(p.read_text(encoding='utf-8'));value['file']=p.name
            # Per-car fit diagnostics are available as a table, not discarded.
            reports.append(value)
        except (OSError,ValueError):continue
    fake=ROOT/'out/rig_fake.parquet'
    example_origin='synthetic'
    if reports:
        latest=reports[-1]
        candidate=ROOT/'out/rig'/latest['file'].replace('.report.json','.parquet')
        if candidate.exists():fake=candidate;example_origin=latest['origin']
    rig_laps=pl.read_parquet(fake).to_dicts() if fake.exists() else []
    write('rig.json',{'reports':reports,'example_origin':example_origin,'laps':rig_laps,
        'live_established':any(r.get('tier2_established') for r in reports)})
    return len(index)
