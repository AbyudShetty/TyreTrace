"""Isolated unseen Austin holdout: freeze, fetch, then score without retuning.

--fetch is an explicit network operation. Default mode reads saved files only.
The original lake is never modified. Changing a frozen scientific source or
history input after downloading the holdout invalidates the experiment.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import sys
import numpy as np
import polars as pl
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tyretrace import artifacts,results
from tyretrace.backtest import fp_race as B,competitors as C
from tyretrace.backtest.eligibility import race_laps
from tyretrace.data import sources,openf1 as O
from tyretrace.data.compounds import nominations
from tyretrace.infer import uncertainty as U
from tyretrace.eval import metrics

DEST=ROOT/'out/fresh_holdout'
METHODS=('identified','naive','conventional_quadratic','ml','published')
COLS=['weekend','driver','stint','lap_in_session','tyre_age','after_stop','y','q05','q50','q95','crps','method']
FROZEN=['tyretrace/backtest/fp_race.py','tyretrace/backtest/competitors.py',
        'tyretrace/backtest/eligibility.py','tyretrace/infer/uncertainty.py',
        'tyretrace/models/practice.py','tyretrace/models/regimes.py',
        'tyretrace/data/nominations.json','audit/m2_online_eligibility.py',
        'audit/m3_fresh_holdout.py','eval/protocol.md',
        'out/lake/practice.parquet','out/lake/quali.parquet','out/lake/race.parquet']


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(frozen):
    changed=[p for p,h in frozen['sha256'].items() if digest(ROOT/p)!=h]
    if changed:raise ValueError('frozen experiment changed after holdout access: '+', '.join(changed))


def fetch():
    DEST.mkdir(parents=True,exist_ok=True)
    frozen={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'target':'2026_18','location':'Austin','meeting_key':1297,
        'selected_method':'ml','primary_2026_tank_kg':70,
        'eligibility':'flag-only green valid non-pit laps after lap one',
        'inputs':'earlier weekends, current practice/quali, five opening race anchors; observed strategy/distance',
        'criterion':'ML lower CRPS than every comparator and cov90 within [0.85,0.95]',
        'scope':'one previously unseen offline race weekend; event predates algorithm freeze',
        'sha256':{p:digest(ROOT/p) for p in FROZEN}}
    with (DEST/'freeze.json').open('x',encoding='utf-8') as f:json.dump(frozen,f,indent=2)
    # New API cache, avoiding a stale sessions listing or mutation of the lake.
    O.CACHE=DEST/'api_cache'
    sessions=O.get('sessions',meeting_key=1297)
    tables={'practice':[],'quali':[],'race':[]}
    for s in sessions:
        code=O.CODES.get(s.get('session_name'))
        if not code or s.get('is_cancelled'):continue
        meta={'uid':f'2026_18_{code}','year':2026,'round':18,'code':code,
              'session_key':s['session_key'],'location':'Austin'}
        laps=O.load_session(meta)
        group='practice' if code.startswith('FP') else 'quali' if code=='Q' else 'race'
        if laps.height:tables[group].append(laps)
        print(f'{code}: {laps.height} laps',flush=True)
    for name,frames in tables.items():
        if not frames:raise ValueError(f'holdout lacks {name} data')
        pl.concat(frames).write_parquet(DEST/f'{name}.parquet')
    manifest={'freeze_sha256':digest(DEST/'freeze.json'),
        'data_sha256':{n:digest(DEST/f'{n}.parquet') for n in tables},
        'scope':frozen['scope']}
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Isolated holdout fetched. No scores inspected.',flush=True)


def score():
    frozen=json.loads((DEST/'freeze.json').read_text());verify(frozen)
    manifest=json.loads((DEST/'manifest.json').read_text())
    if digest(DEST/'freeze.json')!=manifest['freeze_sha256']:raise ValueError('freeze manifest changed')
    for name,h in manifest['data_sha256'].items():
        if digest(DEST/f'{name}.parquet')!=h:raise ValueError('holdout data changed')
    d=sources.load('lake');practice,_=B.prepare(d['practice'],d['race'])
    race=race_laps(d['race']);wk=B.weekend_inputs(d['quali'],d['race'])
    dry=B.dry_weekends(d['race']);keys=sorted(k for k in wk if k in dry and k<frozen['target'])
    rc={k:B.race_lap_coef(race.filter(pl.col('weekend')==k)) for k in keys}
    rd={k:B.race_degradation(race.filter(pl.col('weekend')==k)) for k in keys}
    tf=B.traffic_history(race,keys);reg=nominations();history=[]
    for i,k in enumerate(keys):
        prior=B.degradation_prior([rd[e] for e in keys[:i]])
        traffic=B.traffic_prior([tf[e] for e in keys[:i]])
        history.append(C.frame(wk[k],practice.filter(pl.col('weekend')==k),race.filter(pl.col('weekend')==k),prior,traffic,reg))
    new={n:pl.read_parquet(DEST/f'{n}.parquet') for n in ('practice','quali','race')}
    if not B.dry_weekends(new['race']):raise ValueError('holdout is wet; frozen dry-race experiment not applicable')
    p,_=B.prepare(new['practice'],new['race']);r=race_laps(new['race']);k=frozen['target']
    w=B.weekend_inputs(new['quali'],new['race'])[k]
    prior=B.degradation_prior([rd[e] for e in keys]);traffic=B.traffic_prior([tf[e] for e in keys])
    ft=B.fit_fuel_track([rc[e][0] for e in keys],[rc[e][1] for e in keys],[wk[e] for e in keys])
    sigma=float(np.median([rc[e][2] for e in keys]));target=C.frame(w,p,r,prior,traffic,reg)
    tabs=[]
    for method,fc in B.run_weekend(w,p,r,ft,sigma,seed=73,prior=prior,traffic=traffic,methods=('identified','naive')).items():
        t=U.from_forecast(fc,weekend=k,method=method)
        tabs.append(t.with_columns(pl.Series('crps',metrics.crps_ensemble(fc.draws,t['y'].to_numpy()))))
    for method in ('ml','published','conventional_quadratic'):
        if method=='ml':mean,sd,_=C.ml(history,target)
        elif method=='published':mean,sd,_=C.published(history,target,p,w,prior,reg)
        else:mean,sd,_=C.conventional_quadratic(target,p,r,w,prior,sigma)
        draws=mean[None,:]+sd[None,:]*np.random.default_rng(73).standard_normal((1000,len(mean)))
        t=artifacts.predictive(draws,target.labels['y'].to_numpy(),**{c:target.labels[c].to_numpy() for c in target.labels.columns if c!='y'})
        tabs.append(t.with_columns(pl.lit(method).alias('method')))
    raw=pl.concat([t.select(COLS) for t in tabs],how='vertical_relaxed')
    artifacts.save('fresh_holdout','original',raw)
    old=artifacts.load('online_eligibility','predictive')
    if old is None:raise ValueError('run online_eligibility first')
    calibrated=[];rows=[]
    for method in METHODS:
        hist=old.filter((pl.col('method')==method)&(pl.col('weekend')<k))
        original=raw.filter(pl.col('method')==method)
        model=U.PredictiveUncertainty.fit(hist,target=k)
        t=model.predict(original.drop('y'),target=k).with_columns(original['y'])
        t=t.with_columns(pl.Series('crps',U.student_crps(t['mean'].to_numpy(),t['scale'].to_numpy(),t['y'].to_numpy())))
        calibrated.append(t)
        for subset,g in [('all',t),('after_stop',t.filter(pl.col('after_stop')))]:
            score=U.score(g);rows.append({'method':method,'subset':subset,**{a:b for a,b in score.items() if a!='calibration'}})
    calibrated=pl.concat(calibrated,how='diagonal_relaxed');identity=['weekend','driver','stint','lap_in_session']
    reference=calibrated.filter(pl.col('method')=='identified').sort(identity)
    for method in METHODS:
        t=calibrated.filter(pl.col('method')==method).sort(identity)
        if t.select(identity).to_dicts()!=reference.select(identity).to_dicts() or not np.array_equal(t['y'],reference['y']):raise ValueError('holdout target mismatch')
    artifacts.save('fresh_holdout','predictive',calibrated)
    artifacts.save('fresh_holdout','summary',pl.DataFrame(rows))
    a=next(r for r in rows if r['method']=='ml' and r['subset']=='all')
    passes=a['crps']<min(r['crps'] for r in rows if r['method']!='ml' and r['subset']=='all') and .85<=a['coverage_90']<=.95
    results.record('fresh_holdout',{'summary':rows,'selected_method':'ml','model_gate':'SHIP' if passes else 'HOLD',
        'freeze':frozen,'manifest':manifest,'n_independent_weekends':1,
        'scope':frozen['scope']},gate='PASS')
    print(json.dumps({'scores':rows,'model_gate':'SHIP' if passes else 'HOLD'},indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--fetch',action='store_true');args=ap.parse_args()
    fetch() if args.fetch else score()
    return 0

if __name__=='__main__':raise SystemExit(main())
