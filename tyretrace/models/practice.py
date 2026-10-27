"""Practice-only robust slopes and HMM-adjusted candidates.

All outcomes consumed here are observed practice laps. Race results are never
accepted. No cross-weekend pooling on relative compound labels is performed.
"""
from __future__ import annotations
import numpy as np
import polars as pl
from tyretrace.backtest.fp_race import random_effects
from tyretrace.models.regimes import select_regimes


def robust_line(lap,y,*,iterations=30):
    x=np.asarray(lap,dtype=float);y=np.asarray(y,dtype=float)
    if len(y)<4 or not np.all(np.isfinite(y)) or np.ptp(x)==0:
        raise ValueError('at least four finite laps with varying age required')
    x=x-x.mean();X=np.column_stack([np.ones(len(x)),x])
    coef=np.linalg.lstsq(X,y,rcond=None)[0]
    for _ in range(iterations):
        r=y-X@coef
        scale=max(1.4826*float(np.median(np.abs(r-np.median(r)))),.02)
        w=5/(4+(r/scale)**2)
        new=np.linalg.lstsq(X*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
        if np.max(np.abs(new-coef))<1e-8:coef=new;break
        coef=new
    # Lag-one sandwich covariance: adjacent laps are not independent evidence.
    bread=np.linalg.pinv(X.T@(w[:,None]*X));score=X*(w*(y-X@coef))[:,None]
    meat=score.T@score
    adjacent=score[1:].T@score[:-1]
    meat+=.5*(adjacent+adjacent.T)
    cov=bread@meat@bread*len(y)/max(len(y)-2,1)
    se=max(float(np.sqrt(max(cov[1,1],0))),1e-4)
    return float(coef[1]),se,float(coef[0]),y-X@coef


def slopes(practice: pl.DataFrame,*,regimes=False):
    runs=[]
    for _,g in practice.sort('session_uid','driver','stint_id','lap_in_run').group_by('session_uid','driver','stint_id',maintain_order=True):
        g=g.sort('lap_in_run')
        if g.height<4:continue
        x=g['lap_in_run'].to_numpy().astype(float);y=g['lap_time_s'].to_numpy()
        try:m,se,intercept,residual=robust_line(x,y)
        except ValueError:continue
        runs.append({'compound':str(g['compound_rel'][0]),'x':x,'y':y,'m':m,'se':se,'residual':residual})
    diagnostic={'runs':len(runs),'regimes_used':False}
    if regimes:
        eligible=[r['residual'] for r in runs if len(r['x'])>=8]
        if len(eligible)>=3:
            model,scores=select_regimes(eligible)
            diagnostic.update(regimes_used=True,states=len(model.means),
                heldout_log_likelihood={str(k):v for k,v in scores.items()},
                means=model.means.tolist(),converged=model.converged,iterations=model.iterations)
            for r in runs:
                expected=model.posterior(r['residual'],smooth=True)@model.means
                r['m'],r['se'],_,_=robust_line(r['x'],r['y']-expected)
    out={}
    for c in sorted({r['compound'] for r in runs}):
        chosen=[r for r in runs if r['compound']==c]
        m,se=random_effects(np.array([r['m'] for r in chosen]),np.array([r['se'] for r in chosen]))
        out[c]=(m,max(se,1e-4),len(chosen))
    return out,diagnostic
