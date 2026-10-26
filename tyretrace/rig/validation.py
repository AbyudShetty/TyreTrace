"""Blind estimator versus a fuel-observed reference, with explicit proof scope.

Wear percentage and degradation in seconds per lap are different quantities.
This report never pretends that a time-slope fit recovered hidden wear percent.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def _fit(x,y):
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return None
    if np.linalg.matrix_rank(x)<x.shape[1]:
        return None
    beta,*_=np.linalg.lstsq(x,y,rcond=None)
    residual=y-x@beta
    cov=np.linalg.pinv(x.T@x)*float(residual@residual)/max(len(y)-x.shape[1],1)
    return beta,np.sqrt(np.maximum(np.diag(cov),0))


def blind_estimate(observable: pl.DataFrame) -> dict:
    if any(c.startswith('truth_') for c in observable.columns):
        raise ValueError('truth channels must be stripped before blind estimation')
    x=np.column_stack([np.ones(observable.height),observable['lap_in_session'].to_numpy(),
                       observable['tyre_age'].to_numpy()])
    fit=_fit(x,observable['lap_time_s'].to_numpy())
    if fit is None:
        raise ValueError('pit-stop age resets are needed to identify the two slopes')
    coef,se=fit
    return {'deg':float(coef[2]),'deg_se':float(se[2]),
            'fuel_pace':float(-coef[1]),'fuel_pace_se':float(se[1])}


def recovery_report(laps: pl.DataFrame,*,origin: str) -> dict:
    # Fit each car separately: driver differences must not look like fuel mass.
    clean=laps.filter(pl.col('is_valid') & ~pl.col('pit_in') & ~pl.col('pit_out'))
    rows=[]
    for (driver,),g in clean.group_by('driver'):
        if g.height<15 or g['stint_id'].n_unique()<2:
            continue
        if any(g[c].null_count() for c in ('truth_fuel_kg','truth_wear_fl','truth_wear_fr','truth_wear_rl','truth_wear_rr')):
            continue
        obs=g.drop([c for c in g.columns if c.startswith('truth_')])
        try:
            blind=blind_estimate(obs)
        except ValueError:
            continue
        fuel=g['truth_fuel_kg'].to_numpy()
        age=g['tyre_age'].to_numpy().astype(float)
        y=g['lap_time_s'].to_numpy()
        fit=_fit(np.column_stack([np.ones(g.height),fuel,age]),y)
        if fit is None:
            continue
        coef,se=fit
        burn=_fit(np.column_stack([np.ones(g.height),g['lap_in_session'].to_numpy()]),fuel)
        kappa=-float(burn[0][1])
        rows.append({'driver':driver,**blind,'fuel_observed_deg':float(coef[2]),
            'fuel_observed_deg_se':float(se[2]),'fuel_observed_pace':float(coef[1]*kappa),
            'beta_f_reference':float(coef[1]),'measured_burn_kg_lap':kappa})
    if not rows:
        return {'gate':'HOLD','cars':0,'tier2_established':False,
                'reason':'need at least 15 valid laps, measured fuel/wear and a pit-stop age reset per car'}
    deg_error=np.array([r['deg']-r['fuel_observed_deg'] for r in rows])
    pace_error=np.array([r['fuel_pace']-r['fuel_observed_pace'] for r in rows])
    # Descriptive consistency only: reference and blind errors are correlated.
    return {'gate':'SYNTHETIC_ONLY' if origin=='synthetic' else 'REFERENCE_COMPARISON',
        'cars':len(rows),'per_car':rows,'deg_difference_mae':float(np.abs(deg_error).mean()),
        'fuel_pace_difference_mae':float(np.abs(pace_error).mean()),
        'tier2_established':False,
        'scope':'blind lap-time estimator versus fuel-conditioned reference; wear-percent state recovery is not established',
        'warning':'linear fake burn makes this comparison algebraically circular' if origin=='synthetic'
                  else 'fuel-conditioned fit is a reference, not observed true degradation in seconds'}
