"""Conditional one-stop decisions and regret, with an explicit reference.

Fuel/track terms common to every candidate stop cancel from this decision.
This evaluates two supplied tyre curves and a fixed compound order; it does not
infer an unobserved optimal race strategy from actual pit-stop timing.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OneStop:
    lap: int
    tyre_and_pit_time: float


def costs(n_laps: int, first: float, second: float, *, pit_loss_s: float=20.,
          min_stint: int=3, first_set_age: int=0, second_set_age: int=0):
    """Cost of each legal stop after lap p for linear age-indexed tyre curves.

    Ages include prior scrub; race/stint lengths use completed laps, so a stop
    after p gives p laps on the first set and n_laps-p on the second. Exactly
    one stop is required. Its fixed pit loss cannot affect which lap is best.
    """
    ints=(n_laps,min_stint,first_set_age,second_set_age)
    if any(isinstance(v,bool) or not isinstance(v,(int,np.integer)) for v in ints):
        raise ValueError('race length, minimum stint and set ages must be integers')
    if min_stint<1 or n_laps<2*min_stint or min(first_set_age,second_set_age)<0:
        raise ValueError('race must accommodate two legal stints and nonnegative set ages')
    if not np.all(np.isfinite([first,second,pit_loss_s])) or pit_loss_s<0:
        raise ValueError('finite tyre rates and nonnegative pit loss are required')
    stops=np.arange(min_stint,n_laps-min_stint+1,dtype=int)
    after=n_laps-stops
    total=(first*(stops*(stops+1)/2+first_set_age*stops)
           +second*(after*(after+1)/2+second_set_age*after)+pit_loss_s)
    return stops,total


def optimal_one_stop(n_laps: int,first: float,second: float,**kwargs) -> OneStop:
    laps,total=costs(n_laps,first,second,**kwargs)
    # Earliest optimal lap is the deterministic tie-break.
    best=int(np.argmin(total))
    return OneStop(int(laps[best]),float(total[best]))


def decision_error(n_laps: int,forecast: tuple[float,float],
                   reference: tuple[float,float],**kwargs) -> dict:
    """Stop-lap error and time regret under the SUPPLIED reference curves."""
    predicted=optimal_one_stop(n_laps,*forecast,**kwargs)
    oracle=optimal_one_stop(n_laps,*reference,**kwargs)
    laps,truth_cost=costs(n_laps,*reference,**kwargs)
    chosen=float(truth_cost[np.flatnonzero(laps==predicted.lap)[0]])
    return {'forecast_stop':predicted.lap,'reference_stop':oracle.lap,
            'stop_lap_error':abs(predicted.lap-oracle.lap),
            'regret_s':max(chosen-oracle.tyre_and_pit_time,0.)}


def stop_distribution(n_laps: int,joint_degradation_draws: np.ndarray,**kwargs) -> dict:
    """Decision distribution from JOINT draws supplied by the caller.

    Marginal calibrated lap intervals cannot be used as independent race paths
    here. The caller must supply paired compound-parameter draws that retain
    their posterior dependence.
    """
    a=np.asarray(joint_degradation_draws,dtype=float)
    if a.ndim!=2 or a.shape[1]!=2 or not len(a) or not np.all(np.isfinite(a)):
        raise ValueError('finite joint parameter draws of shape (draws,2) are required')
    stops=np.array([optimal_one_stop(n_laps,*row,**kwargs).lap for row in a])
    values,counts=np.unique(stops,return_counts=True)
    return {'median':float(np.median(stops)),
            'interval90':np.quantile(stops,[.05,.95]).tolist(),
            'probability':{int(k):float(v/len(stops)) for k,v in zip(values,counts)}}
