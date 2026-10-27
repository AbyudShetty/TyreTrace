"""Rolling competitors using only historical races and current public inputs.

The ML baseline is unconstrained gradient boosting. The published baseline is
the source-form logarithmic tyre curve with fuel sensitivity fixed to 0.030,
physical construction/season offsets and practice updates to amplitudes. It is
a fitted lap-model baseline, not the complete TUM race simulator.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import polars as pl
import warnings
warnings.filterwarnings('ignore',message='Could not find the number of physical cores')
from sklearn.ensemble import HistGradientBoostingRegressor
from tyretrace.backtest import fp_race as B
from tyretrace.data.compounds import construction_key,nominations

K2=(.02,.05,.1,.2,.35,.5,.75,1.,1.5,2.5)
CONSTRUCTIONS=['unresolved']+[f'{y}:C{i}' for y in range(2023,2027) for i in range(7)]


@dataclass
class Frame:
    weekend: str
    labels: pl.DataFrame
    features: np.ndarray
    response: np.ndarray
    level: np.ndarray
    fuel: np.ndarray
    current_key: np.ndarray
    anchor_keys: np.ndarray
    current_age: np.ndarray
    anchor_ages: np.ndarray
    mapped_fraction: float

    def published_design(self,k2):
        n=len(CONSTRUCTIONS)
        eye=np.eye(n)
        contrast=eye[self.current_key]-eye[self.anchor_keys].mean(axis=1)
        age=np.maximum(self.current_age-1,0)
        anchor_age=np.maximum(self.anchor_ages-1,0)
        curve=(eye[self.current_key]*np.log1p(k2*age)[:,None]
               -(eye[self.anchor_keys]*np.log1p(k2*anchor_age)[:,:,None]).mean(axis=1))
        return np.column_stack([contrast,curve])


def frame(wk,practice,race,prior,traffic,registry=None):
    reg=nominations() if registry is None else registry
    rows=[];features=[];response=[];level=[];fuel=[];keys=[];a_keys=[];ages=[];a_ages=[]
    sl=B.practice_slopes(practice)
    circuit=str(race['circuit'][0]);year=int(wk.key[:4])
    def key(comp):
        value=construction_key(year,circuit,comp,reg)
        return CONSTRUCTIONS.index(value) if value in CONSTRUCTIONS else 0
    for (driver,),g in race.sort('lap_in_session').group_by('driver',maintain_order=True):
        if g.height<=B.ANCHOR_LAPS:continue
        a=g.head(B.ANCHOR_LAPS);t=g.slice(B.ANCHOR_LAPS)
        ac=a['compound_rel'].cast(pl.Utf8).to_list();tc=t['compound_rel'].cast(pl.Utf8).to_list()
        aa=a['tyre_age'].to_numpy().astype(float);ta=t['tyre_age'].to_numpy().astype(float)
        al=float(a['lap_in_session'].mean());lap=t['lap_in_session'].to_numpy().astype(float)
        wake,expected=traffic or (0.,0.)
        anchor_da=float(a['dirty_air'].fill_null(0).mean()) if 'dirty_air' in a.columns else 0.
        lv=float(a['lap_time_s'].mean())-wake*anchor_da+wake*expected
        for i,c in enumerate(tc):
            p,se,_=sl.get(c,(prior[0]-.030*wk.kappa,prior[1],0))
            ap=float(np.mean([sl.get(comp,(prior[0]-.030*wk.kappa,prior[1],0))[0] for comp in ac]))
            dt=lap[i]-al
            features.append([dt/50,ta[i]/20,float(aa.mean())/20, p,ap,se,
                .030*wk.kappa*dt,wk.track_per_lap,wk.lap_s/100,prior[0],prior[1],
                *[float(c==v) for v in ('HARD','MEDIUM','SOFT')],
                *[float(ac[0]==v) for v in ('HARD','MEDIUM','SOFT')],float(c==ac[0])])
            response.append(float(t['lap_time_s'][i])-lv);level.append(lv)
            fuel.append(.030*wk.kappa*dt);keys.append(key(c));a_keys.append([key(v) for v in ac])
            ages.append(ta[i]);a_ages.append(aa)
            rows.append({'weekend':wk.key,'driver':str(driver),'stint':int(t['stint_id'][i]),
                'lap_in_session':int(lap[i]),'tyre_age':int(ta[i]),'compound':c,
                'after_stop':int(t['stint_id'][i])!=int(a['stint_id'][0]),'y':float(t['lap_time_s'][i])})
    return Frame(wk.key,pl.DataFrame(rows),np.asarray(features),np.asarray(response),
        np.asarray(level),np.asarray(fuel),np.asarray(keys,dtype=int),np.asarray(a_keys,dtype=int),
        np.asarray(ages),np.asarray(a_ages),float(np.mean(np.asarray(keys)!=0)))


def _ml():
    return HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=15,min_samples_leaf=40,
        l2_regularization=10.,random_state=17,early_stopping=False)


def ml(history: list[Frame],target: Frame):
    """Noise is estimated from held-out earlier weekends, never training residuals."""
    _check_history(history,target)
    hold=min(3,max(1,len(history)//5));tr=history[:-hold];val=history[-hold:]
    model=_ml().fit(np.vstack([g.features for g in tr]),np.concatenate([g.response for g in tr]))
    noise=np.sqrt(np.mean([np.mean((g.response-model.predict(g.features))**2) for g in val]))
    model=_ml().fit(np.vstack([g.features for g in history]),np.concatenate([g.response for g in history]))
    return target.level+model.predict(target.features),np.full(target.labels.height,max(float(noise),.05)),{'calibration_weekends':len(val)}


def _curve_fit(history,k2,prior):
    n=len(CONSTRUCTIONS)
    X=np.vstack([g.published_design(k2) for g in history])
    y=np.concatenate([g.response+g.fuel for g in history])
    weights=np.concatenate([np.full(g.labels.height,1/g.labels.height) for g in history])
    # Scale equal-weekend weights to one average race, leaving prior strength
    # independent of driver count while allowing additional weekends to inform.
    weights*=np.mean([g.labels.height for g in history])
    derivative=k2/(1+10*k2)
    m0=np.r_[np.zeros(n),np.full(n,prior[0]/derivative)]
    sd=np.r_[np.full(n,1.),np.full(n,max(prior[1],.02)/derivative)]
    precision=np.diag(sd**-2)
    covariance=np.linalg.inv(precision+X.T@(weights[:,None]*X))
    mean=covariance@(precision@m0+X.T@(weights*y))
    return mean,covariance


def published(history: list[Frame],target: Frame,practice: pl.DataFrame,wk,prior,registry=None):
    _check_history(history,target)
    reg=nominations() if registry is None else registry
    hold=min(3,max(1,len(history)//5));tr=history[:-hold];val=history[-hold:]
    choices=[]
    for k2 in K2:
        mean,cov=_curve_fit(tr,k2,prior)
        errors=[g.response+g.fuel-g.published_design(k2)@mean for g in val]
        choices.append((float(np.mean([np.mean(e*e) for e in errors])),k2))
    variance,k2=min(choices)
    mean,cov=_curve_fit(history,k2,prior)
    n=len(CONSTRUCTIONS);circuit=str(target.labels['weekend'][0])
    # Practice is centred within run: unobserved fuel/programme levels cannot
    # masquerade as differences in compound k0. Only curve amplitudes update.
    for _,g in practice.sort('session_uid','driver','stint_id','lap_in_session').group_by('session_uid','driver','stint_id',maintain_order=True):
        if g.height<4:continue
        comp=str(g['compound_rel'][0]);circuit=str(g['circuit'][0])
        construction=construction_key(int(wk.key[:4]),circuit,comp,reg)
        if construction not in CONSTRUCTIONS:continue
        j=n+CONSTRUCTIONS.index(construction)
        x=np.log1p(k2*np.maximum(g['tyre_age'].to_numpy().astype(float)-1,0));x-=x.mean()
        if x@x<1e-8:continue
        y=g['lap_time_s'].to_numpy()+.030*wk.kappa*g['lap_in_session'].to_numpy();y-=y.mean()
        observation=float(x@y/(x@x));residual=y-observation*x
        observation_variance=max(float(residual@residual/max(len(y)-2,1)/(x@x)),.02**2)
        gain=cov[:,j]/(cov[j,j]+observation_variance)
        mean=mean+gain*(observation-mean[j]);cov=cov-np.outer(gain,cov[j,:])
    X=target.published_design(k2)
    prediction=target.level-target.fuel+X@mean
    spread=np.sqrt(max(variance,.05**2)+np.maximum(np.einsum('ij,jk,ik->i',X,cov,X),0))
    return prediction,spread,{'k2':k2,'mapped_fraction':target.mapped_fraction,'noise_history_weekends':hold}


def _check_history(history,target):
    if len(history)<3:raise ValueError('three earlier race frames required')
    keys=[g.weekend for g in history]
    if keys!=sorted(set(keys)) or any(k>=target.weekend for k in keys):
        raise ValueError('history must contain unique chronological strictly earlier weekends')


def conventional_quadratic(target,practice,race,wk,prior,sigma):
    """Protocol conventional family: 0.030 s/kg and a quadratic tyre curve.

    Runs have unknown levels, so centre the design within each run. Compound
    curves are fitted within the target weekend; earlier races supply only the
    common weak slope prior and noise. No race outcome after the anchor enters.
    """
    compounds=('HARD','MEDIUM','SOFT')
    def design(comp,age):
        eye=np.asarray([[float(c==v) for v in compounds] for c in comp])
        a=np.asarray(age,dtype=float)/20
        return np.column_stack([eye*a[:,None],eye*(a*a)[:,None]])
    xx=[];yy=[]
    for _,g in practice.sort('session_uid','driver','stint_id').group_by('session_uid','driver','stint_id',maintain_order=True):
        if g.height<4:continue
        X=design(g['compound_rel'].cast(pl.Utf8),g['tyre_age'])
        y=g['lap_time_s'].to_numpy()+.030*wk.kappa*g['lap_in_session'].to_numpy()
        xx.append(X-X.mean(axis=0));yy.append(y-y.mean())
    m0=np.r_[np.full(3,20*prior[0]),np.zeros(3)]
    precision=np.diag(np.r_[np.full(3,max(20*prior[1],.4)**-2),np.ones(3)])
    X=np.vstack(xx) if xx else np.empty((0,6));y=np.concatenate(yy) if yy else np.empty(0)
    noise=max(float(sigma),.05)
    covariance=np.linalg.inv(precision+X.T@X/noise**2)
    coef=covariance@(precision@m0+X.T@y/noise**2)
    anchor={}
    for (driver,),g in race.sort('driver','lap_in_session').group_by('driver',maintain_order=True):
        a=g.head(B.ANCHOR_LAPS)
        anchor[str(driver)]=design(a['compound_rel'].cast(pl.Utf8),a['tyre_age']).mean(axis=0)
    t=target.labels
    X=design(t['compound'],t['tyre_age'])-np.vstack([anchor[v] for v in t['driver']])
    mean=target.level-target.fuel+X@coef
    spread=np.sqrt(noise**2*(1+1/B.ANCHOR_LAPS)+np.maximum(np.einsum('ij,jk,ik->i',X,covariance,X),0))
    return mean,spread,{'fuel_s_kg':.030,'curve':'within-weekend quadratic','practice_runs':len(xx)}
