"""Development only: inspect 2023–2025 residual variance; never loads 2026."""
from pathlib import Path
import sys

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def crps(mu, sd, y):
    z = (y - mu) / sd
    return sd * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-z*z/2)/np.sqrt(2*np.pi)
                 - 1/np.sqrt(np.pi))


def arrays(t):
    sd = (t['q95'].to_numpy() - t['q05'].to_numpy()) / 3.2897072539
    lap = t['lap_in_session'].to_numpy()/50
    age = t['tyre_age'].to_numpy()/20
    x = np.column_stack([np.ones(t.height), lap, age])
    v = np.column_stack([np.ones(t.height), sd**2, lap**2, age**2])
    return x, v, t['q50'].to_numpy(), t['y'].to_numpy()


def fit(t, bias=False):
    x, v, mu, y = arrays(t)
    ids=t['weekend'].to_numpy()
    _, inv, counts=np.unique(ids,return_inverse=True,return_counts=True)
    w=1/counts[inv];w/=w.sum()
    def loss(p):
        m=mu+(x@p[4:] if bias else 0)
        s=np.sqrt(np.maximum(v@p[:4], 1e-8))
        z=(y-m)/s
        return np.sum(w*(np.log(s)+z*z/2))
    p=np.array([.1,1.,.1,.1]+([0.,0.,0.] if bias else []))
    res=minimize(loss,p,bounds=[(1e-6,None)]*4+([(None,None)]*3 if bias else []))
    return res.x


def run():
    t=(pl.scan_parquet(ROOT/'out/artifacts/backtest_lake/predictive.parquet')
       .filter((pl.col('weekend')<'2026')&(pl.col('method')=='identified')).collect())
    for bias in [False,True]:
        tr=t.filter(pl.col('weekend')<'2025')
        p=fit(tr,bias)
        print('bias',bias,'parameters',p,flush=True)
        for label,d in [('train',tr),('validation',t.filter(pl.col('weekend')>='2025'))]:
            x,v,mu,y=arrays(d)
            m=mu+(x@p[4:] if bias else 0)
            s=np.sqrt(v@p[:4]);aft=d['after_stop'].to_numpy()
            print(label,'n',len(y),'CRPS',crps(m,s,y).mean(),'base',d['crps'].mean(),
                  'coverage',np.mean(np.abs(y-m)<1.6448536*s),
                  'after',np.mean((np.abs(y-m)<1.6448536*s)[aft]),flush=True)
    rows=[]
    for k in sorted(t['weekend'].unique().to_list()):
        if k<'2025':continue
        p=fit(t.filter(pl.col('weekend')<k),False)
        d=t.filter(pl.col('weekend')==k)
        _,v,mu,y=arrays(d);s=np.sqrt(v@p)
        rows.append(d.select('weekend','after_stop','crps').with_columns(
            pl.Series('new',crps(mu,s,y)),pl.Series('inside',np.abs(y-mu)<1.6448536*s)))
    z=pl.concat(rows)
    print('ROLLING VALIDATION',z.select(pl.col('new').mean(),pl.col('crps').mean(),pl.col('inside').mean()),
          z.filter(pl.col('after_stop')).select(pl.col('inside').mean()),flush=True)


if __name__=='__main__':run()
