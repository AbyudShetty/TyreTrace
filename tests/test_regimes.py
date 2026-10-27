import itertools
import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.special import logsumexp

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tyretrace.models.regimes import PaceHMM,select_regimes
from tyretrace.models.practice import robust_line,slopes


def test_forward_likelihood_matches_exhaustive_hidden_paths():
    m=PaceHMM(np.array([0.,1.]),np.array([.1,.2]),np.array([[.8,.2],[0,1]]),np.array([1.,0.]))
    y=np.array([.1,.2,.8,1.2]);weights=[]
    for path in itertools.product(range(2),repeat=len(y)):
        probability=m.initial[path[0]]
        for a,b in zip(path[:-1],path[1:]):probability*=m.transition[a,b]
        density=np.exp(-.5*(y-m.means[list(path)])**2/m.variances[list(path)])/np.sqrt(2*np.pi*m.variances[list(path)])
        weights.append(probability*density.prod())
    assert m.log_likelihood(y)==pytest.approx(np.log(sum(weights)),abs=1e-10)


def test_filtering_cannot_see_future_observations():
    m=PaceHMM(np.array([.4,0.,1.]),np.full(3,.1),np.array([[.8,.2,0],[0,.95,.05],[0,0,1.]]),np.array([1.,0.,0.]))
    a=m.posterior([.4,.3,0.,0.,.9,1.])
    b=m.posterior([.4,.3,0.,0.,100.,100.])
    assert np.array_equal(a[:4],b[:4])
    assert np.allclose(a.sum(axis=1),1)
    p,mean,var=m.future(a[3],10)
    assert np.allclose(p.sum(axis=1),1) and np.all(var>=0)


def test_em_and_state_selection_identify_staged_sequences():
    rng=np.random.default_rng(71)
    sequences=[np.r_[rng.normal(.8,.08,4),rng.normal(0,.08,14),rng.normal(1.3,.08,8)] for _ in range(10)]
    fitted,scores=select_regimes(sequences,max_iter=40)
    assert len(fitted.means)==3 and scores[3]>scores[2]
    probabilities=fitted.posterior(sequences[0],smooth=True)
    assert probabilities[:3,0].mean()>.8 and probabilities[7:15,1].mean()>.8
    assert probabilities[-5:,2].mean()>.8


def test_robust_slope_resists_a_compromised_lap():
    x=np.arange(1,21,dtype=float);y=90+.05*x
    y[12]+=8
    m,se,_,_=robust_line(x,y)
    assert abs(m-.05)<.005 and se>0
    assert abs(np.polyfit(x,y,1)[0]-.05)>abs(m-.05)*5


def test_practice_candidates_recover_slopes_with_arbitrary_run_levels():
    from test_backtest import practice,DEG,FUEL_PACE,TRACK
    table=practice(seed=11,sigma=.01)
    recovered,diag=slopes(table)
    for c in DEG:assert abs(recovered[c][0]-(DEG[c]-FUEL_PACE+TRACK))<.005
    assert diag['runs']==8


def test_backward_regime_transitions_are_rejected():
    with pytest.raises(ValueError,match='forward'):
        PaceHMM(np.array([0.,1.]),np.ones(2),np.array([[.8,.2],[.2,.8]]),np.array([1.,0.]))
