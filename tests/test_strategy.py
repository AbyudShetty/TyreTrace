"""Check actual decisions against explicit lap-by-lap counterfactual costs."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tyretrace.eval.strategy import costs,optimal_one_stop,decision_error,stop_distribution


@pytest.mark.parametrize('first,second',[(.08,.04),(.03,.07),(0.,0.),(-.01,.04)])
def test_closed_form_costs_equal_explicit_lap_simulation(first,second):
    laps,values=costs(60,first,second,first_set_age=3,second_set_age=1)
    explicit=[sum(first*(k+3) for k in range(1,p+1))
              +sum(second*(k+1) for k in range(1,60-p+1))+20 for p in laps]
    assert np.allclose(values,explicit)
    assert optimal_one_stop(60,first,second,first_set_age=3,second_set_age=1).lap==laps[np.argmin(explicit)]


def test_known_optimum_and_wrong_fuel_assumption_change_the_decision():
    assert optimal_one_stop(60,.08,.04).lap==20
    right=decision_error(60,(.08,.04),(.08,.04))
    wrong=decision_error(60,(.107,.067),(.08,.04))
    assert right['stop_lap_error']==0 and right['regret_s']==0
    assert wrong['stop_lap_error']>=3 and wrong['regret_s']>0


def test_pit_loss_constant_cannot_change_lap_when_one_stop_is_required():
    assert optimal_one_stop(60,.08,.04,pit_loss_s=10).lap==optimal_one_stop(60,.08,.04,pit_loss_s=30).lap


def test_ties_choose_the_first_legal_stop_and_respect_minimum_stints():
    assert optimal_one_stop(20,0,0,min_stint=4).lap==4
    assert optimal_one_stop(20,100,0,min_stint=4).lap==4


def test_joint_draws_reproduce_exact_two_point_decision_distribution():
    draws=np.array([[.08,.04]]*60+[[.04,.08]]*40)
    d=stop_distribution(60,draws)
    assert d['probability']=={20:.6,40:.4}
    assert d['interval90']==[20.,40.]


@pytest.mark.parametrize('kwargs',[{'n_laps':5},{'min_stint':0},{'first_set_age':-1},{'pit_loss_s':-1}])
def test_invalid_decision_domains_fail_loudly(kwargs):
    args={'n_laps':60,'first':.05,'second':.03,**kwargs}
    with pytest.raises(ValueError):costs(**args)
