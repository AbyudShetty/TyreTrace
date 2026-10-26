"""Predictive coverage, proper scores and temporal isolation on known truth."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy.integrate import quad
from scipy.special import stdtr, stdtrit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tyretrace.infer import uncertainty as U
from tyretrace.eval import metrics


def generated(seed=33, n_weekends=24, n_laps=300):
    """Predictive discrepancies increase with race horizon and tyre age.

    Shared weekend scale shifts keep this from being an independent-lap-only
    coverage test. The original forecast knows the mean but misses dispersion.
    """
    rng = np.random.default_rng(seed)
    tabs = []
    for i in range(n_weekends):
        lap = rng.integers(8, 70, n_laps)
        age = rng.integers(1, 32, n_laps)
        raw_sd = rng.uniform(0.15, 0.45, n_laps)
        scale = np.sqrt(0.02 + .7*raw_sd**2 + .25*(lap/50)**2 + .1*(age/20)**2)
        scale *= np.exp(rng.normal(0, .10))
        y = 90 + scale*rng.standard_t(4, n_laps)
        tabs.append(pl.DataFrame({"weekend": [f"2024_{i+1:02d}"]*n_laps,
            "method": ["identified"]*n_laps, "lap_in_session": lap, "tyre_age": age,
            "q05": 90-1.64485362695*raw_sd, "q50": np.full(n_laps,90.),
            "q95": 90+1.64485362695*raw_sd, "y": y,
            "after_stop": lap>25}))
    return pl.concat(tabs)


@pytest.mark.parametrize("df", [2.5, 4., 12.])
@pytest.mark.parametrize("y", [-2., 0., 3.])
def test_student_crps_matches_independent_cdf_integral(df, y):
    # Definition of CRPS, independently of the closed-form implementation.
    numeric = (quad(lambda x: stdtr(df,x)**2, -np.inf, y)[0]
               + quad(lambda x: (stdtr(df,x)-1)**2, y, np.inf)[0])
    actual = U.student_crps(np.array([0.]), np.array([1.]), np.array([y]), df)[0]
    assert actual == pytest.approx(numeric, abs=1e-8)


def test_student_crps_matches_draws_and_obeys_scale_equivariance():
    rng = np.random.default_rng(28)
    draws = rng.standard_t(4, (100000, 3))
    y = np.array([0., 1., 3.])
    exact = U.student_crps(np.zeros(3), np.ones(3), y)
    assert np.allclose(metrics.crps_ensemble(draws, y), exact, atol=.015)
    assert np.allclose(U.student_crps(np.full(3,7.), np.full(3,2.), 7+2*y), 2*exact)


def test_new_weekends_recover_coverage_and_improve_the_proper_score():
    t = generated()
    hist, target = t.filter(pl.col('weekend')<'2024_17'), t.filter(pl.col('weekend')>='2024_17')
    model = U.PredictiveUncertainty.fit(hist, target='2024_17')
    scores, raw_scores, covered = [], [], []
    for k in sorted(target['weekend'].unique().to_list()):
        g=target.filter(pl.col('weekend')==k)
        pred=model.predict(g, target=k)
        scores.append(U.score(pred)['crps'])
        covered.append(U.score(pred)['coverage_90'])
        raw_sd=(g['q95'].to_numpy()-g['q05'].to_numpy())/U.Z90
        raw_scores.append(U.normal_crps(g['q50'].to_numpy(),raw_sd,g['y'].to_numpy()).mean())
        # The point estimate is preserved, and Student-t scale is NOT sigma.
        assert np.array_equal(pred['q50'].to_numpy(),g['q50'].to_numpy())
        assert np.allclose(pred['sd'].to_numpy(),np.sqrt(2)*pred['scale'].to_numpy())
    assert .85 <= np.mean(covered) <= .95
    assert np.mean(scores) < np.mean(raw_scores)


def test_fit_rejects_target_and_future_outcomes_in_history():
    t=generated(n_weekends=10)
    with pytest.raises(ValueError,match='precede'):
        U.PredictiveUncertainty.fit(t,target='2024_10')


def test_prediction_does_not_depend_on_target_outcomes_or_row_order():
    t=generated(n_weekends=10)
    h=t.filter(pl.col('weekend')<'2024_10')
    g=t.filter(pl.col('weekend')=='2024_10')
    model=U.PredictiveUncertainty.fit(h,target='2024_10')
    clean=model.predict(g.drop('y'),target='2024_10')
    poisoned=model.predict(g.with_columns(pl.lit(1e9).alias('y')),target='2024_10')
    assert np.array_equal(clean['q95'].to_numpy(),poisoned['q95'].to_numpy())
    assert np.array_equal(clean['q50'].to_numpy(),poisoned['q50'].to_numpy())


def test_a_thousand_laps_are_not_a_thousand_independent_weekends():
    t=generated(n_weekends=7)
    with pytest.raises(ValueError,match='insufficient history'):
        U.PredictiveUncertainty.fit(t,target='2025_01')
    with pytest.raises(ValueError,match='at least'):
        U.PredictiveUncertainty.fit(t,target='2025_01',min_weekends=2)


def test_weekends_keep_equal_weight_when_one_race_is_duplicated():
    t=generated(n_weekends=10)
    duplicated=pl.concat([t,t.filter(pl.col('weekend')=='2024_03')])
    a=U.PredictiveUncertainty.fit(t,target='2025_01')
    b=U.PredictiveUncertainty.fit(duplicated,target='2025_01')
    assert np.allclose(a.coefficients,b.coefficients,rtol=1e-5,atol=1e-7)


def test_future_prediction_rows_cannot_change_earlier_rolling_calibration():
    t=generated(n_weekends=12)
    a,_=U.rolling(t,score_from='2024_09')
    poisoned=t.with_columns(pl.when(pl.col('weekend')>='2024_11')
        .then(1e5).otherwise(pl.col('y')).alias('y'))
    b,_=U.rolling(poisoned,score_from='2024_09',score_until='2024_11')
    a=a.filter(pl.col('weekend')<'2024_11')
    assert np.array_equal(a['q95'].to_numpy(),b['q95'].to_numpy())


def test_fit_rejects_mixed_methods_and_malformed_intervals():
    t=generated(n_weekends=10)
    bad=t.with_columns(pl.when(pl.col('weekend')=='2024_01').then(pl.lit('other'))
                      .otherwise(pl.col('method')).alias('method'))
    with pytest.raises(ValueError,match='separately'):
        U.PredictiveUncertainty.fit(bad,target='2025_01')
    with pytest.raises(ValueError,match='ordered'):
        U.PredictiveUncertainty.fit(t.with_columns(pl.lit(0.).alias('q95')),target='2025_01')


def test_reported_student_intervals_have_the_stated_probability():
    model=U.PredictiveUncertainty((.01,1.,.1,.1),'2024_10',10,3000)
    g=generated(n_weekends=1).with_columns(pl.lit('2025_01').alias('weekend'))
    p=model.predict(g,target='2025_01')
    z=(p['q95'].to_numpy()-p['mean'].to_numpy())/p['scale'].to_numpy()
    assert np.allclose(stdtr(4,z),.95)
    assert np.all(p['q025'].to_numpy()<p['q05'].to_numpy())
    assert np.allclose(p['q975'].to_numpy(),p['mean'].to_numpy()+stdtrit(4,.975)*p['scale'].to_numpy())


def test_fresh_forecast_uses_same_calibrator_without_target_outcomes():
    from test_backtest import race, DEG, FUEL_PACE, TRACK
    from tyretrace.backtest import fp_race as B
    hist=generated(n_weekends=12)
    model=U.PredictiveUncertainty.fit(hist,target='2025_01')
    fc=B.forecast(race(),{c:(v,.001,'practice') for c,v in DEG.items()},
                  (-FUEL_PACE+TRACK,.001),sigma=.1)
    result=model.predict_forecast(fc,weekend='2025_01',method='identified')
    assert result.height==fc.laps.height and 'y' not in result.columns
    assert np.allclose(result['q50'].to_numpy(),np.median(fc.draws,axis=0))
    assert 'stint' in result.columns
    with pytest.raises(ValueError,match='method'):
        model.predict_forecast(fc,weekend='2025_01',method='conventional')


def test_audit_matches_observation_identities_and_rejects_duplicate_laps():
    from audit.u1_uncertainty import compare
    t=generated(n_weekends=10,n_laps=100)
    t=t.with_columns(pl.int_range(pl.len()).alias('lap_in_session'),
                     pl.lit('D0').alias('driver'),pl.lit(0).alias('stint'))
    model=U.PredictiveUncertainty.fit(t.filter(pl.col('weekend')<'2024_10'),target='2024_10')
    g=t.filter(pl.col('weekend')=='2024_10')
    sd=(g['q95'].to_numpy()-g['q05'].to_numpy())/U.Z90
    g=g.with_columns(pl.Series('crps',U.normal_crps(g['q50'].to_numpy(),sd,g['y'].to_numpy())))
    pred=model.predict(g,target='2024_10')
    a,_=compare(g,pred)
    b,_=compare(g.reverse(),pred)
    assert a==b
    with pytest.raises(ValueError,match='unique'):
        compare(pl.concat([g,g]),pred)


def test_invalid_distribution_parameters_are_rejected():
    with pytest.raises(ValueError,match='nonnegative'):
        U.PredictiveUncertainty((-1.,1.,1.,1.),'2024_10',10,1000)
    with pytest.raises(ValueError,match='exceed 2'):
        U.PredictiveUncertainty((1.,1.,1.,1.),'2024_10',10,1000,df=1.5)
