"""Baselines -- the contract every one of them must honour to be comparable."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tyretrace.eval import metrics  # noqa: E402
from tyretrace.models import baselines as B  # noqa: E402
from tyretrace.models.ladder import Design  # noqa: E402


def race(seed=0, n_drv=6, n_laps=50, pit=25, beta=0.030, kappa=1.8, deg=0.06,
         sigma=0.1):
    """One race, one weekend, two stints per driver, known truth."""
    rng = np.random.default_rng(seed)
    rows = []
    for drv in range(n_drv):
        off = rng.normal(0, 0.3)
        for lap in range(1, n_laps + 1):
            stint = 0 if lap <= pit else 1
            age = lap if lap <= pit else lap - pit
            fuel = 110.0 - kappa * (lap - 1)
            y = 90 + off + beta * fuel + deg * age + rng.normal(0, sigma)
            rows.append((drv, lap, age, stint, y))
    a = np.array(rows, dtype=float)
    lap = a[:, 1]
    d = Design(y=a[:, 4], lap=lap - lap.mean(), age=a[:, 2],
               driver=a[:, 0].astype(np.int32), circuit=np.zeros(len(a), np.int32),
               compound=a[:, 3].astype(np.int32), n_driver=n_drv, n_circuit=1,
               n_compound=2, lap_mean=float(lap.mean()))
    d.kappa = np.full(len(a), kappa)
    d.tau_quali = np.zeros(len(a))
    d.dirty_air = np.zeros(len(a))
    d.stint = (a[:, 0] * 2 + a[:, 3]).astype(np.int32)
    d.n_stint = n_drv * 2
    d.weekend = np.zeros(len(a), np.int32)
    d.n_weekend = 1
    return d


def cut(d, frac=0.7):
    from audit.l3_dirty_air import split, subset
    tr, te = split(d, frac)
    return subset(d, tr), subset(d, te)


@pytest.mark.parametrize("name", list(B.BASELINES))
def test_every_baseline_returns_scorable_draws(name):
    tr, te = cut(race())
    pp = B.BASELINES[name](tr, te)
    assert pp.ndim == 2 and pp.shape[1] == te.n
    assert np.all(np.isfinite(pp))
    s = metrics.score(pp, te.y)                 # must not raise
    assert 0 <= s["coverage_90"] <= 1


def test_gbm_quantiles_never_cross():
    tr, te = cut(race())
    pp = B.gbm(tr, te)
    assert pp.shape[0] == len(B.GBM_LEVELS)
    assert np.all(np.diff(pp, axis=0) >= 0)


def test_naive_falls_back_for_a_stint_unseen_in_training():
    # pit at lap 40 of 50: the second stint starts after the 70% cut, so the
    # naive model has never seen it and must not extrapolate a line it lacks
    tr, te = cut(race(pit=40))
    unseen = ~np.isin(te.stint, tr.stint)
    assert unseen.any()
    pp = B.naive(tr, te)
    assert np.all(np.isfinite(pp[:, unseen]))


def test_conventional_fits_to_noise_when_its_assumption_is_right():
    """With the assumed fuel coefficient equal to the truth the model is
    correctly specified, so in-sample it must fit to the noise level -- a
    check on the fuel-load plumbing (sign, lap centring, kappa).

    In-sample on purpose. Out of sample it does NOT forecast to noise even
    here: the age quadratic, fitted on ages 1-10 of a fresh set, extrapolates
    a noisy curvature to ages 11-25 (measured bias +0.34 s). That is the
    conventional method behaving as specified, not a bug, so it is kept."""
    d = race(beta=B.CONVENTIONAL_S_PER_KG, sigma=0.1, seed=3)
    pp = B.conventional(d, d)
    assert metrics.mae(pp, d.y) < 0.1


def test_a_wrong_fuel_assumption_fits_fine_and_reports_the_wrong_degradation():
    """The project's thesis, as a test.

    Truth beta = 0.045 s/kg; the conventional method assumes 0.030. The fit
    does not reveal the mistake -- in-sample error barely moves, because the
    free age curve absorbs it. What moves is the degradation it REPORTS.
    Measured: the first set reads 0.089 against a true 0.060 (the predicted
    fuel error x burn rate, +0.027); the second set, whose age curve also has
    to absorb the jump at the stop, reads -0.021 -- tyres getting FASTER with
    age. Nothing in the fit warns you.
    """
    kappa, deg = 1.8, 0.060
    right = race(beta=B.CONVENTIONAL_S_PER_KG, kappa=kappa, deg=deg, seed=4)
    wrong = race(beta=0.045, kappa=kappa, deg=deg, seed=4)

    mae_right = metrics.mae(B.conventional(right, right), right.y)
    mae_wrong = metrics.mae(B.conventional(wrong, wrong), wrong.y)
    assert mae_wrong < 1.5 * mae_right                  # the fit looks fine

    bias = (0.045 - B.CONVENTIONAL_S_PER_KG) * kappa    # ~ +0.027 s/lap
    assert np.allclose(B.conventional_degradation(right), deg, atol=0.01)
    reported = B.conventional_degradation(wrong)
    assert np.all(np.abs(reported - deg) > 0.5 * bias)  # the number is wrong


def test_verified_published_curve_recovers_compound_offsets_and_start_age():
    d=race(sigma=.01,seed=91)
    rng=np.random.default_rng(92)
    ci=d.compound
    fuel=B._fuel_kg(d)
    # Actual source model: k0 differs by compound, log curve uses age at START
    # of a lap. The legacy zero-k0 / inclusive-age adaptation is misspecified.
    d.y=(90+.03*fuel+np.array([-.7,.9])[ci]
         +np.array([.8,1.2])[ci]*np.log1p(.2*np.maximum(d.age-1,0))
         +rng.normal(0,.01,d.n))
    tr,te=cut(d)
    verified=B.published_verified(tr,te)
    legacy=B.published(tr,te)
    assert metrics.mae(verified,te.y)<.04
    assert metrics.mae(verified,te.y)<metrics.mae(legacy,te.y)/2
