"""L0 / L1 tests -- identification, geometry, and the ship gate.

Kept small (few draws, one chain) so they run in CI. The full-precision
numbers live in audit/l0_l1_ladder.py --full.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tyretrace.eval import metrics  # noqa: E402
from tyretrace.models import ladder  # noqa: E402
from tyretrace.sim import minimal  # noqa: E402
from tyretrace.sim.minimal import Truth  # noqa: E402


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def test_crps_matches_closed_form_for_a_normal():
    """CRPS of N(0,1) at y=0.

    From the energy form E|X-y| - 0.5*E|X-X'|:  X ~ N(0,1) gives
    E|X| = sqrt(2/pi), and X-X' ~ N(0,2) gives E|X-X'| = 2/sqrt(pi),
    so CRPS = sqrt(2/pi) - 1/sqrt(pi) ~ 0.2337.
    """
    rng = np.random.default_rng(0)
    draws = rng.standard_normal((40000, 1))
    got = float(metrics.crps_ensemble(draws, np.zeros(1))[0])
    expect = np.sqrt(2 / np.pi) - 1 / np.sqrt(np.pi)
    assert abs(got - expect) < 5e-3, f"{got} vs {expect}"


def test_crps_matches_closed_form_shifted_and_scaled():
    """The general form, so a sign or scale slip cannot hide at the origin:
    CRPS(N(mu,s), y) = s*[z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)], z=(y-mu)/s.
    """
    from scipy.stats import norm

    mu, sd, y = 1.0, 2.0, 1.5
    rng = np.random.default_rng(3)
    draws = mu + sd * rng.standard_normal((60000, 1))
    got = float(metrics.crps_ensemble(draws, np.array([y]))[0])

    z = (y - mu) / sd
    expect = sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    assert abs(got - expect) < 5e-3, f"{got} vs {expect}"


def test_crps_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="incompatible"):
        metrics.crps_ensemble(np.zeros((10, 5)), np.zeros(3))


def test_crps_rewards_sharpness_when_correct():
    y = np.zeros(1)
    rng = np.random.default_rng(1)
    sharp = metrics.crps_ensemble(0.1 * rng.standard_normal((5000, 1)), y)[0]
    vague = metrics.crps_ensemble(2.0 * rng.standard_normal((5000, 1)), y)[0]
    assert sharp < vague


def test_ship_gate_rejects_sharper_but_miscalibrated():
    """Protocol §5: better CRPS does NOT ship if calibration breaks."""
    base = {"crps": 0.30, "coverage_90": 0.90}
    better_but_broken = {"crps": 0.10, "coverage_90": 0.70}
    ok, why = metrics.passes_ship_gate(better_but_broken, base)
    assert not ok and "coverage" in why


def test_ship_gate_rejects_no_crps_gain():
    base = {"crps": 0.10, "coverage_90": 0.90}
    worse = {"crps": 0.20, "coverage_90": 0.90}
    ok, why = metrics.passes_ship_gate(worse, base)
    assert not ok and "CRPS" in why


# --------------------------------------------------------------------------
# design contract
# --------------------------------------------------------------------------

def test_design_like_reuses_training_code_space():
    """A test frame encoded independently would silently mean different
    drivers, corrupting every held-out score without failing."""
    rng = np.random.default_rng(4)
    tr = minimal.simulate_race(rng, rnd=1)
    te = minimal.simulate_race(rng, rnd=2)

    d_tr = ladder.build_design(tr)
    d_te = ladder.build_design(te, like=d_tr)

    assert d_te.driver_names == d_tr.driver_names
    assert d_te.lap_mean == d_tr.lap_mean
    assert d_te.n_driver == d_tr.n_driver

    name_of = {v: k for k, v in enumerate(d_tr.driver_names)}
    drivers = te["driver"].to_numpy().astype(str)
    assert all(d_te.driver[i] == name_of[drivers[i]] for i in range(0, len(drivers), 37))


# --------------------------------------------------------------------------
# identification
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_l0_recovers_degradation_and_fused_slope():
    truth = Truth()
    rng = np.random.default_rng(2026)
    # L0 pools compounds and has no circuit term: correctly specified only on
    # one race with one compound.
    d = ladder.build_design(minimal.simulate_race(rng, truth, compounds=["C3"]))
    mcmc = ladder.fit("L0", d, draws=400, warmup=400, chains=1)

    s = mcmc.get_samples()
    lo, hi = np.quantile(s["deg"], [0.05, 0.95])
    assert lo <= truth.deg["C3"] <= hi

    lo, hi = np.quantile(s["lap_coef"], [0.05, 0.95])
    assert lo <= -(truth.fuel_pace + truth.tau_race) <= hi

    assert ladder.diagnostics(mcmc)["divergences"] == 0


@pytest.mark.slow
def test_l1_recovers_per_compound_degradation_without_divergences():
    """Regression test for two real geometry bugs.

    The hierarchy once had a redundant additive degree of freedom (grand mean
    vs driver offsets), giving r_hat 2.23; and circuits were partially pooled
    with a non-centred parameterisation despite ~1250 laps each, which put the
    sampler on a ridge (ESS 51). Both are fixed -- this test fails loudly if
    either comes back.
    """
    truth = Truth()
    rng = np.random.default_rng(2026)
    pooled = pl.concat([minimal.simulate_race(rng, truth, rnd=k + 1)
                        for k in range(3)])
    d = ladder.build_design(pooled)
    mcmc = ladder.fit("L1", d, draws=600, warmup=600, chains=2)

    diag = ladder.diagnostics(mcmc)
    assert diag["divergences"] == 0, f"{diag['divergences']} divergences"
    assert diag["max_rhat"] < 1.05, f"r_hat {diag['max_rhat']}"

    s = mcmc.get_samples()
    for i, c in enumerate(d.compound_names):
        lo, hi = np.quantile(s["deg"][:, i], [0.05, 0.95])
        assert lo <= truth.deg[c] <= hi, f"{c}: [{lo}, {hi}] misses {truth.deg[c]}"


def test_l1_handles_a_single_compound_label():
    """Real data currently has one compound label (Pirelli map pending), which
    would leave sd_deg and z_deg trading off freely if not guarded."""
    rng = np.random.default_rng(9)
    d = ladder.build_design(minimal.simulate_race(rng, compounds=["C3"]))
    assert d.n_compound == 1
    mcmc = ladder.fit("L1", d, draws=200, warmup=200, chains=1)
    s = mcmc.get_samples()
    assert "sd_deg" not in s, "compound hierarchy must be skipped for n=1"
    assert s["deg"].shape[1] == 1


# --------------------------------------------------------------------------
# L2 -- the estimated fuel/track split
# --------------------------------------------------------------------------

def _l2_design(rng, truth, n_weekends=6):
    """Sim races across circuits with differing kappa, plus a known tau."""
    frames = []
    kappas, taus = [], []
    for k in range(n_weekends):
        n_laps = 50 + 5 * k                       # varies kappa
        df = minimal.simulate_race(rng, truth, rnd=k + 1, n_laps=n_laps)
        frames.append(df)
        kappas.append(truth.kappa)
        taus.append(truth.tau_race)
    races = pl.concat(frames)
    d = ladder.build_design(races)
    d.kappa = np.full(d.n, truth.kappa)
    d.tau_quali = np.full(d.n, truth.tau_race)
    wk = races["session_uid"].to_numpy().astype(str)
    lev = sorted(set(wk.tolist()))
    d.weekend = np.array([lev.index(w) for w in wk], dtype=np.int32)
    d.n_weekend = len(lev)
    d.weekend_names = lev
    return d


@pytest.mark.slow
def test_l2_recovers_the_tau_scale():
    """With kappa and tau supplied at their true values, the scale s must
    come back at 1.0 -- the race evolves exactly as the anchor says."""
    truth = Truth()
    rng = np.random.default_rng(17)
    d = _l2_design(rng, truth)
    mcmc = ladder.fit("L2", d, draws=600, warmup=600, chains=2)

    diag = ladder.diagnostics(mcmc)
    assert diag["divergences"] == 0
    s = mcmc.get_samples()

    lo, hi = np.quantile(s["tau_scale"], [0.05, 0.95])
    assert lo <= 1.0 <= hi, f"tau_scale 90% [{lo}, {hi}] misses 1.0"

    lo, hi = np.quantile(s["beta_f"], [0.05, 0.95])
    assert lo <= truth.beta_f <= hi, f"beta_f 90% [{lo}, {hi}] misses {truth.beta_f}"


def test_l2_degradation_is_not_pooled_across_weekends():
    """Relative compound labels are not comparable between races, so `deg`
    must be indexed by weekend AND compound -- pooling them is exactly the
    error the Pirelli C1-C5 map exists to prevent."""
    truth = Truth()
    rng = np.random.default_rng(21)
    d = _l2_design(rng, truth, n_weekends=3)
    mcmc = ladder.fit("L2", d, draws=200, warmup=200, chains=1)
    deg = mcmc.get_samples()["deg"]
    assert deg.ndim == 3, "deg must be (draws, weekend, compound)"
    assert deg.shape[1] == d.n_weekend
    assert deg.shape[2] == d.n_compound
