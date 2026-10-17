"""Run planner -- the information maths, and the identification claims it rests on."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tyretrace.planner import eig as E  # noqa: E402

PR = E.Prior()
DEG = E.TARGETS["degradation per compound"]


def prog(*runs):
    p = E.Programme()
    for r in runs:
        p = p.add(r, PR)
    return p


def test_posterior_covariance_matches_brute_force_simulation():
    """The closed form is only worth trusting if it agrees with reality:
    draw parameters from the prior, simulate laps, compute the conjugate
    posterior mean, and check the realised errors have the claimed spread."""
    p = prog(E.Run("SOFT", 8, swap_to="HARD"), E.Run("MEDIUM", 10))
    X = E.design(p, PR)
    n_run = len(p.runs)
    S0 = np.zeros((E.P + n_run,) * 2)
    S0[:E.P, :E.P] = PR.cov()
    S0[E.P:, E.P:] = np.eye(n_run) * PR.run_sd ** 2
    m0 = np.r_[PR.beta_f[0], [PR.deg[0]] * 3, PR.tau[0], np.zeros(n_run)]
    prec = np.linalg.inv(S0) + X.T @ X / PR.sigma ** 2
    S = np.linalg.inv(prec)

    rng = np.random.default_rng(0)
    L = np.linalg.cholesky(S0)
    errs = []
    for _ in range(4000):
        th = m0 + L @ rng.standard_normal(len(m0))
        y = X @ th + PR.sigma * rng.standard_normal(len(X))
        post = S @ (np.linalg.solve(S0, m0) + X.T @ y / PR.sigma ** 2)
        errs.append((post - th)[:E.P])
    emp = np.std(errs, axis=0)
    claimed = np.sqrt(np.diag(E.posterior_cov(p, PR)))
    assert np.allclose(emp, claimed, rtol=0.06)


def test_start_fuel_carries_no_information_when_run_pace_is_unknown():
    """With a diffuse prior on each run's pace, fuel load only moves the
    run intercept, so it cannot matter. (With the default 3 s prior it
    matters slightly -- a light run against a heavy one says a little about
    beta_f if run pace is known to within seconds.)"""
    pr = E.Prior(run_sd=1e3)
    a, b = E.Programme(), E.Programme()
    a = a.add(E.Run("SOFT", 10, start_fuel_kg=20.0), pr)
    b = b.add(E.Run("SOFT", 10, start_fuel_kg=100.0), pr)
    assert np.isclose(E.eig(a, pr, DEG), E.eig(b, pr, DEG), rtol=1e-6)


# The identification claims are about the LIMIT: what a run cannot teach
# however long it is. On short, noisy runs lap-time noise dominates instead,
# which is exactly the trade-off the planner weighs -- a 12-lap run with a
# tyre change at half distance is WORSE for the degradation level than a
# plain 12-lap run (measured sd 0.032 vs 0.024), because each half is 6 laps.
LONG = 120


def test_a_plain_run_hits_a_floor_on_the_degradation_level():
    """Without a tyre change the level is only learned through the fused
    slope, so its uncertainty stops at what the fuel prior allows."""
    sd = E.posterior_sd(prog(E.Run("SOFT", LONG)), PR)["deg_SOFT"]
    assert sd > 0.25 * PR.deg[1]


def test_a_mid_run_tyre_change_breaks_through_the_floor():
    """Same timed laps, but the age reset breaks the fuel / age lock step."""
    plain = E.posterior_sd(prog(E.Run("SOFT", LONG)), PR)["deg_SOFT"]
    swap = E.posterior_sd(prog(E.Run("SOFT", LONG, swap_to="SOFT")), PR)["deg_SOFT"]
    assert swap < 0.25 * plain


def test_two_compounds_pin_their_contrast_but_not_their_levels():
    sd = E.posterior_sd(prog(E.Run("SOFT", LONG // 2), E.Run("HARD", LONG // 2)), PR)
    assert sd["SOFT−HARD"] < 0.25 * sd["deg_SOFT"]


def test_information_never_decreases_when_a_run_is_added():
    p = E.Programme()
    last = 0.0
    for r in (E.Run("SOFT", 6), E.Run("HARD", 8), E.Run("MEDIUM", 6, swap_to="HARD")):
        p = p.add(r, PR)
        g = E.eig(p, PR, DEG)
        assert g >= last - 1e-9
        last = g


@pytest.mark.parametrize("target", list(E.TARGETS))
def test_plan_fits_the_session_and_beats_a_standard_fp2(target):
    A = E.TARGETS[target]
    steps = E.plan(PR, A, session_min=60.0)
    planned = E.programme(steps)
    std = E.standard_programme(PR)
    assert std.end(PR) <= 60.0, "the comparison must be on equal session time"
    assert planned.end(PR) <= 60.0
    assert E.eig(planned, PR, A) >= E.eig(std, PR, A)
    # the reported running total is the programme's actual information
    assert np.isclose(steps[-1].total_bits, E.eig(planned, PR, A))


def test_reserved_runs_are_kept_and_counted():
    start = E.Programme().add(E.QUALI_SIM, PR)
    steps = E.plan(PR, DEG, session_min=60.0, start=start)
    planned = E.programme(steps, start)
    assert planned.runs[0] == E.QUALI_SIM and planned.starts[0] == start.starts[0]
    assert planned.end(PR) <= 60.0
    assert np.isclose(steps[-1].total_bits, E.eig(planned, PR, DEG))


def test_with_the_quali_sim_kept_the_planner_still_beats_a_standard_fp2():
    """The fair comparison: both programmes spend the same 7.5 minutes on a
    quali sim, so any gain comes from the long-run choices alone."""
    start = E.Programme().add(E.QUALI_SIM, PR)
    planned = E.programme(E.plan(PR, DEG, session_min=60.0, start=start), start)
    assert E.eig(planned, PR, DEG) > E.eig(E.standard_programme(PR), PR, DEG)


def test_planner_matches_exhaustive_search_on_a_short_session():
    """On a session short enough to enumerate every programme, the planner
    must come within 1% of the best one. Measured: 0.78047 vs the exhaustive
    0.78054 bits, a 0.009% gap -- local search stops one hair short of the
    global optimum (run ORDER shifts the clock rows slightly; the enumeration
    fixes one order per combination, the planner another)."""
    from itertools import combinations_with_replacement
    session = 40.0
    cands = E.candidates(lengths=(4, 6, 8, 10, 12))
    best = 0.0
    for k in (1, 2, 3):
        for combo in combinations_with_replacement(cands, k):
            prog = E.Programme()
            for r in combo:
                prog = prog.add(r, PR)
            if prog.end(PR) <= session:
                best = max(best, E.eig(prog, PR, DEG))
    got = E.eig(E.programme(E.plan(PR, DEG, session_min=session, cands=cands)), PR, DEG)
    assert got >= 0.99 * best


def test_fair_comparison_without_tyre_changes_does_not_lose():
    """Regression: with the quali sim fixed and swaps off, greedy + local
    search settled 10% below a typical FP2 whose runs were all candidates."""
    start = E.Programme().add(E.QUALI_SIM, PR)
    cands = E.candidates(allow_swap=False)
    planned = E.programme(E.plan(PR, DEG, session_min=60.0, cands=cands, start=start), start)
    assert E.eig(planned, PR, DEG) >= E.eig(E.standard_programme(PR), PR, DEG) - 1e-9
