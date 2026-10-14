"""Recovery tests. These are the test suite that matters.

Not coverage of utility functions -- simulate from known truth, fit, assert
the truth comes back at the right rate. A statistical regression (L4 quietly
breaking L2's calibration six weeks from now) is invisible to every other
kind of test.

Run:  python -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit.a3_synthetic_recovery import (  # noqa: E402
    COMPOUNDS, compound_rank_check, est_practice_mixed, est_race_joint,
)
from tyretrace import schema  # noqa: E402
from tyretrace.models import track_evolution as te  # noqa: E402
from tyretrace.sim import minimal  # noqa: E402
from tyretrace.sim.minimal import Truth  # noqa: E402


# --------------------------------------------------------------------------
# schema contract -- sim and real ingest must be indistinguishable
# --------------------------------------------------------------------------

def test_sim_emits_canonical_schema():
    rng = np.random.default_rng(0)
    for df in (minimal.simulate_race(rng), minimal.simulate_practice(rng)):
        assert list(df.columns) == list(schema.LAPS_SCHEMA)
        assert df.height > 0


def test_validate_rejects_missing_required_column():
    rng = np.random.default_rng(0)
    df = minimal.simulate_race(rng).drop("lap_in_run")
    with pytest.raises(ValueError, match="missing required"):
        schema.validate(df)


def test_validate_rejects_non_monotone_tyre_age():
    rng = np.random.default_rng(0)
    df = minimal.simulate_race(rng)
    bad = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 3).then(0)
          .otherwise(pl.col("tyre_age")).cast(pl.Int16).alias("tyre_age"))
    with pytest.raises(ValueError, match="tyre_age decreases"):
        schema.validate(bad)


# --------------------------------------------------------------------------
# identification -- the claims from the plan, as assertions
# --------------------------------------------------------------------------

def test_race_design_recovers_degradation():
    """The pit-stop discontinuity identifies per-compound degradation."""
    truth = Truth()
    rng = np.random.default_rng(7)
    fit = est_race_joint(minimal.simulate_race(rng, truth))
    for c in COMPOUNDS:
        m, se = fit[c]
        assert abs(m - truth.deg[c]) <= 3 * se, f"{c}: {m} vs {truth.deg[c]}"


def test_race_leaves_fuel_and_track_fused():
    """Honest limitation: a race cannot split fuel from track evolution.

    The race-lap coefficient must land on -(fuel_pace + tau), NOT on either
    term alone. If this ever stops holding, the identification story in the
    plan is wrong and the write-up needs changing.
    """
    truth = Truth()
    rng = np.random.default_rng(11)
    fit = est_race_joint(minimal.simulate_race(rng, truth))
    m, se = fit["_lap_fused"]
    fused = -(truth.fuel_pace + truth.tau_race)
    assert abs(m - fused) <= 4 * se
    # and it is genuinely NOT the fuel term on its own
    assert abs(m - (-truth.fuel_pace)) > 3 * se


def test_compound_invariance_gives_contrasts_not_levels():
    """Within one session, compound invariance identifies differences only."""
    rng = np.random.default_rng(3)
    rc = compound_rank_check(minimal.simulate_practice(rng))
    assert rc["null_dim"] == 1, "design should be rank deficient by exactly one"
    assert rc["level_leakage"] > 1e-8, "a level must NOT be estimable"
    assert rc["contrast_leakage"] < 1e-8, "a contrast must be estimable"


def _pooled_practice(seed: int, programme: bool, n: int = 8) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.concat([minimal.simulate_practice(rng, Truth(), rnd=k + 1,
                                                fuel_programme=programme)
                      for k in range(n)])


def test_practice_recovers_tau_without_a_fuel_programme():
    """The idealised session the original argument assumed: fuel loads random
    with respect to the clock. Partial pooling recovers track evolution."""
    truth = Truth()
    fit = est_practice_mixed(_pooled_practice(5, programme=False))
    m, se = fit.get("tau_per_min")
    assert abs(m - (-truth.tau_practice)) <= 3 * se


def test_practice_fails_under_a_realistic_fuel_programme():
    """The finding that overturned the original A1 premise.

    Real teams run low-fuel quali sims early and high-fuel race sims late, so
    fuel state trends with the session clock (measured corr +0.54). Under that
    programme the practice arm returns track evolution with the WRONG SIGN --
    the cars get heavier faster than the track rubbers in.

    This test asserts the failure on purpose. If it ever starts passing,
    either the simulator stopped modelling the programme or the estimator
    learned to handle it, and the plan needs rewriting either way.
    """
    truth = Truth()
    fit = est_practice_mixed(_pooled_practice(5, programme=True))
    m, _ = fit.get("tau_per_min")
    assert m > 0, "expected the wrong sign under a realistic fuel programme"
    assert -truth.tau_practice < 0, "truth is negative; the estimate is not"


def test_quali_anchor_recovers_track_evolution():
    """The replacement mechanism. Qualifying runs every car at minimum fuel,
    so the clock coefficient is track evolution and nothing else."""
    truth = Truth()
    rng = np.random.default_rng(99)
    covered = 0
    fits = []
    for k in range(10):
        q = minimal.simulate_quali(rng, truth, rnd=k + 1)
        f = te.fit_session(te.push_laps(q), min_laps=20)
        assert f is not None, "quali session should be fittable"
        fits.append(f)
        covered += abs(f.tau_per_min - (-truth.tau_practice)) <= 1.645 * f.se

    s = te.summary(fits)
    assert s["frac_negative"] == 1.0, "track evolution must be negative"
    assert covered >= 8, f"90% intervals covered only {covered}/10"
    assert s["median_z"] > 5, "quali anchor should be strongly identified"


def test_quali_anchor_is_robust_to_tyre_controls():
    """If the estimate were really Q3-on-fresher-rubber, adding tyre age and
    compound controls would move it. On real data it moves by under 5%."""
    truth = Truth()
    rng = np.random.default_rng(7)
    q = pl.concat([minimal.simulate_quali(rng, truth, rnd=k + 1) for k in range(6)])
    bare = te.summary(te.fit(q, controls=False))["median_tau"]
    ctrl = te.summary(te.fit(q, controls=True))["median_tau"]
    assert abs(ctrl - bare) < 0.25 * abs(bare)


def test_track_status_uses_fastf1_encoding():
    """Regression test for a real schema-drift bug.

    The simulator emitted track_status="green" while the FastF1 ingest emitted
    "1", so every green-only filter silently dropped 100% of simulated laps.
    Nothing failed loudly -- the quali audit just reported zero sessions.
    """
    rng = np.random.default_rng(0)
    df = minimal.simulate_quali(rng)
    assert set(df["track_status"].unique()) <= set("1234567"), \
        "simulator must use FastF1 digit codes, not prose"

    bad = df.with_columns(pl.lit("green").alias("track_status"))
    with pytest.raises(ValueError, match="non-FastF1 track_status"):
        schema.validate(bad)


@pytest.mark.slow
def test_race_intervals_are_calibrated():
    """Coverage gate: 90% intervals must cover ~90% of the time."""
    truth = Truth()
    hits, n = 0, 60
    for s in range(n):
        rng = np.random.default_rng(2000 + s)
        fit = est_race_joint(minimal.simulate_race(rng, truth))
        m, se = fit["C3"]
        if abs(m - truth.deg["C3"]) <= 1.645 * se:
            hits += 1
    cov = hits / n
    assert 0.80 <= cov <= 0.98, f"coverage {cov:.3f} outside sane band"


def test_track_evolution_sign_convention_bridge():
    """Regression test for a sign error that reached a reported result.

    track_evolution measures tau as a regression coefficient on the session
    clock, so an IMPROVING track gives a NEGATIVE tau. The sim and the ladder
    models enter it as ``- tau * lap``, so an improving track is POSITIVE
    there. Passing the measured value straight into a model inverts the
    fuel/track split. tau_model() is the only sanctioned bridge.
    """
    f = te.TrackEvolution("x", tau_per_min=-0.0165, se=0.0016,
                          span_min=60.0, n_laps=95, n_drivers=20)
    assert f.tau_per_min < 0, "improving track is negative in measured units"
    tm = f.tau_model(lap_time_s=90.0)
    assert tm > 0, "improving track must be POSITIVE in model units"
    assert tm == pytest.approx(0.0165 * 90.0 / 60.0)

    # a worsening track must flip both ways round
    g = te.TrackEvolution("y", 0.02, 0.002, 60.0, 95, 20)
    assert g.tau_model(90.0) < 0


def test_quali_fits_convert_to_positive_model_tau():
    """End to end: real-shaped quali data must yield a POSITIVE model tau."""
    rng = np.random.default_rng(4)
    q = minimal.simulate_quali(rng)
    fits = te.fit(q)
    assert fits, "expected fittable quali sessions"
    tau, se = te.summary_tau_model(fits, lap_time_s=90.0)
    assert tau > 0 and se > 0
