"""FP -> race backtest: every building block recovers known truth."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tyretrace.backtest import fp_race as B  # noqa: E402
from tyretrace.eval import metrics  # noqa: E402

FUEL_PACE = 0.032      # s/lap, the "true" fuel effect in these tests
KAPPA = 1.8
TRACK = -0.012         # s/lap of track evolution at the quali rate
DEG = {"MEDIUM": 0.05, "HARD": 0.03}


def practice(seed=0, n_runs=8, laps=10, sigma=0.05):
    """Runs whose within-run slope is deg - fuel + track, levels arbitrary."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_runs):
        comp = "MEDIUM" if r % 2 else "HARD"
        lvl = rng.normal(92, 1.5)              # fuel load, mode: unknown level
        for k in range(1, laps + 1):
            y = lvl + (DEG[comp] - FUEL_PACE + TRACK) * k + rng.normal(0, sigma)
            rows.append(("2024_01_FP2", f"D{r}", r, k, comp, y))
    return pl.DataFrame(rows, schema=["session_uid", "driver", "stint_id", "lap_in_run",
                                      "compound_rel", "lap_time_s"], orient="row")


def race(seed=0, n_drv=10, n_laps=50, pit=22, sigma=0.1, fuel=FUEL_PACE, s=1.0):
    """MEDIUM to lap `pit`, then HARD. Track evolves at s x the quali rate."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_drv):
        off = rng.normal(0, 0.4)
        for lap in range(2, n_laps + 1):
            first = lap <= pit
            comp, age, stint = ("MEDIUM", lap, 0) if first else ("HARD", lap - pit, 1)
            y = 90 + off + (-fuel + s * TRACK) * lap + DEG[comp] * age + rng.normal(0, sigma)
            rows.append(("2024_01_R", f"D{d}", stint, lap, age, comp, y))
    return pl.DataFrame(rows, schema=["session_uid", "driver", "stint_id", "lap_in_session",
                                      "tyre_age", "compound_rel", "lap_time_s"], orient="row")


WK = B.Weekend("2024_01", tau_min=TRACK * 60 / 90.0, lap_s=90.0, kappa=KAPPA)


def test_weekend_track_per_lap_round_trips():
    assert np.isclose(WK.track_per_lap, TRACK)


def test_practice_slopes_recover_the_fused_slope_despite_run_levels():
    sl = B.practice_slopes(practice())
    for c in DEG:
        assert abs(sl[c][0] - (DEG[c] - FUEL_PACE + TRACK)) < 3 * sl[c][1] + 1e-3


def test_degradation_undoes_fuel_and_track():
    deg = B.degradation(B.practice_slopes(practice(sigma=0.01)), (FUEL_PACE, 0.0),
                        TRACK, ["HARD", "MEDIUM", "SOFT"])
    for c in DEG:
        assert abs(deg[c][0] - DEG[c]) < 0.003
    assert deg["SOFT"][2] == "fallback" and deg["SOFT"][1] > B.FALLBACK_SD * 0.9


def test_race_lap_coef_is_the_fused_fuel_and_track_slope():
    coef, se, _ = B.race_lap_coef(race())
    assert abs(coef - (-FUEL_PACE + TRACK)) < 3 * se + 1e-4


def test_fit_fuel_track_recovers_beta_and_scale_across_races():
    """Lap coefficients from races with different burn rates and track
    evolution pin both parameters."""
    rng = np.random.default_rng(1)
    beta, s = 0.020, 0.7
    wks, coefs, ses = [], [], []
    for i in range(8):
        k = rng.uniform(1.5, 2.2)
        tau = rng.uniform(-0.025, -0.008) * 60 / 90
        w = B.Weekend(f"w{i}", tau, 90.0, k)
        wks.append(w)
        coefs.append(-beta * k + s * w.track_per_lap + rng.normal(0, 0.0005))
        ses.append(0.0005)
    ft = B.fit_fuel_track(coefs, ses, wks)
    sd = np.sqrt(np.diag(ft.cov))
    assert abs(ft.beta_f - beta) < 3 * sd[0]
    assert abs(ft.s - s) < 3 * sd[1]


def test_correct_fuel_forecasts_the_post_stop_stint_without_bias():
    r = race(seed=2)
    deg = {c: (v, 0.001, "practice") for c, v in DEG.items()}
    fc = B.forecast(r, deg, (-FUEL_PACE + TRACK, 0.0005), sigma=0.1)
    after = fc.laps["after_stop"].to_numpy()
    err = np.median(fc.draws, axis=0) - fc.laps["lap_time_s"].to_numpy()
    assert abs(err[after].mean()) < 0.05
    assert 0.85 <= metrics.coverage(fc.draws, fc.laps["lap_time_s"].to_numpy()) <= 0.95


def test_an_overstated_fuel_effect_misplaces_the_post_stop_stint():
    """The backtest's whole point, on known truth. Assume fuel pace 0.054
    (the conventional 0.030 s/kg x 1.8) when the truth is 0.032: practice
    slopes turn into overstated degradation, the lap slope is overstated
    too, and the errors land on the laps after the stop."""
    r, p = race(seed=3), practice(seed=3, sigma=0.01)
    wrong = (0.054, 0.0)
    deg = B.degradation(B.practice_slopes(p), wrong, 0.0, ["HARD", "MEDIUM"])
    fc = B.forecast(r, deg, (-wrong[0], 0.0), sigma=0.1)
    after = fc.laps["after_stop"].to_numpy()
    err = np.median(fc.draws, axis=0) - fc.laps["lap_time_s"].to_numpy()

    right = B.degradation(B.practice_slopes(p), (FUEL_PACE, 0.0), TRACK, ["HARD", "MEDIUM"])
    fc2 = B.forecast(r, right, (-FUEL_PACE + TRACK, 0.0), sigma=0.1)
    err2 = np.median(fc2.draws, axis=0) - fc2.laps["lap_time_s"].to_numpy()
    # BIAS, not mean |error|: with the right fuel the error is already at the
    # noise floor (mean |noise| ~0.08 at sigma 0.1), so the signal is that the
    # wrong method is systematically off on every post-stop lap. Measured:
    # -0.22 s against ~0.
    assert abs(err[after].mean()) > 0.15
    assert abs(err2[after].mean()) < 0.05


def test_random_effects_widens_when_runs_disagree():
    """Three runs whose slopes disagree far beyond their own se must not
    pool into a precise answer -- the failure that set Montreal HARD to
    -0.23 s/lap in v1."""
    tight_m, tight_se = B.random_effects(np.array([0.05, 0.051, 0.049]), np.full(3, 0.01))
    wild_m, wild_se = B.random_effects(np.array([-0.3, 0.05, 0.4]), np.full(3, 0.01))
    assert tight_se < 0.01
    assert wild_se > 10 * tight_se


def test_prior_dominates_uninformative_practice_and_yields_to_informative():
    prior = (0.06, 0.02)
    vague = B.degradation({"HARD": (-0.2, 0.2, 3)}, (0.0, 0.0), 0.0, ["HARD"], prior)
    sharp = B.degradation({"HARD": (0.03, 0.002, 20)}, (0.0, 0.0), 0.0, ["HARD"], prior)
    missing = B.degradation({}, (0.0, 0.0), 0.0, ["HARD"], prior)
    assert abs(vague["HARD"][0] - 0.06) < 0.01
    assert abs(sharp["HARD"][0] - 0.03) < 0.002
    assert missing["HARD"][:2] == prior and missing["HARD"][2] == "prior"


def test_race_degradation_ignores_compounds_with_few_laps():
    r = race()
    d = B.race_degradation(r)
    assert set(d) == {"MEDIUM", "HARD"}
    assert abs(d["MEDIUM"] - DEG["MEDIUM"]) < 0.01
    few = r.filter(~((pl.col("compound_rel") == "HARD") & (pl.col("driver") != "D0")))
    assert "HARD" not in B.race_degradation(few)


def test_fit_fuel_only_recovers_beta_with_scatter_between_races():
    rng = np.random.default_rng(5)
    beta = 0.031
    wks = [B.Weekend(f"w{i}", -0.01, 90.0, k) for i, k in enumerate(rng.uniform(1.5, 2.2, 30))]
    coefs = [-beta * w.kappa + rng.normal(0, 0.008) for w in wks]   # real scatter
    m, sd = B.fit_fuel_only(coefs, [0.0005] * 30, wks)
    assert abs(m - beta) < 3 * sd
    # Between-race scatter must widen the interval far beyond what the tiny
    # per-race se alone implies (~0.0005 / (1.85 * sqrt(30)) ~ 5e-5) -- the
    # overconfidence that made L2 so sure about s.
    se_only = 0.0005 / (np.mean([w.kappa for w in wks]) * np.sqrt(30))
    assert sd > 5 * se_only


def test_fixed_effect_pooling_is_the_narrower_ablation():
    est, se = np.array([-0.3, 0.05, 0.4]), np.full(3, 0.01)
    _, sd_re = B.random_effects(est, se)
    _, sd_fe = B.random_effects(est, se, between=False)
    assert sd_fe < sd_re / 10


def test_traffic_aware_anchor_removes_anchor_traffic_bias():
    """Anchor laps in dirty air make the forecast too slow; de-trafficking
    them with the true wake (and expecting no later traffic) removes it."""
    r = race(seed=6).with_columns(
        pl.when(pl.col("lap_in_session") < 8).then(0.8).otherwise(0.0).alias("dirty_air"))
    r = r.with_columns(pl.col("lap_time_s") + 0.5 * pl.col("dirty_air"))
    deg = {c: (v, 0.001, "practice") for c, v in DEG.items()}
    lc = (-FUEL_PACE + TRACK, 0.0005)
    plain = B.forecast(r, deg, lc, sigma=0.1)
    aware = B.forecast(r, deg, lc, sigma=0.1, traffic=(0.5, 0.0))
    y = plain.laps["lap_time_s"].to_numpy()
    bias = lambda fc: float(np.mean(np.median(fc.draws, axis=0) - y))
    assert bias(plain) > 0.3          # anchored in traffic: forecast too slow
    assert abs(bias(aware)) < 0.05


def test_recalibration_uses_the_empirical_shape_and_keeps_the_median():
    r = race(seed=7)
    deg = {c: (v, 0.001, "practice") for c, v in DEG.items()}
    fc = B.forecast(r, deg, (-FUEL_PACE + TRACK, 0.0005), sigma=0.1)
    z_hist = np.concatenate([np.full(900, 0.0), np.full(100, 4.0)])   # skewed history
    cal = B.recalibrate(fc, z_hist)
    assert np.allclose(np.median(cal.draws, axis=0), np.median(fc.draws, axis=0), atol=0.02)
    q95 = np.quantile(cal.draws, 0.95, axis=0) - np.median(cal.draws, axis=0)
    q05 = np.median(cal.draws, axis=0) - np.quantile(cal.draws, 0.05, axis=0)
    assert np.all(q95 > 5 * np.maximum(q05, 1e-9))      # learned the asymmetry


def test_dry_filter_drops_any_race_with_a_wet_lap_and_keeps_the_original_ten():
    dry_race = race().with_columns(pl.lit("2024_01_R").alias("session_uid"))
    wet_race = race().with_columns(pl.lit("2025_12_R").alias("session_uid"),
                                   pl.when(pl.col("lap_in_session") == 3)
                                     .then(pl.lit("INTERMEDIATE"))
                                     .otherwise(pl.col("compound_rel")).alias("compound_rel"))
    assert B.dry_weekends(pl.concat([dry_race, wet_race])) == {"2024_01"}


def test_the_original_ten_weekends_are_all_dry():
    from tyretrace.data import sources
    if not sources.available("audit"):
        import pytest
        pytest.skip("audit data not present")
    r = sources.load("audit")["race"]
    assert len(B.dry_weekends(r)) == r["session_uid"].n_unique() == 10
