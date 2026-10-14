"""A3 -- minimal synthetic recovery.

Gate: truth inside the 90% interval, coverage in [0.85, 0.95] over 200 seeds.

Four things this settles, all on data where we know the answer:

  1. The naive within-stint slope is confounded, and by how much.
  2. The conventional fuel correction is biased by exactly its own
     assumption error -- it does not "mostly work".
  3. The race pit-stop design recovers degradation, but leaves fuel and
     track evolution fused. Practice recovers track evolution. Only both
     together close the system.
  4. Compound invariance alone identifies DIFFERENCES, never levels.
     Demonstrated as a rank argument, not an assertion.

Run:  python audit/a3_synthetic_recovery.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tyretrace.sim import minimal  # noqa: E402
from tyretrace.sim.minimal import Truth  # noqa: E402

COMPOUNDS = ("C2", "C3", "C4")


# --------------------------------------------------------------------------
# OLS with standard errors. No statsmodels dependency for something this small.
# --------------------------------------------------------------------------

def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (coef, se, rank). Uses pinv so rank-deficient designs do not
    raise -- we want to inspect deficiency, not crash on it."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    rank = int(np.linalg.matrix_rank(X))
    resid = y - X @ coef
    dof = max(len(y) - rank, 1)
    s2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * s2, 0.0))
    return coef, se, rank


def _dummies(labels: np.ndarray, levels) -> np.ndarray:
    return np.column_stack([(labels == lv).astype(float) for lv in levels])


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------

def est_naive(df: pl.DataFrame) -> dict[str, float]:
    """Within-stint OLS of lap time on lap-in-run, pooled by compound.

    This is what the collinearity argument says is broken. Expected to land
    on  delta_c - beta_f*kappa - tau  rather than delta_c.
    """
    out = {}
    for comp in COMPOUNDS:
        sub = df.filter(pl.col("compound_abs") == comp)
        if sub.height < 10:
            continue
        # de-mean within stint so each stint contributes only its slope
        sub = sub.with_columns([
            (pl.col("lap_time_s") - pl.col("lap_time_s").mean()
             .over(["driver", "stint_id"])).alias("y_c"),
            (pl.col("lap_in_run").cast(pl.Float64) - pl.col("lap_in_run")
             .mean().over(["driver", "stint_id"])).alias("x_c"),
        ])
        x = sub["x_c"].to_numpy()
        y = sub["y_c"].to_numpy()
        out[comp] = float(x @ y / (x @ x))
    return out


def est_conventional(df: pl.DataFrame, truth: Truth,
                     beta_assumed: float = 0.030) -> dict[str, float]:
    """The standard fix: subtract an assumed s/kg, fit the residual.

    beta_assumed is deliberately close to truth (0.030 vs 0.033) -- the point
    is that even a near-miss injects a bias nobody quantifies.
    """
    fuel = df["truth_fuel_kg"].to_numpy()          # generous: exact fuel path
    corrected = df["lap_time_s"].to_numpy() - beta_assumed * fuel
    return est_naive(df.with_columns(pl.Series("lap_time_s", corrected)))


def est_race_joint(df: pl.DataFrame) -> dict[str, tuple[float, float]]:
    """The pit-stop design. Regress on driver + race lap + age-by-compound.

    Race lap and tyre age are no longer collinear because age resets at every
    stop. Degradation separates. Fuel and track evolution both load on race
    lap and stay fused -- which is the honest limitation, not a bug.
    """
    drivers = df["driver"].to_numpy()
    lap = df["lap_in_session"].to_numpy().astype(float)
    age = df["tyre_age"].to_numpy().astype(float)
    comp = df["compound_abs"].to_numpy().astype(str)
    y = df["lap_time_s"].to_numpy()

    X = np.column_stack([
        _dummies(drivers, sorted(set(drivers))),   # per-driver intercepts
        lap,                                        # fuel + track, fused
        _dummies(comp, COMPOUNDS) * age[:, None],   # degradation by compound
    ])
    coef, se, _ = ols(X, y)
    k = X.shape[1] - len(COMPOUNDS)
    res = {c: (float(coef[k + i]), float(se[k + i])) for i, c in enumerate(COMPOUNDS)}
    res["_lap_fused"] = (float(coef[k - 1]), float(se[k - 1]))
    return res


def _practice_design(df: pl.DataFrame):
    """Shared design for the practice arm. Run = (driver, stint_id)."""
    drivers = df["driver"].to_numpy().astype(str)
    minute = df["session_clock_s"].to_numpy() / 60.0
    age = df["lap_in_run"].to_numpy().astype(float)
    comp = df["compound_abs"].to_numpy().astype(str)
    uid = df["session_uid"].to_numpy().astype(str)
    y = df["lap_time_s"].to_numpy()

    dlv = sorted(set(drivers))[1:]                  # drop first, intercept kept
    X = np.column_stack([
        np.ones(len(y)),
        minute,                                     # track evolution
        _dummies(comp, COMPOUNDS) * age[:, None],   # deg MINUS fuel pace
        _dummies(drivers, dlv),
    ])
    names = ["intercept", "tau_per_min", *[f"run_slope_{c}" for c in COMPOUNDS],
             *[f"drv_{d}" for d in dlv]]
    run = np.char.add(np.char.add(uid, "|"),
                      np.char.add(drivers, df["stint_id"].to_numpy().astype(str)))
    return X, y, run, names


def est_practice_fe(df: pl.DataFrame) -> dict[str, tuple[float, float]]:
    """Practice with run FIXED effects. Absorbs the unknown fuel load -- and
    kills track evolution with it, because within a run the session clock is
    nearly collinear with lap-in-run. Included to show the failure."""
    X, y, run, names = _practice_design(df)
    X = np.column_stack([X, _dummies(run, sorted(set(run))[1:])])
    coef, se, _ = ols(X, y)
    out = {names[i]: (float(coef[i]), float(se[i])) for i in range(len(names))}
    return out


def est_practice_mixed(df: pl.DataFrame):
    """Practice with run RANDOM effects -- partial pooling. The only version
    that can use between-run variation while still modelling unknown fuel."""
    from tyretrace.models import mixed
    X, y, run, names = _practice_design(df)
    return mixed.fit(X, y, run, names)


# --------------------------------------------------------------------------
# Check 4: what compound invariance can and cannot identify
# --------------------------------------------------------------------------

def compound_rank_check(df: pl.DataFrame) -> dict:
    """Inside ONE practice session, is the level of degradation estimable?

    Columns: [age*1{C2}, age*1{C3}, age*1{C4}, lap_in_run(=fuel index)].
    Within a stint lap_in_run == tyre_age, so the fuel column is exactly the
    sum of the three compound columns. Rank deficient by one.

    A linear functional c'beta is estimable iff c is orthogonal to the null
    space. We test a level (delta_C3) and a contrast (delta_C3 - delta_C2).
    """
    age = df["lap_in_run"].to_numpy().astype(float)
    comp = df["compound_abs"].to_numpy().astype(str)
    X = np.column_stack([_dummies(comp, COMPOUNDS) * age[:, None], age])

    rank = int(np.linalg.matrix_rank(X))
    _, s, vt = np.linalg.svd(X, full_matrices=True)
    null = vt[rank:]                                # rows span the null space

    def estimable(c: np.ndarray) -> float:
        return float(np.max(np.abs(null @ c))) if null.size else 0.0

    level = np.array([0.0, 1.0, 0.0, 0.0])          # delta_C3 alone
    contrast = np.array([-1.0, 1.0, 0.0, 0.0])      # delta_C3 - delta_C2

    return {
        "n_cols": X.shape[1], "rank": rank,
        "null_dim": X.shape[1] - rank,
        "level_leakage": estimable(level),
        "contrast_leakage": estimable(contrast),
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main(n_seeds: int = 200) -> int:
    truth = Truth()
    fp = truth.fuel_pace
    print("=" * 74)
    print("A3  MINIMAL SYNTHETIC RECOVERY")
    print("=" * 74)
    print(f"truth   beta_f={truth.beta_f} s/kg   kappa={truth.kappa} kg/lap")
    print(f"        fuel pace = beta_f*kappa = {fp:.4f} s/lap  <- what data sees")
    print(f"        tau_race  = {truth.tau_race:.4f} s/lap")
    print(f"        deg       = " + "  ".join(f"{c}:{truth.deg[c]:.4f}" for c in COMPOUNDS))
    print()

    rng = np.random.default_rng(20260830)
    race = minimal.simulate_race(rng, truth)
    prac = minimal.simulate_practice(rng, truth)

    # ---- 1. naive ------------------------------------------------------
    naive = est_naive(race)
    print("-" * 74)
    print("1  NAIVE within-stint slope        (expect deg - fuel_pace - tau, not deg)")
    print("-" * 74)
    for c in COMPOUNDS:
        pred = truth.deg[c] - fp - truth.tau_race
        print(f"   {c}  got {naive[c]:+.4f}   confounded target {pred:+.4f}   "
              f"true deg {truth.deg[c]:+.4f}   err {naive[c]-truth.deg[c]:+.4f}")

    # ---- 2. conventional ----------------------------------------------
    conv = est_conventional(race, truth, beta_assumed=0.030)
    bias_pred = -(truth.beta_f - 0.030) * truth.kappa - truth.tau_race
    print()
    print("-" * 74)
    print("2  CONVENTIONAL fix, assuming 0.030 s/kg when truth is 0.033")
    print("-" * 74)
    print(f"   predicted bias = -(0.033-0.030)*kappa - tau = {bias_pred:+.4f} s/lap")
    for c in COMPOUNDS:
        err = conv[c] - truth.deg[c]
        print(f"   {c}  got {conv[c]:+.4f}   true {truth.deg[c]:+.4f}   "
              f"err {err:+.4f}  ({100*err/truth.deg[c]:+.1f}%)")

    # ---- 3. race + practice -------------------------------------------
    rj = est_race_joint(race)
    print()
    print("-" * 74)
    print("3  RACE pit-stop design  ->  degradation separates")
    print("-" * 74)
    for c in COMPOUNDS:
        m, se = rj[c]
        lo, hi = m - 1.645 * se, m + 1.645 * se
        ok = "OK " if lo <= truth.deg[c] <= hi else "MISS"
        print(f"   {c}  {m:+.4f} +/- {se:.4f}   90% [{lo:+.4f},{hi:+.4f}]   "
              f"true {truth.deg[c]:+.4f}  {ok}")
    m, se = rj["_lap_fused"]
    print(f"   race-lap coef {m:+.4f}  == -(fuel_pace + tau) = "
          f"{-(fp + truth.tau_race):+.4f}   <- FUSED, as expected")

    print()
    print("-" * 74)
    print("4  PRACTICE arm  ->  unknown fuel load is the binding constraint")
    print("-" * 74)
    tau_true = -truth.tau_practice

    fe = est_practice_fe(prac)
    m, se = fe["tau_per_min"]
    lo, hi = m - 1.645 * se, m + 1.645 * se
    print(f"   run FIXED effects   tau/min {m:+.5f} +/- {se:.5f}  "
          f"true {tau_true:+.5f}  {'OK' if lo <= tau_true <= hi else 'MISS'}")
    print("                       -> between-run variation absorbed; clock is")
    print("                          collinear with lap-in-run inside a run.")

    mf = est_practice_mixed(prac)
    print(f"   run RANDOM effects  {mf.report('tau_per_min', tau_true)}")
    print(f"                       sigma_u {mf.sigma_u:.3f} s (unknown fuel load)"
          f"   sigma_e {mf.sigma_e:.3f} s")
    print(f"                       true sigma_u ~ beta_f*sd(F0) = "
          f"{truth.beta_f * (110 - 30) / np.sqrt(12):.3f} s")
    for c in COMPOUNDS:
        pm, pse = mf.get(f"run_slope_{c}")
        print(f"   {c}  run slope {pm:+.5f} +/- {pse:.5f}   == deg - fuel_pace = "
              f"{truth.deg[c]-fp:+.5f}")

    # ---- combined ------------------------------------------------------
    implied = np.mean([rj[c][0] - mf.get(f"run_slope_{c}")[0] for c in COMPOUNDS])
    print()
    print(f"   COMBINED  fuel_pace = deg(race) - runslope(practice) = "
          f"{implied:+.5f}   true {fp:+.5f}   err {implied-fp:+.5f}")

    # ---- how much practice data does tau actually need? ----------------
    print()
    print("   Track evolution is weakly identified from ONE session. Pooling:")
    for n_sess in (1, 3, 8, 20):
        frames = []
        r2 = np.random.default_rng(777)
        for k in range(n_sess):
            frames.append(minimal.simulate_practice(r2, truth, rnd=k + 1))
        pooled = pl.concat(frames)
        mk = est_practice_mixed(pooled)
        m, se = mk.get("tau_per_min")
        lo, hi = m - 1.645 * se, m + 1.645 * se
        print(f"     {n_sess:>2d} session(s), {pooled.height:>5d} laps  "
              f"tau {m:+.5f} +/- {se:.5f}  "
              f"{'OK' if lo <= tau_true <= hi else 'MISS'}")

    # ---- 5. rank -------------------------------------------------------
    rc = compound_rank_check(prac)
    print()
    print("-" * 74)
    print("5  COMPOUND INVARIANCE ALONE  ->  differences only, never levels")
    print("-" * 74)
    print(f"   design has {rc['n_cols']} columns, rank {rc['rank']}, "
          f"null dim {rc['null_dim']}")
    print(f"   delta_C3 level      leakage {rc['level_leakage']:.4f}  "
          f"-> {'NOT estimable' if rc['level_leakage'] > 1e-8 else 'estimable'}")
    print(f"   delta_C3 - delta_C2 leakage {rc['contrast_leakage']:.4f}  "
          f"-> {'NOT estimable' if rc['contrast_leakage'] > 1e-8 else 'estimable'}")

    # ---- 6. coverage ---------------------------------------------------
    print()
    print("-" * 74)
    print(f"6  CALIBRATION over {n_seeds} seeds  (gate: coverage in [0.85, 0.95])")
    print("-" * 74)
    hits = {c: 0 for c in COMPOUNDS}
    errs = {c: [] for c in COMPOUNDS}
    for s in range(n_seeds):
        r = np.random.default_rng(1000 + s)
        d = minimal.simulate_race(r, truth)
        e = est_race_joint(d)
        for c in COMPOUNDS:
            m, se = e[c]
            errs[c].append(m - truth.deg[c])
            if abs(m - truth.deg[c]) <= 1.645 * se:
                hits[c] += 1

    passed = True
    for c in COMPOUNDS:
        cov = hits[c] / n_seeds
        bias = float(np.mean(errs[c]))
        rmse = float(np.sqrt(np.mean(np.square(errs[c]))))
        good = 0.85 <= cov <= 0.95
        passed &= good
        print(f"   {c}  coverage {cov:.3f}   bias {bias:+.5f}   rmse {rmse:.5f}   "
              f"{'PASS' if good else 'FAIL'}")

    # naive/conventional coverage is definitionally zero -- show it
    print()
    from tyretrace import results
    results.record("a3", {
        "coverage": {c: hits[c] / n_seeds for c in COMPOUNDS},
        "bias": {c: float(np.mean(errs[c])) for c in COMPOUNDS},
        "fuel_pace_truth": truth.fuel_pace,
    }, gate="PASS" if passed else "FAIL")

    print("=" * 74)
    print(f"A3 GATE: {'PASS' if passed else 'FAIL'}")
    print("=" * 74)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
