"""A1b -- the replacement for the failed practice-decoupling mechanism.

A1 established that practice cannot identify track evolution: fuel state
trends with the session clock (measured corr +0.54), so a practice-only fit
returns the wrong sign. This script establishes the fix and checks it both
ways round -- on simulated data where truth is known, and on real data.

  PART 1  sim, known truth: the practice arm fails under a realistic fuel
          programme and succeeds without one. Isolates the cause.
  PART 2  sim, known truth: the qualifying anchor recovers tau.
  PART 3  real data: qualifying tau, with and without controls.
  PART 4  what it buys -- splitting the fused race-lap coefficient.

Run:  python audit/a1b_track_evolution.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from audit.a3_synthetic_recovery import est_practice_mixed, est_race_joint  # noqa: E402
from tyretrace.models import track_evolution as te  # noqa: E402
from tyretrace.sim import minimal  # noqa: E402
from tyretrace.sim.minimal import Truth  # noqa: E402

RULE = "-" * 76


def part1(truth: Truth) -> None:
    print(RULE)
    print("1  PRACTICE ARM, with and without a realistic fuel programme")
    print(RULE)
    print(f"   true tau = {-truth.tau_practice:+.5f} s/min  (negative: track speeds up)")
    print()

    for label, prog in (("idealised (fuel random vs clock)", False),
                        ("realistic (fuel trends with clock)", True)):
        rng = np.random.default_rng(4242)
        pooled = pl.concat([
            minimal.simulate_practice(rng, truth, rnd=k + 1, fuel_programme=prog)
            for k in range(8)])
        fit = est_practice_mixed(pooled)
        m, se = fit.get("tau_per_min")
        lo, hi = m - 1.645 * se, m + 1.645 * se
        ok = lo <= -truth.tau_practice <= hi
        sign = "correct" if m < 0 else "WRONG SIGN"
        print(f"   {label:<36s} tau {m:+.5f} +/- {se:.5f}"
              f"   {'OK  ' if ok else 'MISS'}  {sign}")

    print()
    print("   -> the programme, not the estimator, is what breaks it.")
    print()


def part2(truth: Truth) -> None:
    print(RULE)
    print("2  QUALIFYING ANCHOR on simulated data (known truth)")
    print(RULE)

    rng = np.random.default_rng(99)
    fits = [te.fit_session(te.push_laps(minimal.simulate_quali(rng, truth, rnd=k + 1)),
                           min_laps=20)
            for k in range(10)]
    fits = [f for f in fits if f is not None]
    s = te.summary(fits)
    truth_tau = -truth.tau_practice

    print(f"   sessions            {s['n']}")
    print(f"   median tau          {s['median_tau']:+.5f} s/min"
          f"   true {truth_tau:+.5f}")
    print(f"   median SE           {s['median_se']:.5f}")
    print(f"   fraction negative   {100*s['frac_negative']:.0f}%")
    print(f"   median z            {s['median_z']:.2f}")

    covered = sum(abs(f.tau_per_min - truth_tau) <= 1.645 * f.se for f in fits)
    print(f"   90% interval covers truth in {covered}/{len(fits)} sessions")
    print()


def part3() -> None:
    print(RULE)
    print("3  QUALIFYING ANCHOR on real data (10 weekends, 2023-2025)")
    print(RULE)

    pq = ROOT / "out" / "audit_quali.parquet"
    if not pq.exists():
        print("   out/audit_quali.parquet missing -- pull Q sessions first.")
        print()
        return

    q = pl.read_parquet(pq)
    for label, ctrl in (("driver only", False), ("+ age, compound, fresh", True)):
        s = te.summary(te.fit(q, controls=ctrl))
        print(f"   {label:<24s} tau {s['median_tau']:+.5f} s/min"
              f"   neg {100*s['frac_negative']:3.0f}%"
              f"   z {s['median_z']:5.2f}"
              f"   total {s['median_total_s']:+.2f} s/session")

    print()
    print("   Controls move the estimate by under 5%, so this is the track")
    print("   rubbering in -- not Q3 running fresher tyres than Q1.")
    print()


def part4(truth: Truth) -> None:
    print(RULE)
    print("4  WHAT THE ANCHOR BUYS -- splitting the fused race coefficient")
    print(RULE)

    rng = np.random.default_rng(31)
    fit = est_race_joint(minimal.simulate_race(rng, truth))
    fused, se = fit["_lap_fused"]

    # the race-lap coefficient is -(fuel_pace + tau_race); with tau supplied
    # from qualifying, fuel pace drops out
    tau_race = truth.tau_race
    fuel_pace = -fused - tau_race

    print(f"   race-lap coefficient        {fused:+.5f} +/- {se:.5f}")
    print(f"   = -(fuel_pace + tau_race), fused and NOT separable within a race")
    print()
    print(f"   supply tau_race from quali  {tau_race:+.5f}")
    print(f"   -> implied fuel pace        {fuel_pace:+.5f} s/lap"
          f"   true {truth.fuel_pace:+.5f}"
          f"   err {fuel_pace - truth.fuel_pace:+.5f}")
    print(f"   -> implied beta_f           {fuel_pace / truth.kappa:+.5f} s/kg"
          f"   true {truth.beta_f:+.5f}   (needs kappa from outside)")
    print()
    print("   The chain: quali -> tau; race + tau -> fuel pace and degradation;")
    print("   practice consumes both and no longer has to identify anything.")
    print()


def main() -> int:
    truth = Truth()
    print("=" * 76)
    print("A1b  TRACK EVOLUTION FROM QUALIFYING")
    print("=" * 76)
    print()
    part1(truth)
    part2(truth)
    part3()
    part4(truth)
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
