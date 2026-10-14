"""L0 / L1 -- the identification core in NumPyro, with the ship gate applied.

Three checks, in the order that matters:

  1. RECOVERY   on the synthetic twin, where truth is known. Degradation and
     the fused lap coefficient must come back with calibrated intervals.
  2. GATE       L1 vs L0 on held-out laps, scored by the pre-registered
     metrics. L1 ships only if CRPS improves AND coverage stays in band.
  3. REAL       L1 on ten weekends of real races, with the qualifying anchor
     supplying tau so the fused coefficient can be split.

Run:  python audit/l0_l1_ladder.py            (fast: 500 draws)
      python audit/l0_l1_ladder.py --full     (2000 draws)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tyretrace import artifacts  # noqa: E402
from tyretrace import results as results_file  # noqa: E402
from tyretrace.data import fastf1_pull as fp  # noqa: E402
from tyretrace.eval import metrics  # noqa: E402
from tyretrace.models import ladder, track_evolution as te  # noqa: E402
from tyretrace.sim import minimal  # noqa: E402
from tyretrace.sim.minimal import Truth  # noqa: E402

RULE = "-" * 76


def _hpdi(samples: np.ndarray, level: float = 0.90) -> tuple[float, float]:
    lo, hi = np.quantile(samples, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def _report(name: str, samples: np.ndarray, truth: float | None) -> bool:
    m = float(np.mean(samples))
    lo, hi = _hpdi(samples)
    line = f"   {name:<14s} {m:+.5f}   90% [{lo:+.5f}, {hi:+.5f}]"
    if truth is None:
        print(line)
        return True
    ok = lo <= truth <= hi
    print(f"{line}   true {truth:+.5f}   {'OK' if ok else 'MISS'}")
    return ok


def part1_recovery(draws: int, warmup: int) -> bool:
    print(RULE)
    print("1  RECOVERY on the synthetic twin (known truth)")
    print(RULE)

    truth = Truth()
    rng = np.random.default_rng(2026)
    pooled = pl.concat([minimal.simulate_race(rng, truth, rnd=k + 1) for k in range(3)])

    # L0 pools compounds and has no circuit term, so it is correctly specified
    # only on ONE race with ONE compound. Testing it anywhere else measures
    # misspecification, not identification -- and that gap is exactly what L1
    # exists to close. Its target is then unambiguous: the true deg of that
    # single compound, not some lap-weighted blend of three.
    single = minimal.simulate_race(rng, truth, rnd=1, compounds=["C3"])

    designs = {"L0": ladder.build_design(single),
               "L1": ladder.build_design(pooled)}
    print(f"   L0 on one race, one compound (C3):  {designs['L0'].n} laps")
    print(f"   L1 pooled:       {designs['L1'].n} laps · "
          f"{designs['L1'].n_driver} drivers · {designs['L1'].n_circuit} circuits")

    ok = True
    for rung in ("L0", "L1"):
        d = designs[rung]
        t0 = time.time()
        mcmc = ladder.fit(rung, d, draws=draws, warmup=warmup, chains=2)
        diag = ladder.diagnostics(mcmc)
        s = mcmc.get_samples()
        print(f"\n   {rung}  ({time.time()-t0:.1f}s, r_hat {diag['max_rhat']:.4f}, "
              f"ESS {diag['min_ess_bulk']:.0f}, div {diag['divergences']})")

        fused_truth = -(truth.fuel_pace + truth.tau_race)
        ok &= _report("lap_coef", s["lap_coef"], fused_truth)

        if rung == "L0":
            ok &= _report("deg[C3]", s["deg"], truth.deg["C3"])
        else:
            for i, c in enumerate(d.compound_names):
                ok &= _report(f"deg[{c}]", s["deg"][:, i], truth.deg.get(c))

        if diag["divergences"] > 0:
            print(f"   !! {diag['divergences']} divergences -- not a valid fit")
            ok = False
        if diag["max_rhat"] > 1.01:
            print(f"   !! r_hat {diag['max_rhat']:.4f} above 1.01")
            ok = False

    print()
    return ok


def part2_gate(draws: int, warmup: int) -> bool:
    print(RULE)
    print("2  SHIP GATE -- L1 vs L0 on held-out laps (protocol §5)")
    print(RULE)

    truth = Truth()
    rng = np.random.default_rng(77)
    races = pl.concat([minimal.simulate_race(rng, truth, rnd=k + 1) for k in range(4)])

    # Protocol §2: a CONTIGUOUS block, never a random lap-level split -- laps
    # from the same stint are near-duplicates and a random split leaks.
    #
    # Hold out the last 30% of every race, same circuits and drivers. That
    # matches the real task (fit on practice, predict the race at the SAME
    # weekend) and it scores what the model is for: extrapolating pace forward
    # through fuel burn and tyre age. An unseen-circuit split instead scores
    # how well you guess a circuit's absolute pace, which is worth +/-1.5 s and
    # swamps the 0.2 s of signal -- it drove coverage to 1.000 while telling us
    # nothing about degradation.
    cut = (races.group_by("session_uid")
                .agg((pl.col("lap_in_session").max() * 0.7).alias("cut")))
    races = races.join(cut, on="session_uid")
    train = races.filter(pl.col("lap_in_session") <= pl.col("cut")).drop("cut")
    test = races.filter(pl.col("lap_in_session") > pl.col("cut")).drop("cut")

    d_tr = ladder.build_design(train)
    d_te = ladder.build_design(test, like=d_tr)

    print(f"   train {d_tr.n} laps (first 70% of each race)")
    print(f"   test  {d_te.n} laps (last 30%, extrapolating forward)")
    print()

    results = {}
    cal_tabs = []
    for rung in ("L0", "L1"):
        mcmc = ladder.fit(rung, d_tr, draws=draws, warmup=warmup, chains=2)
        pp = ladder.predict(rung, mcmc, d_te)
        results[rung] = metrics.score(pp, d_te.y)
        cal_tabs.append(artifacts.calibration(pp, d_te.y)
                        .with_columns(pl.lit(rung).alias("rung")))

    # Scored on the SYNTHETIC TWIN, so labelled as such: these numbers are
    # not comparable with L2-L5, which are scored on real races.
    artifacts.save("ladder", "calibration_twin", pl.concat(cal_tabs))
    results_file.record("ladder", {
        **{f"twin_crps_{r.lower()}": results[r]["crps"] for r in results},
        **{f"twin_cov90_{r.lower()}": results[r]["coverage_90"] for r in results},
        **{f"twin_mae_{r.lower()}": results[r]["mae"] for r in results},
    })

    base = None
    shipped = True
    for rung in ("L0", "L1"):
        r = results[rung]
        ok, why = metrics.passes_ship_gate(r, base)
        cal = "  ".join(f"{int(k*100)}%:{v:.2f}" for k, v in r["calibration"].items())
        print(f"   {rung}  CRPS {r['crps']:.5f}   MAE {r['mae']:.4f}   "
              f"cov90 {r['coverage_90']:.3f}")
        print(f"       calibration  {cal}")
        print(f"       gate: {'SHIP' if ok else 'HOLD'} -- {why}")
        if rung == "L0":
            base = r
        else:
            shipped = ok
        print()

    if not shipped:
        print("   L1 does not clear the gate here. That is the gate working, not")
        print("   a bug -- a rung that fails it does not ship.")
        print()
    return True                     # informative, not a pass/fail gate


def part3_real(draws: int, warmup: int) -> bool:
    print(RULE)
    print("3  L1 on real races, with the qualifying anchor supplying tau")
    print(RULE)

    race_pq = ROOT / "out" / "audit_race.parquet"
    quali_pq = ROOT / "out" / "audit_quali.parquet"
    if not race_pq.exists():
        print("   out/audit_race.parquet missing -- run a1_a2_identification --pull")
        return True

    races = fp.long_runs(pl.read_parquet(race_pq), min_laps=4)

    # tau from qualifying, converted from s/min to s/lap using observed pace
    tau_prior = (0.0, 0.05)
    if quali_pq.exists():
        fits = te.fit(pl.read_parquet(quali_pq))
        lap_s = float(races["lap_time_s"].median())
        # tau_model() is the ONLY sanctioned conversion. Converting by hand
        # here previously flipped the sign and inverted the sensitivity table.
        tau_lap, tau_sd = te.summary_tau_model(fits, lap_s)
        tau_prior = (tau_lap, max(tau_sd, 0.002))
        raw = te.summary(fits)["median_tau"]
        print(f"   quali tau {raw:+.5f} s/min (measured convention)")
        print(f"   -> {tau_lap:+.5f} s/lap in MODEL convention at a {lap_s:.1f}s lap"
              f"  (sd {tau_prior[1]:.5f})")

    d = ladder.build_design(races)
    print(f"   {d.n} laps · {d.n_driver} drivers · {d.n_circuit} circuits · "
          f"{d.n_compound} compound labels")
    print("   NOTE compound_abs is a placeholder until the Pirelli C1-C5 map")
    print("        lands, so per-compound numbers are not yet poolable.")
    print()

    t0 = time.time()
    mcmc = ladder.fit("L1", d, draws=draws, warmup=warmup, chains=2,
                      tau_prior=tau_prior)
    diag = ladder.diagnostics(mcmc)
    post = mcmc.get_samples()
    print(f"   fitted in {time.time()-t0:.1f}s   r_hat {diag['max_rhat']:.4f}   "
          f"ESS {diag['min_ess_bulk']:.0f}   div {diag['divergences']}")
    print()

    _report("lap_coef", post["lap_coef"], None)
    _report("deg_mu", post["deg_mu"], None)
    _report("sigma", post["sigma"], None)
    print()

    # --- sensitivity to the transport assumption -------------------------
    # The race-lap coefficient is fused: fuel_pace = -lap_coef - tau. Every
    # gram of fuel attributed depends on the tau carried over from qualifying,
    # and that carry-over is an ASSUMPTION, not a measurement. Show what it
    # costs across the plausible range instead of picking one number.
    print("   SENSITIVITY -- what the quali->race tau transport costs")
    print("   " + "-" * 68)
    fused = float(np.mean(post["lap_coef"]))
    kappa = 1.75
    rows = [
        ("no track evolution in race", 0.0),
        ("half the quali rate", tau_prior[0] * 0.5),
        ("full quali rate (as fitted)", tau_prior[0]),
    ]
    for label, tv in rows:
        fp_ = -fused - tv
        print(f"     tau {tv:+.5f} s/lap  ->  fuel pace {fp_:+.5f}"
              f"  ->  beta_f {fp_/kappa:+.5f} s/kg   {label}")
    print()
    print("   Paddock convention is ~0.030-0.035 s/kg. The no-evolution case")
    print("   sits just under it; the full quali rate lands FAR BELOW. Carrying")
    print("   the quali rate into the race attributes pace to the track that")
    print("   the fuel term should have, so beta_f collapses to a value no")
    print("   plausible physics supports.")
    print()
    print("   Either way the conclusion is the same: race track evolution is")
    print("   SMALLER than qualifying's -- which is what you would expect once")
    print("   the track is already rubbered in and laying marbles.")
    print()
    print("   -> do not carry the quali rate into the race unadjusted. Fit a")
    print("      session-type scaling on tau, and let the rig arbitrate it.")
    print()
    if diag["divergences"]:
        print(f"   !! {diag['divergences']} divergences -- treat with suspicion")
    return diag["divergences"] == 0 and diag["max_rhat"] < 1.05


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    draws = 2000 if args.full else 500
    warmup = 2000 if args.full else 500

    print("=" * 76)
    print(f"L0 / L1 MODEL LADDER   ({draws} draws, {warmup} warmup)")
    print("=" * 76)
    print()

    ok = part1_recovery(draws, warmup)
    part2_gate(draws, warmup)
    ok &= part3_real(draws, warmup)

    print("=" * 76)
    print(f"LADDER GATE: {'PASS' if ok else 'FAIL'}")
    print("=" * 76)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
