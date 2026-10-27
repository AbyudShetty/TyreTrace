"""L3 -- dirty air as a covariate, and the ship gate against L2.

Two questions, and they are separate:

  1. Does wake exposure EARN its place? Held-out CRPS must improve while
     coverage stays in band (protocol section 5).
  2. Does it explain the fuel-pace shortfall L2 exposed? A falsifiable
     prediction says NO -- the field spreads out over a race, so early laps
     are dirtier and slower, which pushes the identified fuel pace further
     from physics rather than toward it.

A component can pass 1 and fail 2. That is the useful case: it means we have
a better model and a still-open question, which is worth more than a fudge
that closes the gap by construction.

Run:  python audit/l3_dirty_air.py [--full]
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

from audit.l2_tau_scale import build  # noqa: E402
from tyretrace.eval import metrics  # noqa: E402
from tyretrace.models import ladder  # noqa: E402

RULE = "-" * 76
PHYSICS_LO, PHYSICS_HI = 0.055, 0.060      # beta_f 0.030-0.035 s/kg at kappa ~1.8


def split(d: ladder.Design, frac: float = 0.7):
    """Contiguous forward split, per protocol section 2. Same circuits, so the
    score is about extrapolating pace, not guessing a circuit's level."""
    keep = np.zeros(d.n, dtype=bool)
    for c in np.unique(d.circuit):
        m = d.circuit == c
        cut = np.quantile(d.lap[m], frac)
        keep |= m & (d.lap <= cut)
    return keep, ~keep


def subset(d: ladder.Design, m: np.ndarray) -> ladder.Design:
    out = ladder.Design(
        y=d.y[m], lap=d.lap[m], age=d.age[m], driver=d.driver[m],
        circuit=d.circuit[m], compound=d.compound[m],
        n_driver=d.n_driver, n_circuit=d.n_circuit, n_compound=d.n_compound,
        driver_names=d.driver_names, circuit_names=d.circuit_names,
        compound_names=d.compound_names, lap_mean=d.lap_mean)
    out.kappa = d.kappa[m]
    out.tau_quali = d.tau_quali[m]
    out.dirty_air = d.dirty_air[m]
    out.stint = d.stint[m]
    out.n_stint = d.n_stint
    out.stint_circuit = d.stint_circuit
    out.weekend = d.weekend[m]
    out.n_weekend = d.n_weekend
    out.weekend_names = d.weekend_names
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--data", choices=("audit", "lake"), default="audit")
    ap.add_argument("--new-only", action="store_true",
                    help="lake weekends NOT in the original ten: out-of-sample confirmation")
    args = ap.parse_args()
    draws = 2000 if args.full else 800
    task = "l3" if args.data == "audit" else ("l3_lake_new" if args.new_only else "l3_lake")

    print("=" * 76)
    print(f"L3  DIRTY AIR AS A COVARIATE   ({draws} draws, data: {args.data}"
          f"{', new weekends only' if args.new_only else ''})")
    print("=" * 76)

    from tyretrace.data import sources
    src = sources.load(args.data)
    races, quali = src["race"], src["quali"]
    from tyretrace.backtest.fp_race import dry_weekends
    dry = dry_weekends(races)
    races = races.filter(pl.col("session_uid").str.slice(0, 7).is_in(list(dry)))
    quali = quali.filter(pl.col("session_uid").str.slice(0, 7).is_in(list(dry)))
    if args.new_only:
        # The 2026-10-21 slope fix was motivated by analysis of the original
        # ten weekends; confirming it must use weekends that analysis never saw.
        seen = {u[:7] for u in sources.load("audit")["race"]["session_uid"].unique().to_list()}
        keep = pl.col("session_uid").str.slice(0, 7).is_in(list(seen)).not_()
        races, quali = races.filter(keep), quali.filter(keep)
        print(f"   excluded the {len(seen)} original weekends")
    d = build(races, quali)

    in_wake = float(np.mean(d.dirty_air > np.exp(-1.0)))   # within ~2 s
    print(f"   {d.n} laps · {d.n_weekend} weekends")
    print(f"   dirty_air  median {np.median(d.dirty_air):.3f}   "
          f"{100*in_wake:.0f}% of laps within ~2 s of a car ahead")
    print()

    tr_m, te_m = split(d)
    d_tr, d_te = subset(d, tr_m), subset(d, te_m)
    print(f"   train {d_tr.n} laps (first 70% of each race)")
    print(f"   test  {d_te.n} laps (last 30%)")
    print()

    fits, scores, diagnostics = {}, {}, {}
    for rung in ("L2", "L3"):
        t0 = time.time()
        mcmc = ladder.fit(rung, d_tr, draws=draws, warmup=draws, chains=2)
        diag = ladder.diagnostics(mcmc)
        diagnostics[rung] = {k:v for k,v in diag.items() if k!='summary'}
        pp = ladder.predict(rung, mcmc, d_te)
        scores[rung] = metrics.score(pp, d_te.y)
        fits[rung] = mcmc
        print(f"   {rung} fitted in {time.time()-t0:.0f}s   "
              f"r_hat {diag['max_rhat']:.4f}   div {diag['divergences']}")
    print()

    # ---- question 1: does it earn its place? ---------------------------
    print(RULE)
    print("1  SHIP GATE -- does wake exposure earn its place?")
    print(RULE)
    base = None
    shipped = False
    for rung in ("L2", "L3"):
        r = scores[rung]
        ok, why = metrics.passes_ship_gate(r, base)
        cal = "  ".join(f"{int(k*100)}%:{v:.2f}" for k, v in r["calibration"].items())
        print(f"   {rung}  CRPS {r['crps']:.5f}   MAE {r['mae']:.4f}   "
              f"cov90 {r['coverage_90']:.3f}")
        print(f"       calibration  {cal}")
        print(f"       gate: {'SHIP' if ok else 'HOLD'} -- {why}")
        if rung == "L2":
            base = r
        else:
            shipped = ok and all(x['divergences']==0 and x['max_rhat']<=1.05 for x in diagnostics.values())
    print()

    w = np.asarray(fits["L3"].get_samples()["wake_s"])
    lo, hi = np.quantile(w, [0.05, 0.95])
    print(f"   wake cost {w.mean():+.4f} s/lap in full wake   90% [{lo:+.4f}, {hi:+.4f}]")
    print()

    # ---- question 2: does it close the fuel-pace gap? ------------------
    print(RULE)
    print("2  DOES IT EXPLAIN THE FUEL-PACE SHORTFALL?  (prediction: no)")
    print(RULE)
    for rung in ("L2", "L3"):
        post = fits[rung].get_samples()
        fp_ = np.asarray(post["beta_f"]) * float(np.mean(d_tr.kappa))
        lo, hi = np.quantile(fp_, [0.05, 0.95])
        print(f"   {rung}  fuel pace {fp_.mean():+.5f} s/lap   90% [{lo:+.5f}, {hi:+.5f}]")
    print(f"   physics expectation  {PHYSICS_LO:.3f} - {PHYSICS_HI:.3f} s/lap")
    print()

    f2 = float(np.mean(np.asarray(fits["L2"].get_samples()["beta_f"])) * np.mean(d_tr.kappa))
    f3 = float(np.mean(np.asarray(fits["L3"].get_samples()["beta_f"])) * np.mean(d_tr.kappa))
    moved = f3 - f2
    away = abs(f3 - PHYSICS_LO) > abs(f2 - PHYSICS_LO)
    print(f"   fuel pace moved {moved:+.5f} s/lap when wake was controlled for")
    print(f"   -> {'AWAY from' if away else 'TOWARD'} physics, as predicted"
          f" ({'prediction holds' if away else 'PREDICTION FAILED'})")
    print()
    print("   So dirty air is real and worth modelling, but it is not the")
    print("   missing slowdown. Remaining candidates, none yet measured:")
    print("     - tyre/fuel management and engine derate late in a race")
    print("     - track degradation (marbles, rubber off-line) late on")
    print("     - beta_f genuinely below the 0.030 s/kg convention for these")
    print("       cars, which would be a result rather than a bug")
    print()

    from tyretrace import results
    results.record(task, {
        "wake_s_per_lap": float(w.mean()),
        "crps_l2": scores["L2"]["crps"], "crps_l3": scores["L3"]["crps"],
        "cov90_l2": scores["L2"]["coverage_90"],
        "cov90_l3": scores["L3"]["coverage_90"],
        "fuel_pace_l2": f2, "fuel_pace_l3": f3,
        "prediction_held": bool(away),
        "diagnostics": diagnostics,
        "converged": all(x['divergences']==0 and x['max_rhat']<=1.05 for x in diagnostics.values()),
    }, gate="SHIP" if shipped else "HOLD")

    ok = all(x['divergences']==0 and x['max_rhat']<=1.05 for x in diagnostics.values())
    print("=" * 76)
    print(f"L3 GATE: {'PASS' if (ok and shipped) else 'HOLD'}"
          f"   (converged: {ok}, ships: {shipped})")
    print("=" * 76)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
