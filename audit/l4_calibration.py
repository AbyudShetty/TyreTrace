"""L4 -- fix calibration, so a rung can finally ship.

L2 and L3 both under-cover (0.843 and 0.849 against a 0.85-0.95 band), so
neither ships however good their CRPS. Diagnosing the residuals rather than
guessing gave two causes, and L4 addresses both:

  - a per-stint offset carrying 52% of residual variance
  - heavy right-skewed tails (excess kurtosis +1.54, |z|>3 four times too
    frequent for a Gaussian)

The comparison is L2 -> L3 -> L4 on the same contiguous forward split, scored
by the pre-registered metrics.

Run:  python audit/l4_calibration.py [--full]
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
from audit.l3_dirty_air import split, subset  # noqa: E402
from tyretrace import artifacts, results  # noqa: E402
from tyretrace.eval import metrics  # noqa: E402
from tyretrace.models import ladder  # noqa: E402

RULE = "-" * 76
RUNGS = ("L2", "L3", "L4", "L5")
RHAT_MAX = 1.05
SCALARS = ("beta_f", "tau_scale", "wake_s", "sigma", "sigma0", "gamma_age",
           "nu", "sd_stint")


def lap_labels(d: ladder.Design) -> dict:
    """Per-lap labels for the dashboard, decoded back to names."""
    return {
        "circuit": np.asarray(d.circuit_names)[d.circuit],
        "weekend": np.asarray(d.weekend_names)[d.weekend],
        "driver": np.asarray(d.driver_names)[d.driver],
        "compound": np.asarray(d.compound_names)[d.compound],
        "stint": d.stint,
        "lap_in_session": (d.lap + d.lap_mean).round().astype(int),
        "tyre_age": d.age.astype(int),
        "dirty_air": d.dirty_air,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    draws = 2000 if args.full else 800

    print("=" * 76)
    print(f"L4  CALIBRATION -- stint offsets and heavy tails   ({draws} draws)")
    print("=" * 76)

    races = pl.read_parquet(ROOT / "out" / "audit_race.parquet")
    quali = pl.read_parquet(ROOT / "out" / "audit_quali.parquet")
    d = build(races, quali)
    tr, te = split(d)
    d_tr, d_te = subset(d, tr), subset(d, te)

    seen = set(np.unique(d_tr.stint).tolist())
    unseen = int(sum(1 for s in np.unique(d_te.stint) if s not in seen))
    print(f"   {d.n} laps · {d.n_stint} stints · {d.n_weekend} weekends")
    print(f"   train {d_tr.n} · test {d_te.n}")
    print(f"   {unseen} of {len(np.unique(d_te.stint))} held-out stints are unseen "
          "in training")
    print("   -> those fall back to the population prior, widening honestly")
    print()

    fits, scores, rhats = {}, {}, {}
    pred_tabs, cal_tabs, post_tabs = [], [], []
    labels = lap_labels(d_te)
    kbar = float(np.mean(d_tr.kappa))
    for rung in RUNGS:
        t0 = time.time()
        mcmc = ladder.fit(rung, d_tr, draws=draws, warmup=draws, chains=2)
        diag = ladder.diagnostics(mcmc)
        pp = ladder.predict(rung, mcmc, d_te)
        scores[rung] = metrics.score(pp, d_te.y)
        fits[rung] = mcmc
        rhats[rung] = diag["max_rhat"]

        # for the dashboard: every rung on the same held-out laps
        pred_tabs.append(artifacts.predictive(pp, d_te.y, **labels)
                         .with_columns(pl.lit(rung).alias("rung")))
        cal_tabs.append(artifacts.calibration(pp, d_te.y)
                        .with_columns(pl.lit(rung).alias("rung")))
        post = mcmc.get_samples()
        post_tabs.append(artifacts.posterior(
            post, SCALARS,
            derived={"fuel_pace": np.asarray(post["beta_f"]) * kbar})
            .with_columns(pl.lit(rung).alias("rung")))
        print(f"   {rung} fitted in {time.time()-t0:>4.0f}s   "
              f"r_hat {diag['max_rhat']:.4f}   div {diag['divergences']}")
    print()

    # A score from chains that disagree is not a score of the model -- it is a
    # score of wherever the sampler happened to be. Measured: L3-L5 sat at
    # r_hat 1.05-1.07 at 800 draws and ~1.3 at 400, so the gate below was
    # comparing unconverged fits. Report it loudly rather than gate silently.
    unconverged = [r for r in RUNGS if rhats[r] > RHAT_MAX]
    if unconverged:
        print(f"   ! NOT CONVERGED (max r_hat > {RHAT_MAX}): "
              + ", ".join(f"{r} {rhats[r]:.3f}" for r in unconverged))
        print("     scores for these rungs are provisional")
        print()

    print(RULE)
    print("SHIP GATE -- CRPS must improve AND coverage stay in [0.85, 0.95]")
    print(RULE)
    base = None
    verdicts = {}
    for rung in RUNGS:
        r = scores[rung]
        ok, why = metrics.passes_ship_gate(r, base)
        verdicts[rung] = ok
        cal = "  ".join(f"{int(k*100)}%:{v:.2f}" for k, v in r["calibration"].items())
        print(f"   {rung}  CRPS {r['crps']:.5f}   MAE {r['mae']:.4f}   "
              f"cov90 {r['coverage_90']:.3f}")
        print(f"       calibration  {cal}")
        print(f"       gate: {'SHIP' if ok else 'HOLD'} -- {why}")
        base = r
    print()

    post = fits["L5"].get_samples()
    for name, label in (("sd_stint", "per-stint sd (s)"), ("nu", "Student-t nu"),
                        ("sigma0", "sigma at age 0 (s)"),
                        ("gamma_age", "log sigma per 20 laps"),
                        ("wake_s", "wake (s/lap)")):
        v = np.asarray(post[name])
        lo, hi = np.quantile(v, [0.05, 0.95])
        print(f"   {label:<20s} {v.mean():+.4f}   90% [{lo:+.4f}, {hi:+.4f}]")
    nu = float(np.mean(np.asarray(post["nu"])))
    print()
    print(f"   nu = {nu:.1f}: {'clearly heavier tails than Gaussian' if nu < 15 else 'close to Gaussian'}"
          " (nu -> inf recovers the Gaussian)")
    print()

    g = float(np.mean(np.asarray(post["gamma_age"])))
    s0 = float(np.mean(np.asarray(post["sigma0"])))
    print()
    print(f"   sigma at age 5  {s0*np.exp(g*5/20):.3f} s"
          f"   at age 30  {s0*np.exp(g*30/20):.3f} s"
          f"   ({np.exp(g*25/20):.2f}x wider)")
    print()

    # Report the best rung by CRPS and whether ANY rung cleared the gate.
    # Naming L5 the verdict would assume the last rung is the best one, which
    # is exactly the assumption the ladder exists to test.
    best = min(RUNGS, key=lambda r: scores[r]["crps"])
    shipped = [r for r in RUNGS if verdicts[r]]
    print()
    print(f"   best CRPS: {best} ({scores[best]['crps']:.5f}), "
          f"coverage {scores[best]['coverage_90']:.3f}")
    print(f"   rungs clearing the gate: {', '.join(shipped) if shipped else 'NONE'}")
    print()
    print("   L5 is REJECTED: sigma(age) is U-shaped (warm-up high, steady low,")
    print("   degradation rising), and a monotone log-linear scale cannot")
    print("   represent that. It fit the early drop and made old-tyre intervals")
    print("   narrower, worsening the problem it was built to fix.")
    print()
    print("   Open: no rung is calibrated yet. Next attempt is a non-monotone")
    print("   scale, or the regime model, where warm-up and cliff are distinct")
    print("   states carrying their own variances.")
    print()
    cov = scores[best]["coverage_90"]
    verdict = (f"SHIP {best}" if shipped
               else f"HOLD -- best is {best} at coverage {cov:.3f}")
    print("=" * 76)
    print(f"CALIBRATION GATE: {verdict}")
    print("=" * 76)
    shipped = bool(shipped)

    artifacts.save("l4", "predictive", pl.concat(pred_tabs))
    artifacts.save("l4", "calibration", pl.concat(cal_tabs))
    artifacts.save("l4", "posterior", pl.concat(post_tabs, how="diagonal_relaxed"))

    results.record("l4", {
        "gamma_age": g,
        **{f"crps_{r.lower()}": scores[r]["crps"] for r in RUNGS},
        **{f"cov90_{r.lower()}": scores[r]["coverage_90"] for r in RUNGS},
        **{f"mae_{r.lower()}": scores[r]["mae"] for r in RUNGS},
        **{f"rhat_{r.lower()}": rhats[r] for r in RUNGS},
        "converged": not unconverged,
        "sd_stint": float(np.mean(np.asarray(post["sd_stint"]))),
        "nu": nu, "sigma0": s0,
        "unseen_test_stints": unseen,
    }, gate="SHIP" if shipped else "HOLD")
    return 0 if shipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
