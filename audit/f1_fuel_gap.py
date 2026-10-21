"""F1 -- the fuel-pace gap. Is it fuel, or is it the track?

L2 identified fuel pace 0.032 s/lap against 0.055-0.060 from physics, and the
backtest (audit/t3_backtest.py) found out of sample that the race-learned
fuel effect is too small. This audit asks where the missing slope went.

THE OBSERVATION THAT FRAMES IT
------------------------------
A race only shows the FUSED lap slope:  lap_coef = -fuel_pace + track_per_lap.
L2 splits it as  -(beta_f * kappa) + s * track_quali,  and estimated s = 0.87:
the race track keeps improving at 87% of the qualifying rate (~0.02 s/lap).
Whatever L2 credits to the track, it subtracts from fuel. If the race track
in fact barely evolves, fuel pace is simply the fused slope -- and the gap
was never about fuel.

L2 also has no room for circuit-to-circuit scatter: each circuit's slope is
EXACTLY -(beta_f kappa + s tau). With ten circuits and tiny per-race standard
errors, that makes the posterior on s far more certain than ten races allow.

THE TEST
--------
Leave-one-race-out prediction of every race's fused slope, by four
pre-declared explanations, each with between-race scatter estimated:

    physics-only     -0.030 s/kg x kappa              (nothing fitted)
    fuel-only        -beta_f x kappa                  (beta_f fitted)
    fuel + track     -beta_f x kappa + s x track      (L2's form)
    mean             average of the other races       (no structure)

The explanation that predicts unseen races best is the one the data supports.
Plus a race-phase check: is the second half of a race slower than the first
half's slope predicts (tyre / fuel management, derate)?

Run:  python audit/f1_fuel_gap.py [--data audit|lake]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tyretrace import artifacts, results  # noqa: E402
from tyretrace.backtest import fp_race as B  # noqa: E402
from tyretrace.data import sources  # noqa: E402

RULE = "-" * 78
PHYSICS_S_PER_KG = 0.030
MIN_LAPS = 300            # a race with fewer clean laps gives a noisy slope


def scatter_wls(X: np.ndarray, y: np.ndarray, se: np.ndarray):
    """Weighted least squares with between-race scatter eta^2 by moments.

    Returns (coef, cov, eta). Without eta, races' tiny within-race standard
    errors would claim a cross-circuit relationship holds almost exactly.
    """
    se2 = np.square(se)
    eta2 = 0.0
    for _ in range(50):
        w = 1.0 / (se2 + eta2)
        XtW = X.T * w
        cov = np.linalg.inv(XtW @ X)
        coef = cov @ (XtW @ y)
        r2 = np.square(y - X @ coef)
        k = X.shape[1]
        new = max(float(np.sum(r2 - se2) / max(len(y) - k, 1)), 0.0)
        if abs(new - eta2) < 1e-14:
            break
        eta2 = new
    return coef, cov, float(np.sqrt(eta2))


def explanations(kappa, track):
    return {
        "physics-only": None,
        "fuel-only": np.column_stack([-kappa]),
        "fuel + track": np.column_stack([-kappa, track]),
        "mean": np.ones((len(kappa), 1)),
    }


def predict_loo(name, X, y, se, kappa):
    preds = np.empty(len(y))
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        if name == "physics-only":
            preds[i] = -PHYSICS_S_PER_KG * kappa[i]
            continue
        coef, _, _ = scatter_wls(X[m], y[m], se[m])
        preds[i] = X[i] @ coef
    return preds


def phase_check(race: pl.DataFrame) -> tuple[float, float]:
    """Fused slope fitted on each half of the race separately (same
    structure), so a late-race slowdown shows up as a shallower second half."""
    half = race["lap_in_session"].max() / 2
    a = B.race_lap_coef(race.filter(pl.col("lap_in_session") <= half))[0]
    b = B.race_lap_coef(race.filter(pl.col("lap_in_session") > half))[0]
    return a, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=("audit", "lake"), default="audit")
    args = ap.parse_args()
    d = sources.load(args.data)

    print("=" * 78)
    print(f"F1  THE FUEL-PACE GAP -- fuel, or track?   (data: {args.data})")
    print("=" * 78)

    _, race = B.prepare(d["practice"], d["race"])
    weekends = B.weekend_inputs(d["quali"], d["race"])
    dry = B.dry_weekends(d["race"])
    rows = []
    for k in sorted(k for k in weekends if k in dry):
        r = race.filter(pl.col("weekend") == k)
        if r.height < MIN_LAPS or r["compound_rel"].n_unique() < 2:
            continue
        coef, se, _ = B.race_lap_coef(r)
        h1, h2 = phase_check(r)
        w = weekends[k]
        rows.append({"weekend": k, "slope": coef, "se": se, "kappa": w.kappa,
                     "track": w.track_per_lap, "first_half": h1, "second_half": h2,
                     "laps": r.height})
    t = pl.DataFrame(rows)
    y, se = t["slope"].to_numpy(), t["se"].to_numpy()
    kappa, track = t["kappa"].to_numpy(), t["track"].to_numpy()
    n = len(y)
    print(f"   {n} dry races with >= {MIN_LAPS} clean laps and two compounds "
          f"({len(weekends) - len(dry & set(weekends))} wet or mixed excluded)")
    print(f"   fused race slope: mean {y.mean():+.4f}  median {np.median(y):+.4f}  "
          f"sd across races {y.std():.4f} s/lap")
    print(f"   physics for fuel ALONE at these kappas: "
          f"{-PHYSICS_S_PER_KG * kappa.mean():+.4f} (0.030 s/kg) to "
          f"{-0.035 * kappa.mean():+.4f} (0.035 s/kg)")
    print()

    # ---- full-sample fits, with honest uncertainty -------------------------
    print(RULE)
    print("   FULL-SAMPLE FITS (between-race scatter included)")
    print(RULE)
    fits = {}
    for name, X in explanations(kappa, track).items():
        if X is None:
            continue
        coef, cov, eta = scatter_wls(X, y, se)
        fits[name] = (coef, cov, eta)
        sd = np.sqrt(np.diag(cov))
        if name == "fuel-only":
            print(f"   fuel-only     beta_f {coef[0]:.4f} ± {sd[0]:.4f} s/kg  ->  fuel pace "
                  f"{coef[0] * kappa.mean():.4f} s/lap   scatter {eta:.4f}")
        elif name == "fuel + track":
            print(f"   fuel + track  beta_f {coef[0]:.4f} ± {sd[0]:.4f} s/kg   "
                  f"s {coef[1]:.2f} ± {sd[1]:.2f}   scatter {eta:.4f}")
            print(f"                 -> fuel pace {coef[0] * kappa.mean():.4f} s/lap; "
                  f"s 90% interval [{coef[1] - 1.645 * sd[1]:.2f}, {coef[1] + 1.645 * sd[1]:.2f}]")
    print()

    # ---- the test: leave-one-race-out ---------------------------------------
    print(RULE)
    print("   LEAVE-ONE-RACE-OUT: predict each race's slope from the others")
    print(RULE)
    loo = {}
    for name, X in explanations(kappa, track).items():
        p = predict_loo(name, X, y, se, kappa)
        e = p - y
        loo[name] = {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
                     "bias": float(np.mean(e))}
        t = t.with_columns(pl.Series(f"pred_{name.replace(' ', '').replace('+', '_').replace('-', '_')}", p))
    for name, s in sorted(loo.items(), key=lambda kv: kv[1]["mae"]):
        print(f"   {name:<14s} MAE {s['mae']:.4f}   RMSE {s['rmse']:.4f}   bias {s['bias']:+.4f} s/lap")
    best = min(loo, key=lambda k: loo[k]["mae"])
    print()

    # ---- race phase -----------------------------------------------------------
    h1, h2 = t["first_half"].to_numpy(), t["second_half"].to_numpy()
    diff = h2 - h1
    # medians: a half-race fit on a short or disrupted race can be absurd
    q25, q75 = np.percentile(diff, [25, 75])
    print(RULE)
    print("   RACE PHASE: fused slope in each half of the race (medians)")
    print(RULE)
    print(f"   first half {np.median(h1):+.4f}   second half {np.median(h2):+.4f}   "
          f"difference {np.median(diff):+.4f}  IQR [{q25:+.4f}, {q75:+.4f}] s/lap")
    late = q25 > 0                     # most races' second half shallower
    print(f"   -> {'second half SHALLOWER: a late-race slowdown is present' if late else 'no significant late-race slowdown'}")
    print()

    # ---- verdict -----------------------------------------------------------
    track_needed = best == "fuel + track"
    fuel_ok = best in ("physics-only", "fuel-only")
    print(RULE)
    if fuel_ok:
        print("   VERDICT: the race slope is explained by fuel ALONE. Crediting ~87% of")
        print("   qualifying's track evolution to the race (L2's s) does not predict")
        print("   unseen races better, and it is what pushed fuel pace below physics.")
        print("   The fuel-pace gap is a track-evolution attribution, not a fuel result.")
    elif track_needed:
        b, sv = fits["fuel + track"][0]
        sd_s = float(np.sqrt(fits["fuel + track"][1][1, 1]))
        print("   VERDICT: a small race track evolution earns its place out of sample:")
        print(f"   s = {sv:.2f} ± {sd_s:.2f} of the qualifying rate (L2 had claimed 0.87), and")
        print(f"   with it fuel is {b:.4f} s/kg against a 0.030-0.035 convention. The gap")
        print("   came from overstating race track evolution; with s estimated honestly")
        print("   it largely closes. Fuel alone does not win this test, so the gate fails.")
    else:
        print("   VERDICT: no structure beats the plain mean; kappa and track do not")
        print("   explain which races have steep slopes.")
    print(RULE)

    artifacts.save("fuel_gap", f"races_{args.data}", t)
    task = "fuel_gap" if args.data == "audit" else "fuel_gap_lake"
    ff = fits.get("fuel-only")
    results.record(task, {
        "races": n, "mean_slope": float(y.mean()), "best": best,
        **{f"loo_mae_{k.replace(' ', '').replace('+', '_').replace('-', '_')}": v["mae"]
           for k, v in loo.items()},
        "beta_f_fuel_only": float(ff[0][0]) if ff else None,
        "beta_f_fuel_only_sd": float(np.sqrt(ff[1][0, 0])) if ff else None,
        "s_fuel_track": float(fits["fuel + track"][0][1]),
        "s_fuel_track_sd": float(np.sqrt(fits["fuel + track"][1][1, 1])),
        "late_slowdown": bool(late), "half_diff": float(np.mean(diff)),
    }, gate="PASS" if fuel_ok else "FAIL")
    print(f"\nF1 GATE: {'PASS' if fuel_ok else 'FAIL'} -- best explanation: {best}")
    return 0 if fuel_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
