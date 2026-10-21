"""T3 -- the FP -> race backtest. Does the fuel / degradation split hold up?

Rolling origin, strictly causal (protocol section 2). For each weekend W with
at least MIN_EARLIER races before it:

    fuel effect and race/quali scale   <- races strictly before W
    degradation per compound           <- W's practice, within-run slopes
    track evolution                    <- W's qualifying
    each driver's pace level           <- their first ANCHOR_LAPS green race laps

then every later race lap is forecast, by four methods from identical inputs.
The laps AFTER a pit stop are where the methods disagree, because only there
does the fuel effect enter separately from degradation.

Gate: PASS if the identified method beats both the conventional and the naive
method on CRPS over post-stop laps -- the subset this test exists for. A FAIL
here would be a finding about the method, not a bug, and is reported as one.

Run:  python audit/t3_backtest.py
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
from tyretrace.eval import metrics  # noqa: E402

RULE = "-" * 78


def score(fc: B.Forecast, mask: np.ndarray | None = None) -> dict:
    y = fc.laps["lap_time_s"].to_numpy()
    d = fc.draws
    if mask is not None:
        if not mask.any():
            return {}
        y, d = y[mask], d[:, mask]
    s = metrics.score(d, y)
    s["bias"] = float(np.mean(np.median(d, axis=0) - y))
    s["n"] = int(len(y))
    return s


def stint_mae(fc: B.Forecast) -> float:
    """Protocol section 3's legible metric: per-stint mean pace, s/lap."""
    t = fc.laps.with_columns(pl.Series("pred", np.median(fc.draws, axis=0)))
    g = t.group_by("driver", "stint_id").agg(pl.col("pred").mean(), pl.col("lap_time_s").mean())
    return float((g["pred"] - g["lap_time_s"]).abs().mean())


# Which method the gate judges. On the original ten weekends it is the method
# as first specified, and that FAIL stands as recorded. On the lake it is the
# fuel-only method, pre-registered 2026-10-21 before the lake was pulled, so
# its verdict is genuinely out of sample for every weekend beyond the ten.
GATED = {"audit": "identified", "lake": "fuel-only"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=("audit", "lake"), default="audit")
    ap.add_argument("--traffic", action="store_true",
                    help="traffic-aware anchors (pre-registered for the lake run)")
    args = ap.parse_args()
    task = "backtest" if args.data == "audit" else "backtest_lake"
    gated = GATED[args.data]

    print("=" * 78)
    print(f"T3  FP -> RACE BACKTEST -- rolling origin, strictly causal   (data: {args.data})")
    print("=" * 78)

    d = sources.load(args.data)
    practice, quali, race_raw = d["practice"], d["quali"], d["race"]

    prac, race = B.prepare(practice, race_raw)
    weekends = B.weekend_inputs(quali, race_raw)
    dry = B.dry_weekends(race_raw)               # wet races are not fuel/tyre running
    keys = sorted(k for k in weekends if k in dry)   # "YYYY_RR" sorts chronologically
    print(f"   {len(keys)} dry weekends ({len(weekends) - len(keys)} wet or mixed excluded)")

    # each race's own fused coefficient -- used ONLY by later weekends
    rc = {k: B.race_lap_coef(race.filter(pl.col("weekend") == k)) for k in keys}
    rd = {k: B.race_degradation(race.filter(pl.col("weekend") == k)) for k in keys}
    tf = B.traffic_history(race, keys) if args.traffic else {}

    per_wk, pred_tabs, ft_rows, params = [], [], [], []
    pooled: dict[str, list] = {m: [] for m in B.METHODS}
    for i, k in enumerate(keys):
        if i < B.MIN_EARLIER:
            continue
        wk = weekends[k]
        earlier = keys[:i]
        ft = B.fit_fuel_track([rc[e][0] for e in earlier], [rc[e][1] for e in earlier],
                              [weekends[e] for e in earlier])
        sigma = float(np.median([rc[e][2] for e in earlier]))
        prior = B.degradation_prior([rd[e] for e in earlier])
        fo = B.fit_fuel_only([rc[e][0] for e in earlier], [rc[e][1] for e in earlier],
                             [weekends[e] for e in earlier])
        fcs = B.run_weekend(wk, prac.filter(pl.col("weekend") == k),
                            race.filter(pl.col("weekend") == k), ft, sigma, seed=i,
                            prior=prior, fuel_only=fo,
                            traffic=B.traffic_prior([tf[e] for e in earlier if e in tf])
                            if args.traffic else None)

        f, f_sd = ft.fuel_pace(wk)
        fb = [c for c, v in next(iter(fcs.values())).deg.items() if v[2] == "prior"]
        # EVALUATION ONLY, after the forecast is made: W's own race slope, to
        # see which method's slope assumption was right. Never fed back.
        lc_id = ft.lap_coef(wk)[0]
        ft_rows.append({"weekend": k, "races_used": len(earlier), "beta_f": ft.beta_f,
                        "slope_actual": rc[k][0], "slope_identified": lc_id,
                        "slope_conventional": -B.CONVENTIONAL_S_PER_KG * wk.kappa,
                        "slope_mean_earlier": float(np.mean([rc[e][0] for e in earlier])),
                        "beta_f_sd": float(np.sqrt(ft.cov[0, 0])), "s": ft.s,
                        "s_sd": float(np.sqrt(ft.cov[1, 1])), "fuel_pace": f,
                        "fuel_pace_sd": f_sd, "kappa": wk.kappa,
                        "prior_only_compounds": ",".join(fb),
                        "deg_prior": prior[0], "deg_prior_sd": prior[1]})
        print(f"\n   {k}  ({len(earlier)} earlier races)   fuel pace {f:.4f} ± {f_sd:.4f} s/lap"
              f"   s {ft.s:.2f}   sigma {sigma:.2f}"
              + f"   deg prior {prior[0]:.3f}+/-{prior[1]:.3f}"
              + (f"   no practice: {', '.join(fb)}" if fb else ""))
        for m, fc in fcs.items():
            # the point parameters each method forecast with -- what the
            # post-race decomposition (audit/p1_postrace.py) explains errors by
            params += [{"weekend": k, "method": m, "lap_coef": fc.lap_coef[0],
                        "compound": c, "deg": v[0], "deg_sd": v[1], "deg_source": v[2]}
                       for c, v in fc.deg.items()]
            after = fc.laps["after_stop"].to_numpy()
            a, b = score(fc), score(fc, after)
            per_wk.append({"weekend": k, "method": m, "crps": a["crps"], "mae": a["mae"],
                           "cov90": a["coverage_90"], "stint_mae": stint_mae(fc),
                           "crps_after": b.get("crps"), "bias_after": b.get("bias"),
                           "cov90_after": b.get("coverage_90"),
                           "n": a["n"], "n_after": b.get("n", 0)})
            pooled[m].append(fc)
            pred_tabs.append(artifacts.predictive(
                fc.draws, fc.laps["lap_time_s"].to_numpy(),
                weekend=np.full(fc.laps.height, k), method=np.full(fc.laps.height, m),
                driver=fc.laps["driver"].to_numpy(), stint=fc.laps["stint_id"].to_numpy(),
                lap_in_session=fc.laps["lap_in_session"].to_numpy(),
                tyre_age=fc.laps["tyre_age"].to_numpy(),
                compound=fc.laps["compound"].to_numpy(),
                after_stop=fc.laps["after_stop"].to_numpy()))
            print(f"      {m:<13s} CRPS {a['crps']:.3f}  cov90 {a['coverage_90']:.2f}"
                  f"  | after stop: CRPS {b.get('crps', float('nan')):.3f}"
                  f"  bias {b.get('bias', float('nan')):+.3f}  ({b.get('n', 0)} laps)")

    # ---- pooled over every evaluated lap --------------------------------
    print()
    print(RULE)
    print(f"   {'method':<13s} {'CRPS':>7s} {'MAE':>7s} {'cov90':>6s} {'stint':>7s}"
          f" | {'CRPS':>7s} {'bias':>7s} {'cov90':>6s}   (after a stop)")
    print(RULE)
    summary = {}
    for m, fcs in pooled.items():
        if not fcs:
            continue
        y = np.concatenate([f.laps["lap_time_s"].to_numpy() for f in fcs])
        aft = np.concatenate([f.laps["after_stop"].to_numpy() for f in fcs])
        crps = np.concatenate([metrics.crps_ensemble(f.draws, f.laps["lap_time_s"].to_numpy())
                               for f in fcs])
        med = np.concatenate([np.median(f.draws, axis=0) for f in fcs])
        inside = np.concatenate([
            (f.laps["lap_time_s"].to_numpy() >= np.quantile(f.draws, 0.05, axis=0))
            & (f.laps["lap_time_s"].to_numpy() <= np.quantile(f.draws, 0.95, axis=0))
            for f in fcs])
        st = float(np.average([stint_mae(f) for f in fcs], weights=[f.laps.height for f in fcs]))
        summary[m] = {"crps": float(crps.mean()), "mae": float(np.abs(med - y).mean()),
                      "cov90": float(inside.mean()), "stint_mae": st,
                      "crps_after": float(crps[aft].mean()),
                      "bias_after": float((med - y)[aft].mean()),
                      "cov90_after": float(inside[aft].mean()),
                      "n": int(len(y)), "n_after": int(aft.sum())}
        s = summary[m]
        print(f"   {m:<13s} {s['crps']:7.3f} {s['mae']:7.3f} {s['cov90']:6.2f} {s['stint_mae']:7.3f}"
              f" | {s['crps_after']:7.3f} {s['bias_after']:+7.3f} {s['cov90_after']:6.2f}")
    n_w = len({r['weekend'] for r in per_wk})
    print(f"   {n_w} weekends · {summary['identified']['n']} laps · "
          f"{summary['identified']['n_after']} after a stop · stint = per-stint pace MAE, s/lap")
    print()

    idf, conv, nai = (summary[m]["crps_after"] for m in (gated, "conventional", "naive"))
    ok = idf < conv and idf < nai
    print(f"   gate judges: {gated}")
    print(f"   after a stop, {gated} vs conventional: CRPS {100*(1-idf/conv):+.1f}%")
    print(f"   after a stop, {gated} vs naive:        CRPS {100*(1-idf/nai):+.1f}%")
    trk = summary["no-track"]["crps_after"]
    print(f"   ablation -- dropping track evolution:      CRPS {100*(trk/idf-1):+.1f}% worse"
          if trk >= idf else
          f"   ablation -- dropping track evolution:      CRPS {100*(1-trk/idf):.1f}% BETTER (track term not earning its place)")
    print()

    ftab = pl.DataFrame(ft_rows)
    print("   Race lap slope (fuel + track), forecast vs what the race did:")
    slope_mae = {}
    for col, name in (("slope_identified", "identified"),
                      ("slope_conventional", "conventional"),
                      ("slope_mean_earlier", "mean of earlier races")):
        e = (ftab[col] - ftab["slope_actual"]).to_numpy()
        slope_mae[name] = float(np.mean(np.abs(e)))
        print(f"     {name:<22s} MAE {np.mean(np.abs(e)):.4f}   bias {np.mean(e):+.4f} s/lap")
    print("   A positive bias = forecast too shallow: races sped up more than predicted.")
    print("   The identified fuel effect is learned from earlier races; if it trails a")
    print("   fixed physics value here, the race identification is biased low --")
    print("   out-of-sample evidence on the README's open fuel-pace gap.")
    print()

    artifacts.save(task, "predictive", pl.concat(pred_tabs))
    artifacts.save(task, "weekends", pl.DataFrame(per_wk))
    artifacts.save(task, "fuel_track", ftab)
    artifacts.save(task, "params", pl.DataFrame(params))
    results.record(task, {
        "gated_method": gated, "traffic_anchor": bool(args.traffic),
        **{f"{k}_{m.replace('-', '_')}": v for m, s in summary.items() for k, v in s.items()},
        "weekends": n_w,
        **{f"slope_mae_{k.split()[0]}": v for k, v in slope_mae.items()},
    }, gate="PASS" if ok else "FAIL")

    print("=" * 78)
    print(f"BACKTEST GATE: {'PASS' if ok else 'FAIL'} -- {gated} "
          f"{'beats' if ok else 'does NOT beat'} conventional and naive after pit stops")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
