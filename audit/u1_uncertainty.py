"""Frozen rolling uncertainty audit; keeps the original backtest untouched.

    python audit/u1_uncertainty.py --phase development
    python audit/u1_uncertainty.py --phase confirmation

Confirmation evaluates the algorithm frozen in docs/UNCERTAINTY.md. The target
outcome is used only for scoring, never passed to prediction or fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl
from scipy.special import ndtri

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tyretrace import artifacts, results  # noqa: E402
from tyretrace.infer import uncertainty as U  # noqa: E402
from tyretrace.eval.metrics import passes_ship_gate  # noqa: E402

SOURCE = ROOT / "out/artifacts/backtest_lake/predictive.parquet"
KEYS = ["weekend", "driver", "stint", "lap_in_session"]


def fuel_sensitivity_source(tank_kg: float) -> tuple[pl.DataFrame, dict]:
    """Reforecast under the alternative fuel assumption; original outputs survive."""
    from tyretrace.backtest import fp_race as B
    from tyretrace.data import sources
    from tyretrace.eval import metrics

    d = sources.load('lake')
    prac, race = B.prepare(d['practice'], d['race'])
    weekends = B.weekend_inputs(d['quali'], d['race'], tank={2026: tank_kg})
    dry = B.dry_weekends(d['race'])
    keys = sorted(k for k in weekends if k in dry)
    rc = {k: B.race_lap_coef(race.filter(pl.col('weekend') == k)) for k in keys}
    rd = {k: B.race_degradation(race.filter(pl.col('weekend') == k)) for k in keys}
    tf = B.traffic_history(race, keys)
    tables = []
    for i, k in enumerate(keys):
        if i < B.MIN_EARLIER:
            continue
        earlier = keys[:i]
        ft = B.fit_fuel_track([rc[e][0] for e in earlier], [rc[e][1] for e in earlier],
                              [weekends[e] for e in earlier])
        fc = B.run_weekend(weekends[k], prac.filter(pl.col('weekend') == k),
            race.filter(pl.col('weekend') == k), ft,
            float(np.median([rc[e][2] for e in earlier])), seed=i,
            prior=B.degradation_prior([rd[e] for e in earlier]),
            traffic=B.traffic_prior([tf[e] for e in earlier]), methods=('identified',))['identified']
        table = U.from_forecast(fc, weekend=k, method='identified')
        table = table.with_columns(pl.Series('crps', metrics.crps_ensemble(fc.draws, table['y'].to_numpy())))
        tables.append(table)
        if i % 10 == 0:
            print(f'  fuel sensitivity {tank_kg:g} kg: {k}', flush=True)
    hashes = {name: hashlib.sha256(p.read_bytes()).hexdigest()
              for name, p in sources.PATHS['lake'].items()}
    return pl.concat(tables), hashes


def original_score(t):
    y, mid = (t[c].to_numpy() for c in ("y", "q50"))
    sd = (t["q95"].to_numpy() - t["q05"].to_numpy()) / U.Z90
    return {"crps": t["crps"].mean(), "mae": float(np.abs(y-mid).mean()),
            "coverage_90": float(np.mean((y >= t["q05"].to_numpy()) & (y <= t["q95"].to_numpy()))),
            "width90": float((U.Z90*sd).mean()), "n": t.height,
            "gaussian_crps": float(U.normal_crps(mid, sd, y).mean())}


def compare(raw, repaired):
    # Join on lap identity, never rely on group/join ordering or row count alone.
    if raw.select(KEYS).n_unique() != raw.height or repaired.select(KEYS).n_unique() != repaired.height:
        raise ValueError("prediction identities must be unique")
    matched = raw.join(repaired.select(KEYS), on=KEYS, how="inner").sort(KEYS)
    new = repaired.sort(KEYS)
    if matched.height != new.height or not np.array_equal(matched["y"].to_numpy(), new["y"].to_numpy()):
        raise ValueError("original and calibrated predictions are not the same observations")
    summary, rows = {}, []
    for label, mask in (("all", np.ones(new.height, bool)),
                        ("after_stop", new["after_stop"].to_numpy()),
                        ("before_stop", ~new["after_stop"].to_numpy())):
        a, b = matched.filter(mask), new.filter(mask)
        if not a.height:
            continue
        original, corrected = original_score(a), U.score(b)
        ok, why = passes_ship_gate(corrected, original)
        # Do not count a finite-ensemble scoring artefact as an improvement.
        ok = ok and corrected["crps"] < original["gaussian_crps"]
        summary[label] = {"original": original, "calibrated": corrected,
                          "passes": bool(ok), "reason": why}
    paired = []
    for k in sorted(new["weekend"].unique().to_list()):
        for label in ("all", "after_stop"):
            a = matched.filter(pl.col("weekend") == k)
            b = new.filter(pl.col("weekend") == k)
            if label == "after_stop":
                a, b = a.filter(pl.col("after_stop")), b.filter(pl.col("after_stop"))
            if not a.height:
                continue
            sa, sb = original_score(a), U.score(b)
            delta = sb["crps"] - sa["crps"]
            rows.append({"weekend": k, "subset": label, "raw_crps": sa["crps"],
                         "crps": sb["crps"], "delta": delta,
                         "raw_cov90": sa["coverage_90"], "cov90": sb["coverage_90"],
                         "raw_width90": sa["width90"], "width90": sb["width90"], "n": a.height})
            if label == "all":
                paired.append(delta)
    # Independent resampling unit is the weekend, not its correlated laps.
    rng = np.random.default_rng(20261026)
    paired = np.asarray(paired)
    boot = rng.choice(paired, (10000, len(paired)), replace=True).mean(axis=1)
    summary["paired_weekends"] = {"n": len(paired), "mean_delta": float(paired.mean()),
        "ci95": np.quantile(boot, [0.025, 0.975]).tolist(), "fraction_improved": float((paired < 0).mean())}
    return summary, pl.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("development", "confirmation", "all"), default="all")
    ap.add_argument("--family", choices=("gaussian", "student_t"), default="student_t")
    ap.add_argument('--fuel-sensitivity', action='store_true',
                    help='also regenerate original forecasts under 110 kg in 2026')
    args = ap.parse_args(argv)
    # Push the year restriction down to the scan in development mode.
    scan = pl.scan_parquet(SOURCE).filter(pl.col("method") == "identified")
    if args.phase == "development":
        scan = scan.filter(pl.col("weekend") < "2026")
    raw = scan.collect()
    phase_results, week_tables, preds, fits = {}, [], [], []
    phases = ("development", "confirmation") if args.phase == "all" else (args.phase,)
    for phase in phases:
        start, end = ("2025", "2026") if phase == "development" else ("2026", None)
        calibrated, model_fits = U.rolling(raw, score_from=start, score_until=end, family=args.family)
        summary, per_weekend = compare(raw, calibrated)
        phase_results[phase] = summary
        week_tables.append(per_weekend.with_columns(pl.lit(phase).alias("phase")))
        preds.append(calibrated.with_columns(pl.lit(phase).alias("phase")))
        fits.append(model_fits.with_columns(pl.lit(phase).alias("phase")))
        print(phase.upper(), json.dumps(summary, indent=2), flush=True)
    task = "uncertainty_t" if args.family == "student_t" else "uncertainty"
    if "confirmation" not in phase_results:
        task += "_development"
    artifacts.save(task, "predictive", pl.concat(preds))
    artifacts.save(task, "weekends", pl.concat(week_tables))
    artifacts.save(task, "fits", pl.concat(fits))
    confirmation = phase_results.get("confirmation")
    ships = bool(confirmation and confirmation["all"]["passes"] and confirmation["after_stop"]["passes"])
    sensitivity = None
    if args.fuel_sensitivity and confirmation:
        alternative, hashes = fuel_sensitivity_source(110.0)
        repaired, alternative_fits = U.rolling(alternative, score_from='2026', family=args.family)
        sensitivity, alternative_weekends = compare(alternative, repaired)
        sensitivity['input_sha256'] = hashes
        artifacts.save(task, 'raw_110', alternative)
        artifacts.save(task, 'predictive_110', repaired)
        artifacts.save(task, 'fits_110', alternative_fits)
        artifacts.save(task, 'weekends_110', alternative_weekends)
        ships = ships and sensitivity['all']['passes'] and sensitivity['after_stop']['passes']
        print('110 KG SENSITIVITY', json.dumps(sensitivity, indent=2), flush=True)
    results.record(task, {"phases": phase_results, "method": "identified",
                         "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                         "algorithm_sha256": hashlib.sha256((ROOT / "tyretrace/infer/uncertainty.py").read_bytes()).hexdigest(),
                         "min_history_weekends": U.MIN_WEEKENDS, "family": args.family,
                         "fuel_sensitivity_110": sensitivity,
                         "scope": "marginal predictive lap times; not latent tyre/fuel intervals"},
                   gate="SHIP" if ships else "HOLD")
    print(f"UNCERTAINTY: {'SHIP' if ships else 'HOLD'}", flush=True)
    return 0 if ships or args.phase == "development" else 1


if __name__ == "__main__":
    raise SystemExit(main())
