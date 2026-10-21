"""B1 -- the pre-registered baselines, on the ladder's own held-out laps.

eval/protocol.md section 4: "All four run on every comparison. Beating only
the first proves nothing." Section 8 lists, as a falsification condition, the
causal structure performing no better than the ML baseline.

Every baseline gets the same Design, the same contiguous forward split (last
30% of each race held out) and the same metrics.score() as L2-L5, so the
numbers are directly comparable with results.json["l4"].

WHAT THIS TEST CAN AND CANNOT SHOW
----------------------------------
Forecasting the rest of the SAME race needs only the fused lap slope and the
pit-stop resets. A model that assumes the wrong fuel coefficient can absorb
the error into a free degradation curve and forecast nearly as well. So this
comparison tests predictive skill; it does NOT test whether the fuel /
degradation split is right. That needs a task where fuel load differs between
fit and forecast -- the FP -> race backtest (Tier 3).

Gate: PASS if the best ladder rung beats every baseline on CRPS. Coverage is
reported, not gated -- no rung is calibrated yet either.

Run:  python audit/b1_baselines.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audit.l2_tau_scale import build  # noqa: E402
from audit.l3_dirty_air import split, subset  # noqa: E402
from audit.l4_calibration import lap_labels  # noqa: E402
from tyretrace import artifacts, results  # noqa: E402
from tyretrace.eval import metrics  # noqa: E402
from tyretrace.models import baselines  # noqa: E402

RULE = "-" * 76
LADDER = ("L2", "L3", "L4")      # L5 is rejected; not a contender


def main() -> int:
    print("=" * 76)
    print("B1  BASELINES -- protocol section 4, on the ladder's held-out laps")
    print("=" * 76)

    races = pl.read_parquet(ROOT / "out" / "audit_race.parquet")
    quali = pl.read_parquet(ROOT / "out" / "audit_quali.parquet")
    d = build(races, quali)
    tr, te = split(d)
    d_tr, d_te = subset(d, tr), subset(d, te)
    print(f"   train {d_tr.n} · test {d_te.n}   (same split as L2-L5)")
    print()

    scores, pred_tabs, cal_tabs = {}, [], []
    labels = lap_labels(d_te)
    for name, fn in baselines.BASELINES.items():
        t0 = time.time()
        pp = fn(d_tr, d_te)
        scores[name] = metrics.score(pp, d_te.y)
        q05, q95 = np.quantile(pp, [0.05, 0.95], axis=0)
        scores[name]["above"] = float(np.mean(d_te.y > q95))
        scores[name]["below"] = float(np.mean(d_te.y < q05))
        rung = name.split()[0]
        pred_tabs.append(artifacts.predictive(pp, d_te.y, **labels)
                         .with_columns(pl.lit(rung).alias("rung")))
        cal_tabs.append(artifacts.calibration(pp, d_te.y)
                        .with_columns(pl.lit(rung).alias("rung")))
        print(f"   {name:<22s} fitted in {time.time()-t0:4.1f}s")
    print()

    l4 = results.get("l4").get("metrics", {})
    ladder = {r: {"crps": l4.get(f"crps_{r.lower()}"),
                  "mae": l4.get(f"mae_{r.lower()}"),
                  "coverage_90": l4.get(f"cov90_{r.lower()}")} for r in LADDER}
    ladder = {r: v for r, v in ladder.items() if v["crps"] is not None}
    if not ladder:
        print("   no ladder scores in results.json -- run `python -m tyretrace run l4`")
        return 1

    print(RULE)
    print(f"   {'model':<22s} {'CRPS':>8s} {'MAE':>8s} {'cov90':>7s} {'slow':>6s} {'fast':>6s}")
    print(RULE)
    rows = [(n, s) for n, s in scores.items()] + [(r, v) for r, v in ladder.items()]
    for n, s in sorted(rows, key=lambda x: x[1]["crps"]):
        mae = f"{s['mae']:.4f}" if s.get("mae") is not None else "    -"
        slow = f"{s['above']:.3f}" if "above" in s else "   -"
        fast = f"{s['below']:.3f}" if "below" in s else "   -"
        print(f"   {n:<22s} {s['crps']:8.4f} {mae:>8s} {s['coverage_90']:7.3f} "
              f"{slow:>6s} {fast:>6s}")
    print("   (slow / fast = share of laps beyond the 95th / below the 5th percentile;")
    print("    5% each is calibrated)")
    print()

    best = min(ladder, key=lambda r: ladder[r]["crps"])
    best_base = min(scores, key=lambda n: scores[n]["crps"])
    margin = 100 * (1 - ladder[best]["crps"] / scores[best_base]["crps"])
    beats_all = all(ladder[best]["crps"] < s["crps"] for s in scores.values())
    ml = scores["B3 gradient boosting"]["crps"]
    pub = scores["B4 published"]["crps"]
    print(f"   best ladder rung {best} vs best baseline {best_base}: "
          f"CRPS {margin:+.1f}%")
    print(f"   vs the ML baseline (protocol section 8 falsification check): "
          f"{100*(1-ladder[best]['crps']/ml):+.1f}%")
    print(f"   vs the published race-simulation model (B4): "
          f"{100*(1-ladder[best]['crps']/pub):+.1f}%")
    print()
    print("   Read with care: forecasting the rest of the same race only needs the")
    print("   FUSED slope, so a wrong fuel assumption costs the conventional model")
    print("   little here. This shows predictive skill, not that the fuel /")
    print("   degradation split is right. The FP -> race backtest tests that.")
    print()

    artifacts.save("baselines", "predictive", pl.concat(pred_tabs))
    artifacts.save("baselines", "calibration", pl.concat(cal_tabs))
    results.record("baselines", {
        **{f"crps_{n.split()[0].lower()}": s["crps"] for n, s in scores.items()},
        **{f"mae_{n.split()[0].lower()}": s["mae"] for n, s in scores.items()},
        **{f"cov90_{n.split()[0].lower()}": s["coverage_90"] for n, s in scores.items()},
        "best_rung": best, "best_baseline": best_base,
        "margin_vs_best_baseline_pct": margin,
        "published_baseline_built": True,
    }, gate="PASS" if beats_all else "FAIL")

    print("=" * 76)
    print(f"BASELINE GATE: {'PASS' if beats_all else 'FAIL'} -- {best} "
          f"{'beats' if beats_all else 'does NOT beat'} every baseline built so far "
          "(all 4)")
    print("=" * 76)
    return 0 if beats_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
