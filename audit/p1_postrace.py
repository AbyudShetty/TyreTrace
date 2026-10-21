"""P1 -- post-race validation: which assumption made each forecast wrong?

For every backtest weekend and method, each target lap's forecast error is
split exactly into four parts (tyretrace/backtest/postrace.py):

    slope    fuel + race track evolution, per lap since the anchor
    tyres    degradation per compound
    traffic  dirty air the forecast did not model
    noise    lap-to-lap scatter, including the anchor laps' own noise

Reported as mean signed error (bias) and root-mean-square per part, pooled
over every weekend, and per weekend. A method whose slope part dominates got
fuel or track wrong; one whose tyres part dominates got degradation wrong.

No gate: this is a diagnostic, the protocol's "post-race validation tool".

Run:  python audit/p1_postrace.py [--data audit|lake]
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
from tyretrace.backtest import postrace as P  # noqa: E402
from tyretrace.data import sources  # noqa: E402

RULE = "-" * 78
METHODS = ("identified", "fuel-only", "conventional")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=("audit", "lake"), default="audit")
    args = ap.parse_args()
    bt = "backtest" if args.data == "audit" else "backtest_lake"
    params = artifacts.load(bt, "params")
    if params is None:
        print(f"no backtest parameters -- run audit/t3_backtest.py --data {args.data} first")
        return 1

    print("=" * 78)
    print(f"P1  POST-RACE VALIDATION -- error by cause   (data: {args.data})")
    print("=" * 78)

    race_raw = sources.load(args.data)["race"].with_columns(
        pl.col("session_uid").str.slice(0, 7).alias("weekend"))
    tabs = []
    for wk in sorted(params["weekend"].unique().to_list()):
        rr = P.prepare_race(race_raw.filter(pl.col("weekend") == wk))
        for m in METHODS:
            p = params.filter((pl.col("weekend") == wk) & (pl.col("method") == m))
            if p.height == 0:
                continue
            deg = dict(zip(p["compound"].to_list(), p["deg"].to_list()))
            tab = P.decompose(rr, float(p["lap_coef"][0]), deg)
            tabs.append(tab.with_columns(pl.lit(wk).alias("weekend"), pl.lit(m).alias("method")))
    allt = pl.concat(tabs)

    print(f"   {allt['weekend'].n_unique()} weekends · {allt.filter(pl.col('method') == METHODS[0]).height:,} "
          "laps per method · parts sum exactly to the error (checked per lap)")
    print()
    print(RULE)
    print(f"   {'method':<13s} {'':>6s} " + "".join(f"{p:>9s}" for p in P.PARTS) + f"{'total':>9s}")
    print(RULE)
    summary = {}
    for m in METHODS:
        t = allt.filter(pl.col("method") == m)
        if t.height == 0:
            continue
        bias = {p: float(t[p].mean()) for p in P.PARTS}
        rms = {p: float(np.sqrt((t[p] ** 2).mean())) for p in P.PARTS}
        summary[m] = {"bias": bias, "rms": rms,
                      "total_bias": float(t["error"].mean()),
                      "total_rms": float(np.sqrt((t["error"] ** 2).mean()))}
        print(f"   {m:<13s} {'bias':>6s} " + "".join(f"{bias[p]:+9.3f}" for p in P.PARTS)
              + f"{summary[m]['total_bias']:+9.3f}")
        print(f"   {'':<13s} {'rms':>6s} " + "".join(f"{rms[p]:9.3f}" for p in P.PARTS)
              + f"{summary[m]['total_rms']:9.3f}")
    print("   (seconds; positive bias = forecast too slow)")
    print()

    # which assumption dominated, per method -- the sentence a strategist wants
    print(RULE)
    for m, s in summary.items():
        sys_parts = {p: abs(s["bias"][p]) for p in ("slope", "tyres", "traffic")}
        top = max(sys_parts, key=sys_parts.get)
        print(f"   {m:<13s} largest systematic error: {top:<8s} "
              f"({s['bias'][top]:+.3f} s mean)   lap noise rms {s['rms']['noise']:.3f} s")
    print()

    per_wk = (allt.group_by("weekend", "method")
                  .agg(*[pl.col(p).mean().alias(f"{p}_bias") for p in P.PARTS],
                       pl.col("error").mean().alias("error_bias"),
                       (pl.col("error") ** 2).mean().sqrt().alias("error_rms"))
                  .sort("weekend", "method"))
    task = "postrace" if args.data == "audit" else "postrace_lake"
    artifacts.save(task, "laps", allt)
    artifacts.save(task, "weekends", per_wk)
    results.record(task, {f"{m.replace('-', '_')}_{k}_{p}": v
                          for m, s in summary.items()
                          for k in ("bias", "rms") for p, v in s[k].items()})
    print("P1: diagnostic, no gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
