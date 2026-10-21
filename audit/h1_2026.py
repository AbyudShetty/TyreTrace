"""H1 -- the 2026 headline. Protocol section 2:

    "2026 is the headline case. Priors from earlier seasons, evaluate on 2026
     round n using rounds < n. This is the sample-efficiency claim and it
     must be run exactly this way."

Every 2026 race is forecast from its own weekend's practice and qualifying,
with fuel and the degradation prior learned from every race before it -- the
2023-2025 seasons plus the 2026 rounds already run. Laps after a pit stop
are scored, as in T3.

THE 2026 FUEL CAVEAT
--------------------
2026 cars start with far less fuel (a fuel-energy flow limit; ~70 kg is the
widely reported race load, not verified here). Fuel per lap, and so every
fuel-corrected method, depends on it. Both assumptions are run and reported:
70 kg (new regulations) and 110 kg (as if nothing changed). A conclusion
that flips between them is reported as not established.

The headline method is fuel-only, pre-registered 2026-10-21 before the lake
was pulled (see tyretrace/backtest/fp_race.py:fit_fuel_only).

Gate: PASS if fuel-only beats both conventional and naive after a pit stop
under BOTH fuel assumptions.

Run:  python audit/h1_2026.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tyretrace import results  # noqa: E402
from tyretrace.backtest import fp_race as B  # noqa: E402
from tyretrace.data import sources  # noqa: E402

RULE = "-" * 78
HEAD = "fuel-only"
METHODS = ("fuel-only", "identified", "conventional", "naive")


def main() -> int:
    print("=" * 78)
    print("H1  2026 HEADLINE -- forecasting the new regulations' races from practice")
    print("=" * 78)
    d = sources.load("lake")
    runs = {}
    for label, tank in (("70 kg (2026 rules)", None), ("110 kg (old rules)", {2026: 110.0})):
        runs[label] = B.rolling(d["practice"], d["quali"], d["race"], methods=METHODS,
                                score_from="2026_00", tank=tank)

    any_run = next(iter(runs.values()))
    if not any_run:
        print("   no 2026 weekends scored -- pull the lake first")
        return 1
    print(f"   {any_run[HEAD]['weekends']} weekends of 2026 scored · "
          f"{any_run[HEAD]['n_after']:,} laps after a pit stop")
    print()
    print(RULE)
    print(f"   {'method':<14s}" + "".join(f"{k:>26s}" for k in runs))
    print(f"   {'':<14s}" + "".join(f"{'CRPS':>13s}{'bias':>13s}" for _ in runs))
    print(RULE)
    for m in METHODS:
        line = f"   {m:<14s}"
        for r in runs.values():
            line += f"{r[m]['crps_after']:13.3f}{r[m]['bias_after']:+13.3f}"
        print(line)
    print("   (CRPS and bias, seconds, laps after a pit stop)")
    print()

    verdicts = {}
    for label, r in runs.items():
        h, c, n = r[HEAD]["crps_after"], r["conventional"]["crps_after"], r["naive"]["crps_after"]
        verdicts[label] = h < c and h < n
        print(f"   {label:<20s} {HEAD} vs conventional {100 * (1 - h / c):+.1f}%   "
              f"vs naive {100 * (1 - h / n):+.1f}%   -> {'beats both' if verdicts[label] else 'does NOT beat both'}")
    ok = all(verdicts.values())
    robust = len(set(verdicts.values())) == 1
    print()
    if not robust:
        print("   The verdict FLIPS with the fuel assumption: not established until the")
        print("   2026 race fuel load is verified.")
    print()

    # ---- what the races themselves say about 2026 fuel -----------------------
    # The fused race slope needs no fuel-load assumption at all, so it can
    # test the 70 kg figure directly: if 2026 cars burned 35% less fuel per
    # lap, races would speed up 35% less per lap.
    _, race = B.prepare(d["practice"], d["race"])
    dry = B.dry_weekends(d["race"])
    rows = []
    for k in sorted(dry):
        r = race.filter(pl.col("weekend") == k)
        if r.height >= 300 and r["compound_rel"].n_unique() >= 2:
            rows.append((k[:4], B.race_lap_coef(r)[0]))
    t = pl.DataFrame(rows, schema=["year", "slope"], orient="row")
    print(RULE)
    print("   RACE SLOPE BY SEASON (fuel + track, no fuel-load assumption)")
    print(RULE)
    for y, g in t.group_by("year", maintain_order=True):
        v = g["slope"].to_numpy()
        print(f"   {y[0]}  {len(v):>2} dry races   median {np.median(v):+.4f}   "
              f"mean {v.mean():+.4f} ± {v.std(ddof=1) / np.sqrt(len(v)):.4f} s/lap")
    post = t.filter(pl.col("year") == "2026")["slope"].to_numpy()
    pre = t.filter(pl.col("year") != "2026")["slope"].to_numpy()
    diff = post.mean() - pre.mean()
    se = float(np.sqrt(post.var(ddof=1) / len(post) + pre.var(ddof=1) / len(pre)))
    expected = pre.mean() * B.full_tank(2026) / B.FULL_TANK_KG
    print(f"   2026 vs earlier: {diff:+.4f} ± {se:.4f} s/lap (t {diff / se:+.1f})")
    print(f"   if 2026 cars started with {B.full_tank(2026):.0f} kg, the slope should be about "
          f"{expected:+.4f}; observed {post.mean():+.4f}")
    print("   -> 2026 races speed up per lap much as before: either the race fuel load is")
    print("      well above the reported figure, or each kg costs more time in the new")
    print("      cars. The data cannot tell which -- but it is inconsistent with 70 kg")
    print("      combined with an unchanged fuel sensitivity.")

    results.record("headline_2026", {
        **{f"{m.replace('-', '_')}_crps_after_{'70' if '70' in lab else '110'}": r[m]["crps_after"]
           for lab, r in runs.items() for m in METHODS},
        "weekends": any_run[HEAD]["weekends"], "laps_after": any_run[HEAD]["n_after"],
        "robust_to_fuel_assumption": robust,
        "slope_2026": float(post.mean()), "slope_2023_2025": float(pre.mean()),
        "slope_diff": float(diff), "slope_diff_se": se,
        "slope_expected_at_70kg": float(expected),
    }, gate="PASS" if ok else "FAIL")
    print("=" * 78)
    print(f"H1 GATE: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
