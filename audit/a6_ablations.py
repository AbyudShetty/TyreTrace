"""A6 -- the ablation table, protocol section 6.

"Every component is removed individually and the full metric set
re-reported. A component that cannot show its contribution does not ship."

Two families, each measured where it actually operates:

  FORECAST PIPELINE  (rolling FP -> race backtest, laps after a pit stop)
      track evolution     race and practice track terms dropped
      degradation prior   practice alone, missing compounds borrow others (v1)
      random effects      practice runs pooled as if they all agreed (v1)
      lap filtering       the slow-lap trim switched off

  MODEL LADDER  (held-out end of each race; already-gated rung comparisons)
      hierarchy           L1 vs L0, synthetic twin
      dirty air           L3 vs L2
      stint offsets + heavy tails   L4 vs L3

  ADDITIONS            measured the other way round -- ADDED to the reference:
      traffic-aware anchor   anchor laps de-trafficked, expected traffic added
      recalibration          rolling empirical recalibration of the intervals

  SEPARATE CANDIDATE   regime switching is evaluated in rolling_models; it is
                       not part of this original reference pipeline.
  NOT BUILT            telemetry channel and censoring model.

A component EARNS its place if removing it makes CRPS worse. The table is the
deliverable; there is no single gate, but every component that fails to earn
its place is named.

Run:  python audit/a6_ablations.py [--data audit|lake]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tyretrace import artifacts, results  # noqa: E402
from tyretrace.backtest import fp_race as B  # noqa: E402
from tyretrace.data import sources  # noqa: E402

RULE = "-" * 78
FOCUS = "identified"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=("audit", "lake"), default="audit")
    args = ap.parse_args()
    d = sources.load(args.data)

    print("=" * 78)
    print(f"A6  ABLATIONS -- does every component earn its place?   (data: {args.data})")
    print("=" * 78)

    full = B.rolling(d["practice"], d["quali"], d["race"])
    base = full[FOCUS]
    variants = {
        "degradation prior": B.rolling(d["practice"], d["quali"], d["race"], prior=False),
        "random effects": B.rolling(d["practice"], d["quali"], d["race"], between=False),
        "lap filtering": B.rolling(d["practice"], d["quali"], d["race"], trim=False),
    }
    rows = [{"family": "forecast", "component": "(full pipeline)", "crps": base["crps_after"],
             "change_pct": 0.0, "cov90": base["cov90"], "status": "reference"}]

    def add(family, comp, crps, cov, ref):
        change = 100 * (crps / ref - 1)
        rows.append({"family": family, "component": comp, "crps": crps,
                     "change_pct": change, "cov90": cov,
                     "status": "earns its place" if change > 0.5 else
                               ("no measurable effect" if change > -0.5 else "HURTS: removing it helps")})

    # track evolution lives in the method spec: "no-track" is identified minus it
    nt = full["no-track"] if "no-track" in full else B.rolling(
        d["practice"], d["quali"], d["race"], methods=("no-track",))["no-track"]
    add("forecast", "track evolution", nt["crps_after"], nt["cov90"], base["crps_after"])
    for comp, r in variants.items():
        add("forecast", comp, r[FOCUS]["crps_after"], r[FOCUS]["cov90"], base["crps_after"])
    # lap filtering changes the scored laps too, so flag it
    lap_filter_note = (f"   note: without the trim {variants['lap filtering'][FOCUS]['n_after']:,} laps "
                       f"are scored instead of {base['n_after']:,}; part of its change is a different test set")

    lad = results.get("l4").get("metrics", {})
    tw = results.get("ladder").get("metrics", {})
    for comp, (a, b), src, fam in (
            ("hierarchy", ("twin_crps_l0", "twin_crps_l1"), tw, "ladder (twin)"),
            ("dirty air", ("crps_l2", "crps_l3"), lad, "ladder"),
            ("stint offsets + heavy tails", ("crps_l3", "crps_l4"), lad, "ladder")):
        if src.get(a) is not None and src.get(b) is not None:
            # removing the component takes you from the richer rung (b) to the poorer (a)
            add(fam, comp, src[a], None, src[b])
    adds = {
        "traffic-aware anchor": B.rolling(d["practice"], d["quali"], d["race"], traffic=True),
        "recalibration": B.rolling(d["practice"], d["quali"], d["race"], recal=True),
    }
    for comp, r in adds.items():
        a = r[FOCUS]
        change = 100 * (a["crps_after"] / base["crps_after"] - 1)
        rows.append({"family": "addition", "component": comp, "crps": a["crps_after"],
                     "change_pct": change, "cov90": a["cov90"],
                     "status": ("helps: adopt" if change < -0.5 else
                                "no measurable effect" if change < 0.5 else
                                "hurts CRPS: do not adopt")})
    rows.append({"family": "separate candidate", "component": "regime switching", "crps": None,
                 "change_pct": None, "cov90": None,
                 "status": "built; evaluated in rolling_models, outside this reference"})
    for comp in ("telemetry channel", "censoring model"):
        rows.append({"family": "not built", "component": comp, "crps": None,
                     "change_pct": None, "cov90": None, "status": "not built"})

    print(f"   reference: {FOCUS}, CRPS after a pit stop {base['crps_after']:.3f} "
          f"over {base['weekends']} weekends, {base['n_after']:,} laps")
    print()
    print(RULE)
    print(f"   {'family':<14s} {'remove...':<30s} {'CRPS':>7s} {'change':>9s}   verdict")
    print(RULE)
    for r in rows[1:]:
        crps = f"{r['crps']:.3f}" if r["crps"] is not None else "   -"
        ch = f"{r['change_pct']:+.1f}%" if r["change_pct"] is not None else "    -"
        print(f"   {r['family']:<14s} {r['component']:<30s} {crps:>7s} {ch:>9s}   {r['status']}")
    print("   change = CRPS without the component vs with it; positive = it was helping")
    print("   (additions: change = CRPS WITH the addition vs the reference; negative = it helps)")
    print(lap_filter_note)
    print()

    t = pl.DataFrame(rows)
    artifacts.save("ablations" if args.data == "audit" else "ablations_lake", "table", t)
    weak = [r["component"] for r in rows[1:]
            if r["family"] != "addition" and r["status"].startswith(("no measurable", "HURTS"))]
    results.record("ablations" if args.data == "audit" else "ablations_lake", {
        **{f"change_{r['component'].replace(' ', '_').replace('+', 'and')}": r["change_pct"]
           for r in rows[1:] if r["change_pct"] is not None},
        "not_earning": weak,
    })
    print(f"A6: {len(weak)} component(s) not earning their place"
          + (f": {', '.join(weak)}" if weak else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
