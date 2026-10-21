"""D1 -- do the two data sources agree?

The ten weekends in out/audit_*.parquet came from FastF1 before F1's
live-timing archive went behind a login; everything since comes from OpenF1
(tyretrace/data/openf1.py). Before a single new lap is trusted, every session
both sources have is compared lap by lap:

    lap time, tyre age, stint, compound   must agree almost exactly
    track status (green or not)           rebuilt from race control, so it
                                          is allowed to differ more
    long-run filter                       the laps the models actually see:
                                          do both sources keep the same ones?

TYRE AGE ON USED SETS IS DEFINED DIFFERENTLY
--------------------------------------------
Measured on the first pass: every tyre-age disagreement was on a USED set,
a constant offset per stint (FastF1 counts laps the set ran in earlier
sessions; OpenF1's age-at-stint-start counts fewer -- typically the top ten
starting the race on their qualifying tyres, 2-3 laps). Fresh sets agree, and
within a stint both count up one per lap, so degradation SLOPES are
unaffected; only a used stint's starting age differs. The gate therefore
checks fresh-set ages and within-stint increments, and the used-set offset
is reported as a known property of the lake rather than hidden.

Gate: PASS if, over matched laps, compound, fresh-set tyre age and
within-stint age increments agree on >= 99%, lap time agrees to the
millisecond on >= 95%, and the long-run filtered sets overlap >= 90%.

Run:  python audit/d1_source_parity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tyretrace import artifacts, results  # noqa: E402
from tyretrace.data import fastf1_pull as fp  # noqa: E402
from tyretrace.data import sources  # noqa: E402

RULE = "-" * 78
KEY = ["session_uid", "driver", "lap_in_session"]


def _fresh_agree(j: pl.DataFrame) -> float:
    f = j.filter(pl.col("is_fresh"))
    return float((f["tyre_age"] == f["tyre_age_o"]).mean()) if f.height else 1.0


def _step_agree(j: pl.DataFrame) -> float:
    """Within a stint, does age advance by the same amount between laps?"""
    k = j.sort("driver", "stint_id", "lap_in_session").with_columns(
        pl.col("tyre_age").diff().over("driver", "stint_id").alias("_d"),
        pl.col("tyre_age_o").diff().over("driver", "stint_id").alias("_do"))
    k = k.filter(pl.col("_d").is_not_null() & pl.col("_do").is_not_null())
    return float((k["_d"] == k["_do"]).mean()) if k.height else 1.0


def _used_offset(j: pl.DataFrame):
    u = j.filter(~pl.col("is_fresh") & (pl.col("tyre_age") != pl.col("tyre_age_o")))
    return float((u["tyre_age_o"] - u["tyre_age"]).median()) if u.height else None


def main() -> int:
    print("=" * 78)
    print("D1  SOURCE PARITY -- FastF1 (original 10 weekends) vs OpenF1 (the lake)")
    print("=" * 78)
    a, b = sources.load("audit"), sources.load("lake")
    rows = []
    for kind in ("race", "quali", "practice"):
        fa, fb = a[kind], b[kind]
        uids = sorted(set(fa["session_uid"].unique().to_list())
                      & set(fb["session_uid"].unique().to_list()))
        for uid in uids:
            x = fa.filter(pl.col("session_uid") == uid)
            y = fb.filter(pl.col("session_uid") == uid)
            j = x.join(y, on=KEY, suffix="_o")
            if j.height == 0:
                continue
            lx = fp.long_runs(x, min_laps=4).select(KEY)
            ly = fp.long_runs(y, min_laps=4).select(KEY)
            inter = lx.join(ly, on=KEY).height
            union = lx.height + ly.height - inter
            rows.append({
                "session_uid": uid, "kind": kind,
                "laps_fastf1": x.height, "laps_openf1": y.height, "matched": j.height,
                "lap_time_exact": float(((j["lap_time_s"] - j["lap_time_s_o"]).abs() < 1e-3).mean()),
                "lap_time_max_diff": float((j["lap_time_s"] - j["lap_time_s_o"]).abs().max()),
                "tyre_age_agree": float((j["tyre_age"] == j["tyre_age_o"]).mean()),
                "fresh_age_agree": _fresh_agree(j),
                "age_step_agree": _step_agree(j),
                "used_offset_median": _used_offset(j),
                "stint_agree": float((j["stint_id"] == j["stint_id_o"]).mean()),
                "compound_agree": float((j["compound_rel"].cast(pl.Utf8)
                                         == j["compound_rel_o"].cast(pl.Utf8)).mean()),
                "green_agree": float(((j["track_status"] == "1")
                                      == (j["track_status_o"] == "1")).mean()),
                "long_run_jaccard": inter / union if union else 1.0,
            })
    t = pl.DataFrame(rows)
    if t.height == 0:
        print("   no overlapping sessions yet -- run python -m tyretrace.data.bulk")
        return 1

    w = t["matched"].to_numpy()

    def wmean(col):
        return float((t[col].to_numpy() * w).sum() / w.sum())

    agg = {c: wmean(c) for c in ("lap_time_exact", "tyre_age_agree", "fresh_age_agree",
                                 "age_step_agree", "stint_agree", "compound_agree",
                                 "green_agree", "long_run_jaccard")}
    agg["used_offset_median"] = float(t["used_offset_median"].drop_nulls().median() or 0.0)
    print(f"   {t.height} sessions in both sources · {int(w.sum()):,} matched laps")
    print(f"   (FastF1 {t['laps_fastf1'].sum():,} laps, OpenF1 {t['laps_openf1'].sum():,})")
    print()
    print(RULE)
    for kind in ("race", "quali", "practice"):
        s = t.filter(pl.col("kind") == kind)
        if s.height:
            print(f"   {kind:<9s} {s.height:>3} sessions   lap time exact {s['lap_time_exact'].mean():.3f}   "
                  f"tyre age {s['tyre_age_agree'].mean():.3f}   compound {s['compound_agree'].mean():.3f}   "
                  f"long-run overlap {s['long_run_jaccard'].mean():.3f}")
    print(RULE)
    for k, v in agg.items():
        print(f"   {k:<18s} {v:.4f}")
    worst = t.sort("long_run_jaccard").head(3)
    print("   lowest long-run overlap:", ", ".join(f"{r['session_uid']} {r['long_run_jaccard']:.2f}"
                                                for r in worst.iter_rows(named=True)))
    print()

    ok = (agg["compound_agree"] >= 0.99 and agg["fresh_age_agree"] >= 0.99
          and agg["age_step_agree"] >= 0.99
          and agg["lap_time_exact"] >= 0.95 and agg["long_run_jaccard"] >= 0.90)
    print(f"   used sets: OpenF1 age is {agg['used_offset_median']:+.0f} laps vs FastF1 (median); "
          "fresh sets and per-lap increments are what the gate checks")
    artifacts.save("parity", "sessions", t)
    results.record("parity", {**agg, "sessions": t.height, "matched_laps": int(w.sum())},
                   gate="PASS" if ok else "FAIL")
    print("=" * 78)
    print(f"D1 GATE: {'PASS' if ok else 'FAIL'} -- the lake {'can' if ok else 'can NOT'} stand "
          "alongside the original data")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
