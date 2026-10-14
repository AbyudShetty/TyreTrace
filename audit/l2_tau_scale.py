"""L2 -- estimate the fuel/track split instead of asserting it.

L0 and L1 leave the lap coefficient fused. Declaring tau from qualifying was
worth a factor of two on beta_f (0.028 with no track evolution, 0.015 at the
full quali rate), so the split is far too consequential to assert.

L2 estimates it. Per circuit:

    lap_coef_c = -(beta_f * kappa_c  +  s * tau_quali_c)

with beta_f and s global and both regressors observed. Ten weekends of
variation in kappa (1.55-2.16 kg/lap) and tau_quali, only mildly collinear
(corr -0.54, VIF 1.42), identify both.

Also delivers estimand E1 -- the within-weekend compound contrast -- which
needs no Pirelli map because relative labels are unambiguous inside a weekend.

Run:  python audit/l2_tau_scale.py [--full]
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

from tyretrace import artifacts, results  # noqa: E402
from tyretrace.data import fastf1_pull as fp  # noqa: E402
from tyretrace.features import traffic  # noqa: E402
from tyretrace.models import ladder, track_evolution as te  # noqa: E402

RULE = "-" * 76
FULL_TANK_KG = 110.0


def build(races_raw: pl.DataFrame, quali: pl.DataFrame) -> ladder.Design:
    """Attach kappa and the qualifying tau to every lap."""
    # traffic first: the running order depends on every car on track,
    # including ones whose laps get trimmed by the long-run filter
    races_raw = traffic.dirty_air_index(traffic.add_gap_ahead(races_raw))
    races = fp.long_runs(races_raw, min_laps=4)

    # kappa from race length: a full tank divided by the laps it must cover
    n_laps = (races_raw.group_by("session_uid")
                       .agg(pl.col("lap_in_session").max().alias("race_laps")))
    races = races.join(n_laps, on="session_uid")

    # tau per weekend, in MODEL units, via the sanctioned bridge
    lap_by_uid = dict(races.group_by("session_uid")
                            .agg(pl.col("lap_time_s").median().alias("m"))
                            .iter_rows())
    tau_by_weekend: dict[str, float] = {}
    for f in te.fit(quali):
        wk = f.session_uid.rsplit("_", 1)[0]
        # convert at that weekend's own race lap time
        race_uid = f"{wk}_R"
        tau_by_weekend[wk] = f.tau_model(lap_by_uid.get(race_uid, 90.0))

    races = races.with_columns(
        pl.col("session_uid").str.replace(r"_R$", "").alias("weekend"))
    races = races.filter(pl.col("weekend").is_in(list(tau_by_weekend)))

    d = ladder.build_design(races)
    d.kappa = (FULL_TANK_KG / races["race_laps"].to_numpy()).astype(np.float64)
    wk = races["weekend"].to_numpy().astype(str)
    d.tau_quali = np.array([tau_by_weekend[w] for w in wk], dtype=np.float64)

    levels = sorted(set(wk.tolist()))
    d.weekend = np.array([levels.index(w) for w in wk], dtype=np.int32)
    d.n_weekend = len(levels)
    d.weekend_names = levels

    # E1 needs the within-weekend relative label, which IS unambiguous
    rel = races["compound_rel"].cast(pl.Utf8).to_numpy().astype(str)
    rlev = sorted(set(rel.tolist()))
    d.compound = np.array([rlev.index(c) for c in rel], dtype=np.int32)
    d.n_compound = len(rlev)
    d.compound_names = rlev
    d.dirty_air = races["dirty_air"].to_numpy().astype(np.float64)

    # Stint identity, for L4's per-stint offset. Indexed over the WHOLE frame
    # so a train/test split shares the code space: a stint seen only in the
    # held-out block then gets no likelihood contribution, its offset falls
    # back to the population prior, and its interval widens honestly.
    sk = np.char.add(np.char.add(races["session_uid"].to_numpy().astype(str), "|"),
                     np.char.add(races["driver"].to_numpy().astype(str),
                                 races["stint_id"].to_numpy().astype(str)))
    slev = sorted(set(sk.tolist()))
    idx = {v: i for i, v in enumerate(slev)}
    d.stint = np.array([idx[v] for v in sk], dtype=np.int32)
    d.n_stint = len(slev)
    # which circuit each stint belongs to, for the within-circuit demeaning
    sc = np.zeros(d.n_stint, dtype=np.int32)
    sc[d.stint] = d.circuit
    d.stint_circuit = sc
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    draws = 2000 if args.full else 800
    warmup = draws

    print("=" * 76)
    print(f"L2  ESTIMATING THE FUEL / TRACK SPLIT   ({draws} draws)")
    print("=" * 76)

    races_raw = pl.read_parquet(ROOT / "out" / "audit_race.parquet")
    quali = pl.read_parquet(ROOT / "out" / "audit_quali.parquet")
    d = build(races_raw, quali)

    print(f"   {d.n} laps · {d.n_weekend} weekends · {d.n_driver} drivers · "
          f"{d.n_compound} compound labels {d.compound_names}")
    print(f"   kappa      {d.kappa.min():.3f} - {d.kappa.max():.3f} kg/lap")
    print(f"   tau_quali  {d.tau_quali.min():+.5f} - {d.tau_quali.max():+.5f} s/lap"
          "  (model units: positive = improving)")
    r = float(np.corrcoef(d.kappa, d.tau_quali)[0, 1])
    print(f"   corr(kappa, tau) {r:+.4f}   VIF {1/(1-r*r):.2f}")
    print()

    t0 = time.time()
    mcmc = ladder.fit("L2", d, draws=draws, warmup=warmup, chains=2)
    diag = ladder.diagnostics(mcmc)
    post = mcmc.get_samples()
    print(f"   fitted in {time.time()-t0:.0f}s   r_hat {diag['max_rhat']:.4f}   "
          f"ESS {diag['min_ess_bulk']:.0f}   div {diag['divergences']}")
    print()

    print(RULE)
    print("THE SPLIT, ESTIMATED RATHER THAN ASSERTED")
    print(RULE)

    # Protocol section 1: report s/lap as PRIMARY. beta_f in s/kg is derived, and the
    # conversion needs kappa from outside the data -- so its error propagates
    # straight through. The audit below shows that is the binding constraint,
    # not the statistics.
    ts = np.asarray(post["tau_scale"])
    bf = np.asarray(post["beta_f"])
    kbar = float(np.mean(d.kappa))
    fuel_pace = bf * kbar                      # s/lap, the identified quantity

    lo, hi = np.quantile(fuel_pace, [0.05, 0.95])
    print(f"   fuel pace (s/lap)  {fuel_pace.mean():+.5f}   90% [{lo:+.5f}, {hi:+.5f}]"
          "   <- IDENTIFIED")
    lo, hi = np.quantile(ts, [0.05, 0.95])
    print(f"   tau scale s        {ts.mean():+.5f}   90% [{lo:+.5f}, {hi:+.5f}]"
          "   <- IDENTIFIED")
    print()
    print(f"   P(s < 1)   = {float(np.mean(ts < 1)):.2f}"
          "   race evolves LESS than qualifying")
    print(f"   P(s < 0.5) = {float(np.mean(ts < 0.5)):.2f}")
    print()

    print("   DERIVED, and only as good as kappa:")
    print(f"   at the assumed kappa {kbar:.2f} kg/lap -> beta_f "
          f"{float(fuel_pace.mean())/kbar:+.5f} s/kg")
    print()
    print("   kappa was assumed as (full tank)/(race laps), which is an UPPER")
    print("   bound -- cars do not start full everywhere and they lift-and-coast.")
    print("   An overstated kappa understates beta_f by exactly the same factor:")
    for k in (1.85, 1.60, 1.40, 1.25):
        b = float(fuel_pace.mean()) / k
        flag = "  <- in the 0.030-0.035 band" if 0.030 <= b <= 0.035 else ""
        print(f"     kappa {k:.2f} kg/lap  ->  beta_f {b:+.5f} s/kg{flag}")
    print()
    print("   So the s/kg number is NOT a statistical result, it is a kappa")
    print("   assumption wearing a posterior. Report s/lap; treat s/kg as")
    print("   derived and carry kappa's uncertainty, exactly as protocol section 1 says.")
    print()

    print(RULE)
    print("E1 -- WITHIN-WEEKEND COMPOUND CONTRASTS (no Pirelli map needed)")
    print(RULE)
    deg = np.asarray(post["deg"])          # (draws, weekend, compound)
    names = d.compound_names
    if "SOFT" in names and "HARD" in names:
        i, j = names.index("SOFT"), names.index("HARD")
        con = deg[:, :, i] - deg[:, :, j]
        print(f"   delta(SOFT) - delta(HARD), per weekend, s/lap")
        for w in range(d.n_weekend):
            v = con[:, w]
            lo, hi = np.quantile(v, [0.05, 0.95])
            sig = "*" if (lo > 0 or hi < 0) else " "
            print(f"     {d.weekend_names[w]:<14s} {v.mean():+.4f}  "
                  f"90% [{lo:+.4f}, {hi:+.4f}] {sig}")
        pooled = con.mean(axis=1)
        lo, hi = np.quantile(pooled, [0.05, 0.95])
        print(f"   (* = interval excludes zero)")
        print(f"   mean across weekends {pooled.mean():+.4f}  90% [{lo:+.4f}, {hi:+.4f}]")
        print()
        print("   NOTE this averages contrasts, it does not pool them -- SOFT is")
        print("   a different rubber each weekend. Pooling needs the C1-C5 map.")
    print()

    ok = diag["divergences"] == 0 and diag["max_rhat"] < 1.05

    # for the dashboard: the split's posterior, and every weekend x compound
    # degradation rate with the within-weekend SOFT-HARD contrast (E1)
    artifacts.save("l2", "posterior", artifacts.posterior(
        post, ("beta_f", "tau_scale", "sigma"), derived={"fuel_pace": fuel_pace}))
    rows = []
    for w in range(d.n_weekend):
        for c, name in enumerate(names):
            v = deg[:, w, c]
            lo, hi = np.quantile(v, [0.05, 0.95])
            rows.append({"weekend": d.weekend_names[w], "term": name,
                         "mean": float(v.mean()), "q05": float(lo), "q95": float(hi)})
        if "SOFT" in names and "HARD" in names:
            v = con[:, w]
            lo, hi = np.quantile(v, [0.05, 0.95])
            rows.append({"weekend": d.weekend_names[w], "term": "SOFT-HARD",
                         "mean": float(v.mean()), "q05": float(lo), "q95": float(hi)})
    artifacts.save("l2", "degradation", pl.DataFrame(rows))

    lo_fp, hi_fp = np.quantile(fuel_pace, [0.05, 0.95])
    lo_s, hi_s = np.quantile(ts, [0.05, 0.95])
    results.record("l2", {
        "fuel_pace": float(fuel_pace.mean()),
        "fuel_pace_q05": float(lo_fp), "fuel_pace_q95": float(hi_fp),
        "tau_scale": float(ts.mean()),
        "tau_scale_q05": float(lo_s), "tau_scale_q95": float(hi_s),
        "p_tau_scale_lt_1": float(np.mean(ts < 1)),
        "kappa_mean": kbar,
        "max_rhat": diag["max_rhat"], "divergences": diag["divergences"],
    })

    print("=" * 76)
    print(f"L2 GATE: {'PASS' if ok else 'FAIL'}")
    print("=" * 76)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
