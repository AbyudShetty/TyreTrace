"""A1 + A2 -- does the identification argument survive contact with real data?

A1  PRACTICE DECORRELATION
    Gate: |corr(lap_in_run, session_clock)| < 0.3, condition number < 30.
    Plus the number A3 showed actually binds: the spread of run START times.
    Track evolution is identified from BETWEEN-run variation under partial
    pooling, so what matters is how widely runs are staggered, not just the
    pooled correlation.

A2  RACE DESIGN CONDITIONING
    Gate: implied posterior SD on per-compound degradation small relative to
    the ~0.008 s/lap spread the 2026 regulations leave between compounds.
    Turns "the pit stop identifies it" into a number.

Run:  python audit/a1_a2_identification.py           (uses cached parquet)
      python audit/a1_a2_identification.py --pull    (re-pull from FastF1)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tyretrace.data import fastf1_pull as fp  # noqa: E402

OUT = ROOT / "out"
PRAC_PQ = OUT / "audit_practice.parquet"
RACE_PQ = OUT / "audit_race.parquet"

# Ten weekends spread across eras: pre-2022 aero, ground effect, and the
# current regulations. Deliberately mixed circuits -- high deg and low deg.
WEEKENDS = [
    (2023, 4), (2023, 9), (2023, 14), (2023, 18),
    (2024, 1), (2024, 6), (2024, 11), (2024, 16),
    (2025, 3), (2025, 10),
]
PRACTICE_SESSIONS = ("FP1", "FP2", "FP3")


# --------------------------------------------------------------------------

def pull() -> tuple[pl.DataFrame, pl.DataFrame]:
    OUT.mkdir(exist_ok=True)
    prac, race = [], []
    for year, rnd in WEEKENDS:
        print(f"  {year} round {rnd}")
        for s in PRACTICE_SESSIONS:
            d = fp.load_session(year, rnd, s)
            if d.height:
                prac.append(d)
                print(f"    {s}: {d.height} laps")
        d = fp.load_session(year, rnd, "R")
        if d.height:
            race.append(d)
            print(f"    R:  {d.height} laps")

    pdf = pl.concat(prac) if prac else fp.schema.empty()
    rdf = pl.concat(race) if race else fp.schema.empty()
    pdf.write_parquet(PRAC_PQ)
    rdf.write_parquet(RACE_PQ)
    return pdf, rdf


def load() -> tuple[pl.DataFrame, pl.DataFrame]:
    if not PRAC_PQ.exists():
        return pull()
    return pl.read_parquet(PRAC_PQ), pl.read_parquet(RACE_PQ)


def cond(X: np.ndarray) -> float:
    """Condition number of the standardised design -- scale-free."""
    Z = X - X.mean(0)
    sd = Z.std(0)
    sd[sd == 0] = 1.0
    return float(np.linalg.cond(Z / sd))


# --------------------------------------------------------------------------
# A1
# --------------------------------------------------------------------------

def a1(prac: pl.DataFrame) -> bool:
    print("=" * 78)
    print("A1  PRACTICE DECORRELATION")
    print("=" * 78)

    lr = fp.long_runs(prac, min_laps=4)
    print(f"  long-run laps {lr.height}   "
          f"runs {lr.select(['session_uid','driver','stint_id']).unique().height}   "
          f"sessions {lr['session_uid'].n_unique()}")
    print()

    age = lr["lap_in_run"].to_numpy().astype(float)
    minute = lr["session_clock_s"].to_numpy() / 60.0

    r = float(np.corrcoef(age, minute)[0, 1])
    X = np.column_stack([np.ones(len(age)), minute, age])
    k = cond(X[:, 1:])

    print(f"  corr(lap_in_run, session_clock)   r = {r:+.4f}   "
          f"{'PASS' if abs(r) < 0.30 else 'FAIL'}   (gate |r| < 0.30)")
    print(f"  condition number [minute, age]        {k:6.2f}   "
          f"{'PASS' if k < 30 else 'FAIL'}   (gate < 30)")
    print()

    # what actually binds, per A3: between-run stagger under partial pooling
    runs = (lr.group_by(["session_uid", "driver", "stint_id"])
              .agg([pl.col("session_clock_s").min().alias("t0"),
                    pl.len().alias("n")]))
    per_sess = (runs.group_by("session_uid")
                    .agg([(pl.col("t0").std() / 60.0).alias("stagger_min"),
                          pl.len().alias("n_runs"),
                          pl.col("n").mean().alias("laps_per_run")])
                    .sort("session_uid"))
    med_stagger = float(per_sess["stagger_min"].median())
    med_runs = float(per_sess["n_runs"].median())
    med_len = float(per_sess["laps_per_run"].median())

    print("  BETWEEN-RUN STAGGER  (what track evolution is actually identified from)")
    print(f"    median run-start SD per session   {med_stagger:5.1f} min")
    print(f"    median runs per session           {med_runs:5.0f}")
    print(f"    median laps per run               {med_len:5.1f}")

    # within a run the clock is nearly collinear with lap-in-run; quantify
    wr = []
    for (_, _, _), g in lr.group_by(["session_uid", "driver", "stint_id"],
                                    maintain_order=True):
        if g.height >= 4:
            a = g["lap_in_run"].to_numpy().astype(float)
            m = g["session_clock_s"].to_numpy() / 60.0
            if a.std() > 0 and m.std() > 0:
                wr.append(abs(float(np.corrcoef(a, m)[0, 1])))
    print(f"    median WITHIN-run |corr(age, clock)|  {np.median(wr):.4f}"
          "   <- ~1.0 means run fixed effects destroy tau")
    print()

    gate_r = abs(r) < 0.30 and k < 30
    print(f"  ORIGINAL GATE (|r|<0.30, cond<30): {'PASS' if gate_r else 'FAIL'}")
    print(f"    but VIF = 1/(1-r^2) = {1/(1-r*r):.3f}  -- that is benign collinearity,")
    print( "    and cond<30 is equivalent to |r|<0.998, so the two gates were the")
    print( "    same quantity inconsistently calibrated. See AMENDMENT below.")
    print()

    ok = a1_direct(lr)
    return ok


def a1_direct(lr: pl.DataFrame, target_s_per_session: float = 0.5) -> bool:
    """The corrected A1 test.

    AMENDMENT (2026-08-30). The pooled correlation is the wrong diagnostic: it
    mixes within-run variation (where clock and lap-in-run are collinear by
    construction, r = 1.0) with between-run variation (which is what actually
    identifies track evolution under partial pooling). A3 established that run
    FIXED effects destroy tau entirely, so the quantity that matters is the
    posterior SE on tau from a PARTIAL POOLING fit.

    New gate: a track-evolution effect of `target_s_per_session` over a 60-min
    session must be resolvable at z >= 2 from a single session.
    """
    from tyretrace.models import mixed

    print("  CORRECTED TEST -- posterior SE on track evolution, partial pooling")
    print("  (run random intercepts absorb driver AND unknown fuel load)")

    comps = ("SOFT", "MEDIUM", "HARD")
    ses = []
    for uid, g in lr.group_by("session_uid", maintain_order=True):
        uid = uid[0] if isinstance(uid, tuple) else uid
        if g.height < 60:
            continue
        minute = g["session_clock_s"].to_numpy() / 60.0
        age = g["lap_in_run"].to_numpy().astype(float)
        comp = g["compound_rel"].to_numpy().astype(str)
        y = g["lap_time_s"].to_numpy()
        present = [c for c in comps if (comp == c).sum() >= 8]
        if not present:
            continue

        X = np.column_stack([np.ones(len(y)), minute,
                             _dum(comp, present) * age[:, None]])
        names = ["intercept", "tau_per_min", *[f"slope_{c}" for c in present]]
        run = np.char.add(g["driver"].to_numpy().astype(str),
                          g["stint_id"].to_numpy().astype(str))
        try:
            fit = mixed.fit(X, y, run, names)
        except Exception:                                 # noqa: BLE001
            continue
        m, se = fit.get("tau_per_min")
        ses.append((uid, m, se, fit.sigma_u, fit.sigma_e))

    if not ses:
        print("    no fittable sessions")
        return False

    arr = np.array([[s[1], s[2], s[3], s[4]] for s in ses])
    med_se = float(np.median(arr[:, 1]))
    print(f"    sessions fitted                   {len(ses)}")
    print(f"    median tau estimate               {np.median(arr[:,0]):+.5f} s/min")
    print(f"    median SE(tau)                    {med_se:.5f} s/min")
    print(f"    median sigma_u (run intercepts)   {np.median(arr[:,2]):.3f} s"
          "   <- unknown fuel load")
    print(f"    median sigma_e (per lap)          {np.median(arr[:,3]):.3f} s")
    print()

    tau_target = target_s_per_session / 60.0
    z1 = tau_target / med_se
    print(f"    to resolve {target_s_per_session:.2f} s of evolution over a 60-min session")
    print(f"    (tau = {tau_target:.5f} s/min):")
    for n in (1, 3, 8, 20):
        zn = tau_target / (med_se / np.sqrt(n))
        print(f"      {n:>2d} session(s)  z = {zn:5.2f}"
              f"   {'resolvable' if zn >= 2 else 'below noise'}")
    print()

    ok = z1 >= 2.0
    print(f"  A1 GATE (amended): {'PASS' if ok else 'FAIL'}"
          f"   single-session z = {z1:.2f}")
    if not ok:
        need = int(np.ceil((2.0 * med_se / tau_target) ** 2))
        print(f"    -> track evolution needs ~{need} pooled sessions, not one.")
        print("    -> this is a REAL constraint the proposal did not have: pooled")
        print("       sessions must have comparable track evolution.")
    print()
    return ok


def _dum(labels: np.ndarray, levels) -> np.ndarray:
    return np.column_stack([(labels == lv).astype(float) for lv in levels])


# --------------------------------------------------------------------------
# A2
# --------------------------------------------------------------------------

def a2(race: pl.DataFrame, target_contrast: float = 0.008) -> bool:
    print("=" * 78)
    print("A2  RACE DESIGN CONDITIONING")
    print("=" * 78)

    lr = fp.long_runs(race, min_laps=4)
    print(f"  race laps {lr.height}   sessions {lr['session_uid'].n_unique()}")
    print()

    per_sess_sd, per_sess_r, sigmas = [], [], []

    for uid, g in lr.group_by("session_uid", maintain_order=True):
        uid = uid[0] if isinstance(uid, tuple) else uid
        drivers = g["driver"].to_numpy().astype(str)
        lap = g["lap_in_session"].to_numpy().astype(float)
        age = g["tyre_age"].to_numpy().astype(float)
        y = g["lap_time_s"].to_numpy()
        if len(y) < 100 or len(set(drivers)) < 5:
            continue

        lv = sorted(set(drivers))
        D = np.column_stack([(drivers == d).astype(float) for d in lv])
        X = np.column_stack([D, lap, age])

        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        dof = max(len(y) - np.linalg.matrix_rank(X), 1)
        s2 = float(resid @ resid) / dof
        xtx = np.linalg.pinv(X.T @ X)
        se_age = float(np.sqrt(max(xtx[-1, -1] * s2, 0.0)))

        per_sess_sd.append(se_age)
        per_sess_r.append(abs(float(np.corrcoef(lap, age)[0, 1])))
        sigmas.append(float(np.sqrt(s2)))

    per_sess_sd = np.array(per_sess_sd)
    med_sd = float(np.median(per_sess_sd))
    med_sig = float(np.median(sigmas))
    med_r = float(np.median(per_sess_r))

    print(f"  median |corr(race lap, tyre age)|     {med_r:.4f}"
          "   <- pit stops break the tie")
    print(f"  median per-lap residual SD            {med_sig:.4f} s")
    print(f"  median posterior SD on deg slope      {med_sd:.5f} s/lap"
          "   (single race, no pooling)")
    print()

    print("  POWER against the 2026 compound spread")
    print(f"    target contrast to resolve          {target_contrast:.4f} s/lap")
    z = target_contrast / (med_sd * np.sqrt(2))     # SD of a difference
    print(f"    single-race z for that contrast     {z:5.2f}")
    for n in (1, 3, 6, 12):
        zn = target_contrast / (med_sd * np.sqrt(2) / np.sqrt(n))
        print(f"    pooling {n:>2d} race(s)  z = {zn:5.2f}"
              f"   {'resolvable' if zn >= 2 else 'below noise'}")
    print()

    ok = med_sd < target_contrast
    print(f"  A2 GATE: {'PASS' if ok else 'FAIL'}"
          f"   (posterior SD {med_sd:.5f} vs contrast {target_contrast:.4f})")
    print()
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    args = ap.parse_args()

    prac, race = pull() if args.pull else load()
    print()
    ok1 = a1(prac)
    ok2 = a2(race)
    print("=" * 78)
    print(f"A1 {'PASS' if ok1 else 'FAIL'}    A2 {'PASS' if ok2 else 'FAIL'}")
    print("=" * 78)
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
