"""Export pipeline results to JSON for the web dashboard (web/).

    python -m tyretrace export

The site is static: it reads only what this writes to web/public/data/. Same
contract as the Streamlit app -- audits fit and write, the front end only
reads -- so a page can never show a number no audit produced.

Every file is small, rounded to the precision the page displays, and safe to
commit, so a fresh clone opens a working site without running the pipeline.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from tyretrace import pipeline, results
from tyretrace.models import track_evolution as te
from tyretrace.planner import eig as E

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
DEST = ROOT / "web" / "public" / "data"


def _clean(x):
    """JSON-safe: numpy scalars to python, NaN/inf to null."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _write(name: str, obj) -> Path:
    DEST.mkdir(parents=True, exist_ok=True)
    p = DEST / name
    p.write_text(json.dumps(_clean(obj), separators=(",", ":")), encoding="utf-8")
    return p


def _m(task: str, key: str):
    return results.get(task).get("metrics", {}).get(key)


def _table(task: str, name: str):
    """An artifact table as a list of row dicts, or None if not produced yet."""
    from tyretrace import artifacts
    t = artifacts.load(task, name)
    return t.to_dicts() if t is not None else None


# --------------------------------------------------------------------------
# summary -- overview page
# --------------------------------------------------------------------------

def summary() -> dict:
    tasks = []
    for name, t in pipeline.TASKS.items():
        e = results.get(name)
        gate = e.get("gate")
        tasks.append({"task": name, "gate": gate, "expect": t.expect,
                      "why": t.expect_why, "summary": t.summary,
                      "as_expected": gate == t.expect if gate else None,
                      "recorded_at": e.get("recorded_at")})

    laps = {k: pl.read_parquet(OUT / f"audit_{k}.parquet")
            for k in ("race", "quali", "practice") if (OUT / f"audit_{k}.parquet").exists()}
    tau = te.summary(te.fit(laps["quali"])) if "quali" in laps else {}
    weekends = (sorted({u[:7] for u in laps["race"]["session_uid"].unique().to_list()})
                if "race" in laps else [])

    ladder = [{"rung": r, "crps": _m("l4", f"crps_{r.lower()}"),
               "mae": _m("l4", f"mae_{r.lower()}"), "cov90": _m("l4", f"cov90_{r.lower()}"),
               "rhat": _m("l4", f"rhat_{r.lower()}")} for r in ("L2", "L3", "L4", "L5")]
    twin = [{"rung": r, "crps": _m("ladder", f"twin_crps_{r.lower()}"),
             "cov90": _m("ladder", f"twin_cov90_{r.lower()}")} for r in ("L0", "L1")]
    base = [{"name": n, "key": k, "crps": _m("baselines", f"crps_{k}"),
             "mae": _m("baselines", f"mae_{k}"), "cov90": _m("baselines", f"cov90_{k}")}
            for k, n in (("b1", "Naive"), ("b2", "Conventional"), ("b3", "Gradient boosting"),
                         ("b4", "Published (race sim)"))]
    def bt_methods(task):
        return [{"method": m, **{k: _m(task, f"{k}_{m.replace('-', '_')}")
                                 for k in ("crps", "crps_after", "bias_after", "cov90",
                                           "stint_mae", "n", "n_after")}}
                for m in ("identified", "fuel-only", "no-track", "conventional", "naive")]

    bt = bt_methods("backtest")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {"weekends": len(weekends), "first": weekends[0] if weekends else None,
                 "last": weekends[-1] if weekends else None,
                 **{f"{k}_laps": v.height for k, v in laps.items()}},
        "gates": tasks,
        "headline": {
            "tau": {"median": tau.get("median_tau"), "frac_negative": tau.get("frac_negative"),
                    "z": tau.get("median_z"), "n": tau.get("n"),
                    "total_s": tau.get("median_total_s")},
            "fuel_pace": {"mean": _m("l2", "fuel_pace"), "q05": _m("l2", "fuel_pace_q05"),
                          "q95": _m("l2", "fuel_pace_q95"), "physics": [0.055, 0.060]},
            "tau_scale": {"mean": _m("l2", "tau_scale"), "q05": _m("l2", "tau_scale_q05"),
                          "q95": _m("l2", "tau_scale_q95"),
                          "p_lt_1": _m("l2", "p_tau_scale_lt_1")},
            "wake": _m("l3", "wake_s_per_lap"),
        },
        "ladder": ladder, "twin": twin, "baselines": base,
        "backtest": {"gate": results.get("backtest").get("gate"),
                     "weekends": _m("backtest", "weekends"), "methods": bt,
                     "slope_mae": {k: _m("backtest", f"slope_mae_{k}")
                                   for k in ("identified", "conventional", "mean")}},
        "backtest_lake": ({"gate": results.get("backtest_lake").get("gate"),
                           "weekends": _m("backtest_lake", "weekends"),
                           "methods": bt_methods("backtest_lake")}
                          if results.get("backtest_lake") else None),
        "fuel_gap": {t: {k: _m(t, k) for k in ("races", "mean_slope", "best",
                                               "beta_f_fuel_only", "beta_f_fuel_only_sd",
                                               "s_fuel_track", "s_fuel_track_sd",
                                               "loo_mae_physics_only", "loo_mae_fuel_only",
                                               "loo_mae_fuel_track", "loo_mae_mean")}
                     | {"gate": results.get(t).get("gate")}
                     for t in ("fuel_gap", "fuel_gap_lake") if results.get(t)},
        "postrace": {t: results.get(t).get("metrics", {})
                     for t in ("postrace", "postrace_lake") if results.get(t)},
        "ablations": _table("ablations", "table"),
        "parity": ({**results.get("parity").get("metrics", {}),
                    "gate": results.get("parity").get("gate")} if results.get("parity") else None),
        "headline_2026": ({**results.get("headline_2026").get("metrics", {}),
                           "gate": results.get("headline_2026").get("gate")}
                          if results.get("headline_2026") else None),
    }


# --------------------------------------------------------------------------
# planner parity fixtures -- the TypeScript port is tested against these
# --------------------------------------------------------------------------

def planner_fixtures() -> dict:
    """Reference values from the Python planner, which is itself checked
    against brute-force simulation (tests/test_planner.py). The web port must
    reproduce them -- web/src/lib/planner/eig.test.ts."""
    cases = []
    priors = {"default": E.Prior(),
              "noisy": E.Prior(sigma=0.5, beta_f=(0.033, 0.015), deg=(0.05, 0.06)),
              "diffuse_runs": E.Prior(run_sd=1e3)}
    progs = {
        "soft10": [E.Run("SOFT", 10, 60.0)],
        "swap16": [E.Run("SOFT", 16, 100.0, swap_to="MEDIUM")],
        "typical": [E.QUALI_SIM, E.Run("MEDIUM", 11, 90.0), E.Run("HARD", 11, 90.0)],
        "mixed": [E.Run("HARD", 8, 30.0), E.Run("MEDIUM", 12, 100.0, swap_to="HARD")],
    }
    for pn, pr in priors.items():
        for gn, runs in progs.items():
            prog = E.Programme()
            for r in runs:
                prog = prog.add(r, pr)
            cases.append({
                "prior": pn, "programme": gn,
                "runs": [{"compound": r.compound, "laps": r.laps,
                          "start_fuel_kg": r.start_fuel_kg, "swap_to": r.swap_to}
                         for r in runs],
                "starts": prog.starts,
                "eig": {t: E.eig(prog, pr, A) for t, A in E.TARGETS.items()},
                "posterior_sd": E.posterior_sd(prog, pr),
            })
    plans = []
    for reserve in (True, False):
        for swaps in (True, False):
            pr = E.Prior()
            A = E.TARGETS["degradation per compound"]
            start = E.Programme().add(E.QUALI_SIM, pr) if reserve else None
            steps = E.plan(pr, A, session_min=60.0,
                           cands=E.candidates(allow_swap=swaps), start=start)
            plans.append({"reserve": reserve, "swaps": swaps,
                          "bits": E.eig(E.programme(steps, start), pr, A),
                          "runs": [s.run.label() for s in steps]})
    return {
        "priors": {k: {"beta_f": v.beta_f, "deg": v.deg, "tau": v.tau, "run_sd": v.run_sd,
                       "sigma": v.sigma, "kappa": v.kappa, "lap_s": v.lap_s}
                   for k, v in priors.items()},
        "cases": cases, "plans": plans,
        "standard_bits": E.eig(E.standard_programme(E.Prior()), E.Prior(),
                               E.TARGETS["degradation per compound"]),
    }


def main() -> int:
    written = [_write("summary.json", summary())]
    fx = ROOT / "web" / "src" / "lib" / "planner" / "fixtures.json"
    fx.parent.mkdir(parents=True, exist_ok=True)
    fx.write_text(json.dumps(_clean(planner_fixtures()), indent=1), encoding="utf-8")
    written.append(fx)
    for p in written:
        print(f"  wrote {p.relative_to(ROOT)}  ({p.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
