"""Task runner. One command instead of eight, in dependency order.

    python -m tyretrace run all        everything that does not need network
    python -m tyretrace run l3         that task and whatever it needs
    python -m tyretrace list           the graph and what is stale
    python -m tyretrace status         last run's gates
    python -m tyretrace app            the dashboard, reading out/

Tasks are the audit scripts, executed as subprocesses so a crash in one is
contained and reported rather than taking the run down. Each records its
headline numbers to out/results.json via tyretrace.results, which is what the
runner summarises and what a frontend will read.

Staleness is deliberate and conservative: a task reruns if its outputs are
missing, if its own source changed, or if anything it depends on is newer than
its last successful run. Nothing is cached across a code edit.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from tyretrace import results

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"


@dataclass(frozen=True)
class Task:
    name: str
    script: str                       # path relative to repo root
    summary: str
    deps: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()      # artifacts that must already exist
    outputs: tuple[str, ...] = ()     # artifacts this produces
    network: bool = False
    minutes: float = 1.0
    args: tuple[str, ...] = field(default_factory=tuple)
    # The gate we EXPECT. Some tasks are documented negative results -- A1's
    # practice-decorrelation premise was overturned by measurement, and that
    # failure is a finding, not breakage. Recording the expectation keeps a
    # real regression visible instead of hiding in a wall of red.
    expect: str = "PASS"
    expect_why: str = ""
    # Optional tasks need data a fresh clone may not have (the lake). Missing
    # inputs SKIP them quietly instead of failing the run.
    optional: bool = False
    pull_hint: str = "python -m tyretrace run pull"


LAKE_INPUTS = ("out/lake/race.parquet", "out/lake/quali.parquet", "out/lake/practice.parquet")

TASKS: dict[str, Task] = {t.name: t for t in [
    Task("pull", "audit/a1_a2_identification.py",
         "Fetch 40 sessions from FastF1 (practice, quali, race)",
         outputs=("out/audit_practice.parquet", "out/audit_race.parquet"),
         network=True, minutes=15.0, args=("--pull",)),

    Task("a3", "audit/a3_synthetic_recovery.py",
         "Identification core on the synthetic twin",
         minutes=1.0),

    Task("a5", "audit/a5_rig_recovery.py",
         "Game rig: recover hidden state from a synthetic packet stream",
         minutes=0.5),

    Task("a1a2", "audit/a1_a2_identification.py",
         "Practice decorrelation (fails) and race conditioning (passes)",
         inputs=("out/audit_practice.parquet", "out/audit_race.parquet"),
         minutes=1.5, expect="FAIL",
         expect_why="A1's premise was overturned; the quali anchor (a1b) replaces it"),

    Task("a1b", "audit/a1b_track_evolution.py",
         "Qualifying anchor for track evolution",
         inputs=("out/audit_quali.parquet",), minutes=1.0),

    Task("ladder", "audit/l0_l1_ladder.py",
         "L0/L1: identification core in NumPyro, ship gate",
         deps=("a1b",), inputs=("out/audit_race.parquet",), minutes=6.0,
         # 500 draws leaves r_hat at 1.02 and the gate fails on a convergence
         # threshold rather than on the science. Run it converged by default;
         # a gate that only passes with a flag is a trap.
         args=("--full",)),

    Task("l2", "audit/l2_tau_scale.py",
         "L2: estimate the fuel/track split; compound contrasts (E1)",
         deps=("a1b",), inputs=("out/audit_race.parquet",), minutes=3.0),

    Task("l3", "audit/l3_dirty_air.py",
         "L3: dirty air covariate and its ship gate against L2",
         deps=("l2",), inputs=("out/audit_race.parquet",), minutes=3.0,
         expect="SHIP",
         expect_why="since the per-weekend slope fix (2026-10-21): CRPS -4.0% vs L2, "
                    "coverage 0.858 -- the first rung to ship"),

    Task("l4", "audit/l4_calibration.py",
         "L4/L5: calibration -- stint offsets, heavy tails, variance model",
         deps=("l3",), inputs=("out/audit_race.parquet",), minutes=8.0,
         expect="SHIP",
         expect_why="L3 ships (coverage 0.858); L4 does not (0.807, r_hat 1.10); "
                    "L5 rejected, sigma(age) is U-shaped not monotone"),

    Task("baselines", "audit/b1_baselines.py",
         "Protocol baselines (naive, conventional, GBM) on the ladder's split",
         deps=("l4",), inputs=("out/audit_race.parquet",), minutes=1.5),

    Task("backtest", "audit/t3_backtest.py",
         "Tier 3: forecast each race from its practice, rolling origin",
         deps=("a1b",), minutes=0.5,
         inputs=("out/audit_race.parquet", "out/audit_quali.parquet",
                 "out/audit_practice.parquet"),
         expect="FAIL",
         expect_why="beats naive by 63% after stops, but the fixed 0.030 s/kg "
                    "beats the race-learned fuel effect by 3.9% -- evidence the "
                    "race identification of fuel is biased low"),

    Task("fuel_gap", "audit/f1_fuel_gap.py",
         "Fuel-pace gap: is the missing race slope fuel or track?",
         deps=("a1b",), minutes=0.2,
         inputs=("out/audit_race.parquet", "out/audit_quali.parquet")),

    Task("postrace", "audit/p1_postrace.py",
         "Post-race validation: forecast error split by cause",
         deps=("backtest",), minutes=0.3,
         inputs=("out/audit_race.parquet",)),

    Task("ablations", "audit/a6_ablations.py",
         "Ablation table: every component removed one at a time",
         deps=("backtest", "l4", "ladder"), minutes=0.5,
         inputs=("out/audit_race.parquet", "out/audit_quali.parquet",
                 "out/audit_practice.parquet")),

    # ---- the lake: every weekend since 2023, via OpenF1 ----------------------
    Task("lake", "audit/d0_lake_pull.py",
         "Pull every weekend since 2023 from OpenF1 (resumable)",
         network=True, minutes=55.0, outputs=("out/lake/race.parquet", "out/lake/quali.parquet", "out/lake/practice.parquet")),

    Task("parity", "audit/d1_source_parity.py",
         "Source parity: FastF1 vs OpenF1 on every shared session",
         inputs=("out/lake/race.parquet", "out/lake/quali.parquet", "out/lake/practice.parquet"), optional=True, minutes=0.5,
         pull_hint="python -m tyretrace run lake"),

    Task("backtest_lake", "audit/t3_backtest.py",
         "Tier 3 on every lake weekend; gate on pre-registered fuel-only",
         deps=("parity",), inputs=("out/lake/race.parquet", "out/lake/quali.parquet", "out/lake/practice.parquet"), optional=True, minutes=2.0,
         args=("--data", "lake", "--traffic"), pull_hint="python -m tyretrace run lake",
         expect="FAIL",
         expect_why='pre-registered fuel-only does not beat conventional on 68 dry weekends; fuel-corrected methods tie (paired t < 2), all beat naive (t -9.6)'),

    Task("fuel_gap_lake", "audit/f1_fuel_gap.py",
         "Fuel-pace gap on every lake race",
         deps=("parity",), inputs=("out/lake/race.parquet", "out/lake/quali.parquet", "out/lake/practice.parquet"), optional=True, minutes=0.5,
         args=("--data", "lake"), pull_hint="python -m tyretrace run lake",
         expect="FAIL",
         expect_why='on 71 dry races a small race track evolution (s 0.35 +/- 0.14) earns its place; fuel 0.027 s/kg with it, 0.032 without -- gap largely closed'),

    Task("postrace_lake", "audit/p1_postrace.py",
         "Post-race validation on every lake weekend",
         deps=("backtest_lake",), inputs=("out/lake/race.parquet", "out/lake/quali.parquet", "out/lake/practice.parquet"), optional=True, minutes=1.0,
         args=("--data", "lake"), pull_hint="python -m tyretrace run lake"),

    Task("l3_lake_new", "audit/l3_dirty_air.py",
         "L3 ship gate on lake weekends the slope fix never saw",
         deps=("parity",), inputs=LAKE_INPUTS, optional=True, minutes=15.0,
         args=("--data", "lake", "--new-only"), pull_hint="python -m tyretrace run lake",
         expect="HOLD",
         expect_why="the 10-weekend L3 ship does not replicate on 61 new weekends: coverage 0.797, CRPS 0.626 vs L2 0.624"),

    Task("headline_2026", "audit/h1_2026.py",
         "2026 headline: every 2026 race from its practice, both fuel loads",
         deps=("parity",), inputs=("out/lake/race.parquet", "out/lake/quali.parquet", "out/lake/practice.parquet"), optional=True, minutes=1.0,
         pull_hint="python -m tyretrace run lake",
         expect="FAIL",
         expect_why='verdict flips with the unverified 2026 fuel load; identified best under 70 kg (-8%) and most robust; 2026 race slopes contradict 70 kg'),

    Task("uncertainty_t", "audit/u1_uncertainty.py",
         "Predictive uncertainty: rolling Student-t calibration and fuel sensitivity",
         inputs=("out/artifacts/backtest_lake/predictive.parquet", *LAKE_INPUTS),
         outputs=("out/artifacts/uncertainty_t/predictive.parquet",
                  "out/artifacts/uncertainty_t/predictive_110.parquet",
                  "out/artifacts/uncertainty_t/fits.parquet"),
         optional=True, minutes=1.0, args=("--fuel-sensitivity",),
         pull_hint="python -m tyretrace run backtest_lake",
         expect="SHIP", expect_why="retrospective rolling lap-time calibration; "
                    "CRPS improves and coverage is in band under both fuel assumptions"),

    Task("strategy_reference", "audit/s1_strategy.py",
         "Conditional one-stop error/regret versus a fitted post-race reference",
         inputs=("out/artifacts/backtest_lake/params.parquet", *LAKE_INPUTS),
         outputs=("out/artifacts/strategy_reference/decisions.parquet",
                  "out/artifacts/strategy_reference/summary.parquet"),
         optional=True, minutes=0.2,
         pull_hint="python -m tyretrace run backtest_lake"),

    Task('published_verified','audit/b4_published_parity.py',
         'Source-verified TUM tyre curve baseline; diagnostic comparison',
         inputs=('out/audit_race.parquet','out/audit_quali.parquet'),
         outputs=('out/artifacts/published_verified/predictive.parquet',),
         optional=True,minutes=0.3),
]}

# everything runnable without touching the network
DEFAULT = tuple(n for n, t in TASKS.items() if not t.network)


def resolve(names: list[str]) -> list[str]:
    """Expand to dependency-closed, topologically ordered task names."""
    order, seen = [], set()

    def visit(n: str, stack: tuple[str, ...] = ()) -> None:
        if n in stack:
            raise ValueError(f"dependency cycle: {' -> '.join(stack + (n,))}")
        if n in seen:
            return
        if n not in TASKS:
            raise SystemExit(f"unknown task {n!r}. known: {', '.join(TASKS)}")
        for d in TASKS[n].deps:
            visit(d, stack + (n,))
        seen.add(n)
        order.append(n)

    for n in names:
        visit(n)
    return order


def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


def is_stale(t: Task) -> tuple[bool, str]:
    """Conservative: any doubt means rerun."""
    entry = results.get(t.name)
    last = entry.get("finished_at_epoch")
    if not last:
        return True, "never run"

    for o in t.outputs:
        if not (ROOT / o).exists():
            return True, f"missing output {o}"

    if _mtime(ROOT / t.script) > last:
        return True, "script changed"

    for p in t.inputs:
        if _mtime(ROOT / p) > last:
            return True, f"input {p} changed"

    # a code edit anywhere in the package invalidates everything; cheaper to
    # be wrong about a rerun than to report a stale gate as current
    newest_src = max((_mtime(p) for p in (ROOT / "tyretrace").rglob("*.py")),
                     default=0.0)
    if newest_src > last:
        return True, "package changed"

    for d in t.deps:
        if results.get(d).get("finished_at_epoch", 0) > last:
            return True, f"dependency {d} reran"

    return False, "up to date"


def missing_inputs(t: Task) -> list[str]:
    return [i for i in t.inputs if not (ROOT / i).exists()]


def run_task(t: Task, *, extra: list[str] | None = None) -> tuple[bool, float]:
    cmd = [sys.executable, str(ROOT / t.script), *t.args, *(extra or [])]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    # Clear the previous verdict FIRST. Carrying it forward would let a stale
    # gate survive a rerun -- the run reports a result it did not produce,
    # which is the worst failure mode a results file can have.
    data = results.load()
    data["tasks"].setdefault(t.name, {}).pop("gate", None)
    results._atomic_write(data, results.RESULTS)

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    dur = time.time() - t0
    ok = proc.returncode == 0

    # An audit that recorded its own gate during this run wins; otherwise fall
    # back to the exit code.
    self_reported = results.get(t.name).get("gate")
    results.record(t.name, {
        "duration_s": round(dur, 1),
        "exit_code": proc.returncode,
        "script": t.script,
        "args": list(t.args) + list(extra or []),
    }, gate=self_reported or ("PASS" if ok else "FAIL"))

    # stamp completion separately so staleness survives a metrics-only write
    data = results.load()
    data["tasks"].setdefault(t.name, {})["finished_at_epoch"] = time.time()
    results._atomic_write(data, results.RESULTS)
    return ok, dur


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_run(args) -> int:
    names = list(DEFAULT) if args.tasks in (["all"], []) else args.tasks
    order = resolve(names)

    plan = []
    for n in order:
        t = TASKS[n]
        stale, why = is_stale(t)
        plan.append((t, stale or args.force, why if not args.force else "forced"))

    est = sum(t.minutes for t, run, _ in plan if run)
    print(f"{len(plan)} task(s), {sum(1 for _, r, _ in plan if r)} to run, "
          f"~{est:.0f} min\n")

    failed = []
    for t, should_run, why in plan:
        miss = missing_inputs(t)
        if miss:
            print(f"  SKIP  {t.name:<8s} needs {', '.join(miss)}")
            print(f"        run:  {t.pull_hint}\n")
            if not t.optional:
                failed.append(t.name)
            continue
        if not should_run:
            gate = results.get(t.name).get("gate", "?")
            print(f"  CACHE {t.name:<8s} {gate:<5s} ({why})")
            continue

        print(f"  RUN   {t.name:<8s} {t.summary}  [{why}]")
        print("  " + "-" * 70)
        ok, dur = run_task(t, extra=(["--full"] if args.full else None))
        print("  " + "-" * 70)
        print(f"  {'OK' if ok else 'FAILED'}  {t.name} in {dur:.0f}s\n")
        gate = results.get(t.name).get("gate")
        if not ok and gate != t.expect:
            failed.append(t.name)
        elif not ok:
            code = results.get(t.name)["metrics"]["exit_code"]
            print(f"  (exit {code} is the expected {t.expect} outcome, "
                  "not a break)\n")
            continue
        if not ok:
            if not args.keep_going:
                print("  stopping (use --keep-going to continue)\n")
                break

    return cmd_status(args, exit_nonzero=bool(failed))


def cmd_status(args, exit_nonzero: bool = False) -> int:
    data = results.load()
    print("=" * 72)
    print(f"{'TASK':<10s} {'GATE':<6s} {'TIME':>7s}   SUMMARY")
    print("=" * 72)
    surprises = []
    incomplete = []
    for n, t in TASKS.items():
        e = data["tasks"].get(n, {})
        gate = e.get("gate", "-")
        dur = e.get("metrics", {}).get("duration_s")
        note = ""
        if gate == "-":
            if e:
                note = "INCOMPLETE: saved metrics have no recorded verdict"
                incomplete.append(n)
            else:
                note = "not run"
        elif gate != t.expect:
            note = f"UNEXPECTED (wanted {t.expect})"
            surprises.append(n)
        elif t.expect != "PASS":
            note = f"expected: {t.expect_why}"
        print(f"{n:<10s} {gate:<6s} {(f'{dur:.0f}s' if dur else '-'):>7s}   "
              f"{t.summary[:40]}")
        if note:
            print(f"{'':<10s} {'':<6s} {'':>7s}   -> {note}")
    print("=" * 72)
    if surprises:
        print(f"UNEXPECTED gates: {', '.join(surprises)}")
    elif incomplete:
        print("recorded verdicts match expectations; incomplete tasks: " + ', '.join(incomplete))
    else:
        print("all gates as expected")
    print(f"results: {results.RESULTS.relative_to(ROOT)}")
    return 1 if (exit_nonzero or surprises) else 0


def cmd_list(args) -> int:
    for n in resolve(list(TASKS)):
        t = TASKS[n]
        stale, why = is_stale(t)
        miss = missing_inputs(t)
        flag = "NEEDS INPUT" if miss else ("stale" if stale else "cached")
        dep = f" <- {', '.join(t.deps)}" if t.deps else ""
        net = " [network]" if t.network else ""
        print(f"  {n:<10s} {flag:<12s} ~{t.minutes:>4.1f} min{net}{dep}")
        print(f"             {t.summary}")
    return 0


def cmd_app(args) -> int:
    """Launch the dashboard. It only reads out/, so run tasks first."""
    app = ROOT / "tyretrace" / "app" / "main.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app),
           "--server.port", str(args.port)]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        return subprocess.run(cmd, cwd=str(ROOT), env=env).returncode
    except KeyboardInterrupt:
        return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tyretrace",
                                 description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="run tasks and their dependencies")
    r.add_argument("tasks", nargs="*", default=["all"])
    r.add_argument("--force", action="store_true", help="ignore the cache")
    r.add_argument("--full", action="store_true", help="more MCMC draws")
    r.add_argument("--keep-going", action="store_true")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("status", help="gates from the last run")
    s.set_defaults(fn=lambda a: cmd_status(a))

    l = sub.add_parser("list", help="task graph and staleness")
    l.set_defaults(fn=cmd_list)

    x = sub.add_parser("export", help="write JSON for the web dashboard")
    x.set_defaults(fn=lambda a: __import__("tyretrace.export", fromlist=["main"]).main())

    a = sub.add_parser("app", help="open the dashboard")
    a.add_argument("--port", type=int, default=8501)
    a.set_defaults(fn=cmd_app)

    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        ap.print_help()
        return 0
    return args.fn(args)
