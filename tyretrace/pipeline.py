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
         expect="HOLD",
         expect_why="improves CRPS but coverage 0.849 is below the 0.85 band"),

    Task("l4", "audit/l4_calibration.py",
         "L4/L5: calibration -- stint offsets, heavy tails, variance model",
         deps=("l3",), inputs=("out/audit_race.parquet",), minutes=8.0,
         expect="HOLD",
         expect_why="L4 has the best CRPS but covers 0.813; L5 rejected, "
                    "sigma(age) is U-shaped not monotone"),
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
            print(f"        run:  python -m tyretrace run pull\n")
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
    for n, t in TASKS.items():
        e = data["tasks"].get(n, {})
        gate = e.get("gate", "-")
        dur = e.get("metrics", {}).get("duration_s")
        note = ""
        if gate == "-":
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

    a = sub.add_parser("app", help="open the dashboard")
    a.add_argument("--port", type=int, default=8501)
    a.set_defaults(fn=cmd_app)

    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        ap.print_help()
        return 0
    return args.fn(args)
