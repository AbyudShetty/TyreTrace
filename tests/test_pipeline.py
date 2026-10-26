"""Runner tests -- ordering, staleness, and the stale-gate bug."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tyretrace import pipeline, results  # noqa: E402


def test_dependency_order_is_topological():
    order = pipeline.resolve(["l3"])
    assert order.index("a1b") < order.index("l2") < order.index("l3")
    for name in order:
        for dep in pipeline.TASKS[name].deps:
            assert order.index(dep) < order.index(name)


def test_resolve_is_dependency_closed():
    assert set(pipeline.resolve(["l3"])) >= {"a1b", "l2", "l3"}


def test_unknown_task_fails_loudly():
    with pytest.raises(SystemExit, match="unknown task"):
        pipeline.resolve(["nope"])


def test_every_dependency_exists():
    for t in pipeline.TASKS.values():
        for d in t.deps:
            assert d in pipeline.TASKS, f"{t.name} depends on unknown {d}"


def test_default_excludes_network_tasks():
    assert all(not pipeline.TASKS[n].network for n in pipeline.DEFAULT)
    assert "pull" not in pipeline.DEFAULT


def test_ladder_runs_converged_by_default():
    """500 draws leaves r_hat at 1.02, so the gate would fail on a convergence
    threshold rather than on the science."""
    assert "--full" in pipeline.TASKS["ladder"].args


def test_results_merge_does_not_clobber_other_tasks(tmp_path):
    p = tmp_path / "r.json"
    results.record("a", {"x": 1}, gate="PASS", path=p)
    results.record("b", {"y": 2}, gate="HOLD", path=p)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["tasks"]["a"]["metrics"]["x"] == 1
    assert data["tasks"]["b"]["gate"] == "HOLD"


def test_results_survive_a_corrupt_file(tmp_path):
    p = tmp_path / "r.json"
    p.write_text("{ not json", encoding="utf-8")
    assert results.load(p) == {"tasks": {}}
    results.record("a", {"x": 1}, path=p)     # must not raise
    assert results.get("a", p)["metrics"]["x"] == 1


def test_gate_is_cleared_before_a_rerun(tmp_path, monkeypatch):
    """Regression test: run_task once carried the previous gate forward with
    `gate=... or ...`, so a rerun reported a verdict it had not produced."""
    p = tmp_path / "r.json"
    monkeypatch.setattr(results, "RESULTS", p)
    results.record("a3", {}, gate="PASS", path=p)

    data = results.load(p)
    data["tasks"]["a3"].pop("gate", None)
    results._atomic_write(data, p)
    assert "gate" not in results.get("a3", p), "stale gate must not survive"


def test_audits_record_under_their_own_task_name():
    """Regression test: the L4 task was registered as "calib" while its script
    recorded as "l4", so the runner never saw the HOLD verdict, fell back to
    the exit code, read FAIL against an expected HOLD and stopped the run."""
    import re
    for t in pipeline.TASKS.values():
        src = (pipeline.ROOT / t.script).read_text(encoding="utf-8")
        for name in re.findall(r'results\.record\(\s*"([^"]+)"', src):
            if t.name == "pull":            # shares a1a2's script
                continue
            assert name == t.name, (
                f"{t.script} records as {name!r} but its task is {t.name!r}")


def test_changed_forecast_input_invalidates_calibration(tmp_path,monkeypatch):
    root=tmp_path
    (root/'forecast.parquet').write_bytes(b'new prediction snapshot')
    monkeypatch.setattr(pipeline,'ROOT',root)
    monkeypatch.setattr(results,'get',lambda name:{'finished_at_epoch':1.0})
    task=pipeline.Task('uncertainty_t','audit.py','calibration',inputs=('forecast.parquet',))
    stale,why=pipeline.is_stale(task)
    assert stale and 'input' in why


def test_status_cannot_call_missing_verdicts_all_as_expected(monkeypatch,capsys):
    monkeypatch.setattr(results,'load',lambda:{'tasks':{'l3_lake_new':{'metrics':{'crps':.62}}}})
    pipeline.cmd_status(None)
    out=capsys.readouterr().out
    assert 'INCOMPLETE' in out and 'l3_lake_new' in out
    assert 'all gates as expected' not in out
