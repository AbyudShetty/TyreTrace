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


def test_cached_child_is_rechecked_after_its_dependency_reruns(monkeypatch):
    from types import SimpleNamespace
    parent = pipeline.Task('parent', 'parent.py', 'parent')
    child = pipeline.Task('child', 'child.py', 'child', deps=('parent',))
    completed = []
    monkeypatch.setattr(pipeline, 'TASKS', {'parent': parent, 'child': child})
    monkeypatch.setattr(pipeline, 'is_stale', lambda t: (
        t.name == 'parent' or 'parent' in completed, 'dependency changed'))
    monkeypatch.setattr(pipeline, 'missing_inputs', lambda t: [])
    monkeypatch.setattr(pipeline, 'run_task', lambda t, **kw: (completed.append(t.name) is None, 0))
    monkeypatch.setattr(results, 'get', lambda n: {'gate': 'PASS'})
    monkeypatch.setattr(pipeline, 'cmd_status', lambda *a, **kw: 0)
    args = SimpleNamespace(tasks=['child'], force=False, full=False, keep_going=True)
    assert pipeline.cmd_run(args) == 0
    assert completed == ['parent', 'child']


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


def test_failed_execution_is_never_treated_as_a_fresh_cached_gate(monkeypatch):
    task=pipeline.Task('test','audit.py','example',expect='SHIP')
    monkeypatch.setattr(results,'get',lambda name:{'gate':'FAIL','finished_at_epoch':9e12,'metrics':{'exit_code':1}})
    stale,why=pipeline.is_stale(task)
    assert stale and 'failed' in why


def test_windows_process_crash_cannot_masquerade_as_an_expected_negative_result(monkeypatch,capsys):
    task=pipeline.Task('backtest','audit.py','example',expect='FAIL')
    entry={'gate':'FAIL','finished_at_epoch':9e12,'metrics':{'exit_code':3221226091}}
    monkeypatch.setattr(results,'get',lambda name:entry)
    assert pipeline.is_stale(task)==(True,'previous audit process crashed')
    monkeypatch.setattr(results,'load',lambda:{'tasks':{'backtest':entry}})
    assert pipeline.cmd_status(None)==1
    assert 'audit process crashed' in capsys.readouterr().out


def test_python_exception_without_a_verdict_is_incomplete_even_if_fail_is_expected(tmp_path,monkeypatch):
    from types import SimpleNamespace
    path=tmp_path/'results.json'
    load,get,record=results.load,results.get,results.record
    monkeypatch.setattr(results,'RESULTS',path)
    monkeypatch.setattr(results,'load',lambda p=path:load(p))
    monkeypatch.setattr(results,'get',lambda name,p=path:get(name,p))
    monkeypatch.setattr(results,'record',lambda name,metrics,**kw:record(name,metrics,path=path,**kw))
    task=pipeline.Task('negative','audit.py','negative experiment',expect='FAIL')
    results.record(task.name,{},gate='FAIL')
    monkeypatch.setattr(pipeline.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1))
    ok,_=pipeline.run_task(task)
    entry=results.get(task.name)
    assert not ok and not entry.get('gate')
    assert entry['metrics']['execution_incomplete']
    assert pipeline.is_stale(task)[0]


def test_app_command_launches_react_without_a_model_or_network_pull(tmp_path,monkeypatch):
    from types import SimpleNamespace
    (tmp_path/'web/node_modules/vite').mkdir(parents=True)
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    monkeypatch.setattr(pipeline.shutil,'which',lambda name:'npm.cmd')
    calls=[]
    def run(cmd,**kwargs):
        calls.append((cmd,kwargs));return SimpleNamespace(returncode=0)
    monkeypatch.setattr(pipeline.subprocess,'run',run)
    assert pipeline.cmd_app(SimpleNamespace(port=None,legacy=False))==0
    cmd,kwargs=calls[0]
    assert cmd[:3]==['npm.cmd','run','dev'] and '5173' in cmd and '--strictPort' in cmd
    assert kwargs['cwd']==str(tmp_path/'web')


def test_presentation_changes_do_not_refit_models_but_science_and_args_do(tmp_path,monkeypatch):
    import os
    pkg=tmp_path/'tyretrace';pkg.mkdir();model=pkg/'scientific.py';model.write_text('x=1')
    ui=pkg/'web_data.py';ui.write_text('updated export')
    os.utime(model,(1,1));os.utime(ui,(100,100))
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    monkeypatch.setattr(results,'get',lambda n:{'finished_at_epoch':50,'gate':'PASS','metrics':{'args':[]}})
    task=pipeline.Task('x','audit.py','example')
    assert not pipeline.is_stale(task)[0]
    assert pipeline.is_stale(pipeline.Task('x','audit.py','example',args=('--different',)))[0]
    os.utime(model,(100,100))
    assert pipeline.is_stale(task)==(True,'package changed')


def test_status_cannot_call_missing_verdicts_all_as_expected(monkeypatch,capsys):
    monkeypatch.setattr(results,'load',lambda:{'tasks':{'l3_lake_new':{'metrics':{'crps':.62}}}})
    pipeline.cmd_status(None)
    out=capsys.readouterr().out
    assert 'INCOMPLETE' in out and 'l3_lake_new' in out
    assert 'all gates as expected' not in out
