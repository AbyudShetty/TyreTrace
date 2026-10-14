"""One results file the whole project writes to and the frontend reads from.

Audits print for a human; this records for a machine. Every headline number
lands in ``out/results.json`` so the runner can summarise gates, and a
frontend can render them, without anybody parsing stdout.

Writes are atomic (temp file + replace) and merge rather than overwrite, so a
partial run never truncates results from tasks that already succeeded.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULTS = Path(__file__).resolve().parents[1] / "out" / "results.json"


def load(path: Path = RESULTS) -> dict:
    if not path.exists():
        return {"tasks": {}}
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        # a corrupt results file must not brick the pipeline
        return {"tasks": {}}
    data.setdefault("tasks", {})
    return data


def _atomic_write(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True, default=str)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def record(task: str, metrics: dict[str, Any], *, gate: str | None = None,
           path: Path = RESULTS) -> None:
    """Merge one task's metrics into the results file.

    `gate` is the pass/fail/hold verdict, kept separate from the numbers so a
    reader never has to infer a verdict from a metric.
    """
    data = load(path)
    entry = data["tasks"].get(task, {})
    entry.update({
        "metrics": {**entry.get("metrics", {}), **metrics},
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    if gate is not None:
        entry["gate"] = gate
    data["tasks"][task] = entry
    _atomic_write(data, path)


def get(task: str, path: Path = RESULTS) -> dict:
    return load(path)["tasks"].get(task, {})
