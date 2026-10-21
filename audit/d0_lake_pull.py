"""D0 -- pull every race weekend since 2023 into the lake (OpenF1).

A thin wrapper so the task runner can call tyretrace.data.bulk as a script.

Run:  python audit/d0_lake_pull.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tyretrace.data import bulk  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(bulk.main([]))
