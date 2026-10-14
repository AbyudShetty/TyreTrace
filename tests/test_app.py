"""Dashboard smoke test -- every page renders without raising.

Runs each page headlessly through Streamlit's AppTest. With out/ populated it
exercises every chart; without it each page must degrade to a "run this task"
hint rather than a traceback, so this passes on a fresh clone too.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

VIEWS = sorted((ROOT / "tyretrace" / "app" / "views").glob("*.py"))


@pytest.mark.parametrize("view", VIEWS, ids=lambda p: p.stem)
def test_page_renders(view):
    at = AppTest.from_file(str(view), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]


def test_every_view_is_in_the_navigation():
    main = (ROOT / "tyretrace" / "app" / "main.py").read_text(encoding="utf-8")
    for v in VIEWS:
        assert f'page("{v.stem}"' in main, f"{v.stem}.py is not reachable from the nav"
