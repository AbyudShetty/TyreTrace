"""Web export -- the JSON contract the dashboard (web/src/lib/data.ts) reads."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tyretrace import export  # noqa: E402


def test_clean_makes_numpy_and_nan_json_safe():
    out = export._clean({"a": np.float32(1.5), "b": float("nan"), "c": [np.int64(3), np.bool_(True)],
                         "d": (math.inf,)})
    assert out == {"a": 1.5, "b": None, "c": [3, True], "d": [None]}
    json.dumps(out, allow_nan=False)            # must not raise


def test_planner_fixtures_cover_every_prior_programme_and_plan():
    fx = export.planner_fixtures()
    assert len(fx["cases"]) == len(fx["priors"]) * 4
    assert {(p["reserve"], p["swaps"]) for p in fx["plans"]} == {
        (True, True), (True, False), (False, True), (False, False)}
    for c in fx["cases"]:
        assert set(c["eig"]) == {"degradation per compound", "SOFT − HARD contrast",
                                 "degradation + fuel effect"}
        assert all(v >= 0 for v in c["eig"].values())
    json.dumps(export._clean(fx), allow_nan=False)


def test_summary_has_the_keys_the_web_types_declare():
    s = export.summary()
    assert {"generated_at", "data", "gates", "headline", "ladder", "twin",
            "baselines", "backtest"} <= set(s)
    assert {"task", "gate", "expect", "why", "summary", "as_expected"} <= set(s["gates"][0])
    json.dumps(export._clean(s), allow_nan=False)
