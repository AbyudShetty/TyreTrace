"""Artifact tables -- the contract between audits and the dashboard."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tyretrace import artifacts  # noqa: E402


def test_round_trip_and_missing_is_none(tmp_path):
    df = pl.DataFrame({"a": [1, 2]})
    artifacts.save("t", "x", df, root=tmp_path)
    assert artifacts.load("t", "x", root=tmp_path).equals(df)
    assert artifacts.load("t", "nope", root=tmp_path) is None


def test_corrupt_artifact_loads_as_none(tmp_path):
    p = artifacts.path("t", "x", tmp_path)
    p.parent.mkdir(parents=True)
    p.write_bytes(b"not parquet")
    assert artifacts.load("t", "x", root=tmp_path) is None


def test_predictive_quantiles_are_ordered_and_labelled():
    rng = np.random.default_rng(0)
    draws = rng.normal(size=(400, 30))
    y = rng.normal(size=30)
    t = artifacts.predictive(draws, y, driver=np.array(["VER"] * 30))
    cols = [artifacts.qcol(q) for q in artifacts.QUANTILES]
    assert cols == ["q05", "q25", "q50", "q75", "q95"]
    q = t.select(cols).to_numpy()
    assert np.all(np.diff(q, axis=1) >= 0)
    assert t["driver"][0] == "VER" and t.height == 30
    assert (t["crps"] >= 0).all()


def test_calibration_is_a_whole_curve_and_honest_on_a_true_model():
    rng = np.random.default_rng(1)
    draws = rng.normal(size=(2000, 4000))
    y = rng.normal(size=4000)            # drawn from the predictive itself
    cal = artifacts.calibration(draws, y)
    assert cal.height == len(artifacts.LEVELS)
    gap = (cal["coverage"] - cal["level"]).abs().to_numpy()
    assert gap.max() < 0.03


def test_posterior_skips_vectors_and_thins():
    samples = {"beta_f": np.arange(5000.0), "deg": np.ones((5000, 3))}
    t = artifacts.posterior(samples, ("beta_f", "deg", "absent"), max_draws=100,
                            derived={"fuel_pace": np.arange(5000.0) * 2})
    assert t.columns == ["beta_f", "fuel_pace"]
    assert t.height == 100
    assert np.allclose(t["fuel_pace"], 2 * t["beta_f"])
