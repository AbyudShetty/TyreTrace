"""A5 -- the rig gate. Recover a hidden state the estimator is never shown.

The rig stream carries true fuel mass and true per-wheel wear. This script
strips every truth_* column, hands the estimator only the channels real F1
gives us (lap time, tyre age, lap number, compound, driver), and checks
whether it recovers what it could not see.

This is Tier 2 of the validation ladder, and it is the strongest single
experiment available: unlike real F1 data there IS a ground truth, and unlike
the synthetic twin the data has been through the full packet -> decoder ->
recorder -> canonical-schema path, so a bug anywhere in that chain shows up.

Gate: recovered degradation and fuel pace within 3 SE of the values the rig
was driven with. Fuel-reconstruction and wear checks are deliberately not
scored here -- they are tautologies of the generator until a live game
supplies non-linear burn and wear.

Run:  python audit/a5_rig_recovery.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audit.a3_synthetic_recovery import ols  # noqa: E402
from tyretrace.rig.fake import FakeSession  # noqa: E402
from tyretrace.rig.record import LapAccumulator  # noqa: E402
from tyretrace.rig import packets as P  # noqa: E402


def build_rig_laps(fs: FakeSession) -> pl.DataFrame:
    """Drive the full packet path and return canonical laps."""
    acc = LapAccumulator(f"rig_{fs.session_uid:016x}")
    for lap_pkt, status_pkt, damage_pkt in fs.stream():
        t = P.parse_header(lap_pkt).session_time
        acc.update(P.PACKET_CAR_STATUS, P.decode_car_status(status_pkt), t)
        acc.update(P.PACKET_CAR_DAMAGE, P.decode_car_damage(damage_pkt), t)
        acc.update(P.PACKET_LAP_DATA, P.decode_lap_data(lap_pkt), t)
    return acc.frame()


def estimate(observable: pl.DataFrame) -> dict:
    """The race pit-stop design, fitted on observable channels only."""
    drivers = observable["driver"].to_numpy().astype(str)
    lap = observable["lap_in_session"].to_numpy().astype(float)
    age = observable["tyre_age"].to_numpy().astype(float)
    y = observable["lap_time_s"].to_numpy()

    levels = sorted(set(drivers))
    X = np.column_stack([
        np.column_stack([(drivers == d).astype(float) for d in levels]),
        lap,        # fuel burn (no track evolution in the rig)
        age,        # degradation
    ])
    coef, se, _ = ols(X, y)
    return {"deg": (float(coef[-1]), float(se[-1])),
            "lap_coef": (float(coef[-2]), float(se[-2]))}


def main() -> int:
    fs = FakeSession(n_cars=20, n_laps=40, seed=11)
    truth_fuel_pace = fs.beta_f * fs.kappa

    print("=" * 76)
    print("A5  RIG RECOVERY -- hidden state the estimator never sees")
    print("=" * 76)
    print(f"   rig driven with   deg {fs.deg_per_lap:+.5f} s/lap"
          f"   beta_f {fs.beta_f:.4f} s/kg   kappa {fs.kappa:.3f} kg/lap")
    print(f"                     fuel pace = beta_f*kappa = {truth_fuel_pace:+.5f} s/lap")
    print()

    laps = build_rig_laps(fs)
    truth_cols = [c for c in laps.columns if c.startswith("truth_")]
    observable = laps.drop(truth_cols)

    print("-" * 76)
    print("   laps recorded        ", laps.height)
    print("   cars                 ", laps["driver"].n_unique())
    print("   stints               ",
          laps.select(["driver", "stint_id"]).unique().height)
    print("   truth columns hidden ", len(truth_cols), "->", ", ".join(truth_cols[:3]), "...")
    print("   columns given to fit ", observable.width)
    print("-" * 76)
    print()

    est = estimate(observable)
    ok = True

    m, se = est["deg"]
    hit = abs(m - fs.deg_per_lap) <= 3 * se
    ok &= hit
    print(f"   degradation   {m:+.5f} +/- {se:.5f}   true {fs.deg_per_lap:+.5f}"
          f"   err {m - fs.deg_per_lap:+.5f}   {'PASS' if hit else 'FAIL'}")

    m, se = est["lap_coef"]
    recovered = -m
    hit = abs(recovered - truth_fuel_pace) <= 3 * se
    ok &= hit
    print(f"   fuel pace     {recovered:+.5f} +/- {se:.5f}   true {truth_fuel_pace:+.5f}"
          f"   err {recovered - truth_fuel_pace:+.5f}   {'PASS' if hit else 'FAIL'}")

    implied_beta = recovered / fs.kappa
    print(f"   implied beta_f {implied_beta:+.5f} s/kg   true {fs.beta_f:+.5f}"
          "   (kappa supplied externally)")
    print()

    # --- deliberately NOT scored here ------------------------------------
    # Reconstructing fuel as fuel_start - kappa*lap and correlating it with
    # truth_fuel_kg returns r = 1.000 in the fake, because the fake burns fuel
    # by exactly that formula. Same for wear against tyre age. Both are
    # tautologies of the generator, not evidence about the estimator, and
    # reporting them as passing gates would be precisely the unfalsifiable
    # move this project accuses everyone else of.
    #
    # They become real tests against a live game, where fuel burn responds to
    # fuel mix, lift-and-coast and safety cars, and wear is non-linear in age
    # and differs per corner. Wire them up after --self-check passes.
    lap_n = laps["lap_in_session"].to_numpy().astype(float)
    recon = fs.fuel_start - fs.kappa * lap_n
    truth = laps["truth_fuel_kg"].to_numpy()
    print(f"   [not scored] fuel recon vs truth  MAE {np.mean(np.abs(recon-truth)):.3f} kg"
          "   circular in the fake; real test needs a live session")
    print( "   [not scored] wear vs tyre age     linear by construction here")
    print()

    from tyretrace import results
    results.record("a5", {
        "deg_est": est["deg"][0], "deg_true": fs.deg_per_lap,
        "fuel_pace_est": -est["lap_coef"][0],
        "fuel_pace_true": truth_fuel_pace,
        "laps": laps.height, "cars": laps["driver"].n_unique(),
    }, gate="PASS" if ok else "FAIL")

    print("=" * 76)
    print(f"A5 GATE: {'PASS' if ok else 'FAIL'}")
    print("=" * 76)
    print()
    print("NOTE: this proves the decode -> record -> estimate chain is correct")
    print("and self-consistent. It does NOT prove the byte offsets match a real")
    print("game build. Run `python -m tyretrace.rig.record --self-check` against")
    print("a live session before trusting a single real recorded lap.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
