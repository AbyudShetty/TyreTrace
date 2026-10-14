"""Minimal linear-Gaussian twin. Known truth, canonical schema, no JAX.

Deliberately the simplest thing that reproduces the identification problem:
linear fuel burn, linear degradation, linear track evolution, Gaussian noise.
If the estimator cannot recover truth here it will not recover it anywhere,
and this runs in milliseconds so it can sit in CI.

Sign convention: lap_time RISES with tyre age (degradation) and RISES with
fuel mass; track evolution LOWERS it as the session goes on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from tyretrace import schema


@dataclass(frozen=True)
class Truth:
    """Ground-truth parameters. The thing the estimators must recover."""
    beta_f: float = 0.033          # s per kg
    kappa: float = 1.75            # kg burned per lap
    # degradation, s per lap of tyre age, by absolute compound
    deg: dict = field(default_factory=lambda: {"C2": 0.040, "C3": 0.055, "C4": 0.075})
    tau_race: float = 0.010        # s/lap of track evolution during a race
    tau_practice: float = 0.016    # s per session-minute (measured: -0.0165)
    sigma: float = 0.20            # per-lap noise, s
    base_lap: float = 90.0

    # --- the practice programme, measured from real data ------------------
    # Teams run low-fuel quali sims early and high-fuel race sims late, so
    # starting fuel TRENDS with the session clock. Measured on 10 weekends:
    # corr(run start minute, run pace) = +0.54. This is what breaks the
    # "cars run different fuel loads at the same minute" decoupling argument.
    fuel_early_kg: float = 40.0
    fuel_late_kg: float = 95.0
    # Long runs are also driven in management mode, so they are slower for
    # reasons beyond fuel. This is why run length is a CONTAMINATED proxy for
    # fuel load: measured coefficient +0.22 s per lap of run length, against
    # +0.029 predicted by fuel alone.
    manage_s_per_lap_of_run: float = 0.19

    @property
    def fuel_pace(self) -> float:
        """beta_f * kappa -- the s/lap quantity the data can actually see."""
        return self.beta_f * self.kappa


def _row(**kw) -> dict:
    """A canonical row with every optional field defaulted to null."""
    base = {
        "sector_1_s": None, "sector_2_s": None, "sector_3_s": None,
        "fuel_load_kg_est": None, "fuel_load_kg_sd": None,
        "gap_ahead_s": None, "lift_coast_m": None,
        "track_temp_c": None, "air_temp_c": None,
        "track_status": "1", "is_valid": True,
        "pit_in": False, "pit_out": False,
        "compound_rel": "UNKNOWN", "is_fresh": True,
        "corner_feats": None,
        "truth_wear_fl": None, "truth_wear_fr": None,
        "truth_wear_rl": None, "truth_wear_rr": None,
        "truth_regime": None,
    }
    base.update(kw)
    return base


def simulate_race(
    rng: np.random.Generator,
    truth: Truth = Truth(),
    *,
    n_cars: int = 20,
    n_laps: int = 57,
    fuel_start: float = 100.0,
    year: int = 2024,
    rnd: int = 1,
    circuit: str = "SimCircuit",
    compounds: list | None = None,
) -> pl.DataFrame:
    """One race. Tyre age resets at stops; fuel does not. That is the design.

    `compounds` restricts which compounds run. Pass a single one to get data a
    pooled-compound model (L0) is correctly specified for.
    """
    if n_laps < 24:
        # Stop laps are drawn from arange(12, n_laps - 8), which is empty
        # below 21 laps and leaves no room for a stint either side. Say why,
        # rather than surfacing a bare numpy error from inside rng.choice.
        raise ValueError(
            f"n_laps={n_laps} is too short to place a pit stop; need >= 24")

    rows: list[dict] = []
    compounds = list(compounds) if compounds else list(truth.deg)
    uid = f"{year}_{rnd:02d}_R"
    # Each simulated race is its own circuit, and gets a circuit-level pace
    # shift common to every car -- the thing L1's circuit intercept models.
    circuit = f"{circuit}{rnd:02d}"
    circuit_shift = float(rng.normal(0.0, 1.5))

    for car in range(n_cars):
        code = f"D{car:02d}"
        # PERSISTENT across races. Driver/car pace is a property of the driver,
        # not redrawn every weekend; redrawing it while the model fits one
        # intercept per driver is misspecification in the harness, and it
        # shows up as overconfident intervals that miss truth.
        driver_off = float(np.random.default_rng(9000 + car).normal(0.0, 0.25))

        n_stops = int(rng.integers(1, 3))
        stops = sorted(rng.choice(np.arange(12, n_laps - 8), size=n_stops, replace=False))
        bounds = [0, *stops, n_laps]
        stint_comp = list(rng.choice(compounds, size=len(bounds) - 1))

        clock = 0.0
        for sid, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
            comp = str(stint_comp[sid])
            delta = truth.deg[comp]
            for j, lap in enumerate(range(a, b), start=1):
                age = j                                   # fresh set each stint
                fuel = fuel_start - truth.kappa * lap     # continuous across stops
                y = (
                    truth.base_lap
                    + circuit_shift                       # circuit-level pace
                    + driver_off
                    + truth.beta_f * fuel                 # fuel mass effect
                    + delta * age                         # degradation
                    - truth.tau_race * lap                # track evolution
                    + rng.normal(0.0, truth.sigma)
                )
                clock += y
                rows.append(_row(
                    session_uid=uid, year=year, round=rnd, circuit=circuit,
                    session_type="R", driver=code, team=f"T{car // 2:02d}",
                    stint_id=sid, lap_in_run=j, lap_in_session=lap + 1,
                    session_clock_s=clock,
                    tyre_age=age, compound_abs=comp,
                    lap_time_s=y,
                    pit_in=bool(lap == b - 1 and b < n_laps), pit_out=bool(j == 1 and a > 0),
                    truth_fuel_kg=fuel, truth_deg_s=delta * age,
                ))

    df = pl.DataFrame(rows, schema_overrides=schema.LAPS_SCHEMA)
    return schema.validate(df, source="sim.race")


def simulate_practice(
    rng: np.random.Generator,
    truth: Truth = Truth(),
    *,
    n_cars: int = 20,
    session_minutes: float = 60.0,
    year: int = 2024,
    rnd: int = 1,
    circuit: str = "SimCircuit",
    fuel_programme: bool = True,
) -> pl.DataFrame:
    """One practice session.

    With `fuel_programme=True` (the default, and what real sessions look like)
    starting fuel trends upward through the session, so the fuel state is
    confounded with the session clock -- and therefore with track evolution.
    A practice-only estimator cannot separate them; see
    tyretrace.models.track_evolution for the qualifying anchor that can.

    Set `fuel_programme=False` to get the idealised session the original
    decoupling argument assumed, where fuel loads are random with respect to
    the clock. Useful for showing the contrast, not for anything else.
    """
    rows: list[dict] = []
    compounds = list(truth.deg)
    uid = f"{year}_{rnd:02d}_FP2"
    horizon = session_minutes * 60.0

    for car in range(n_cars):
        code = f"D{car:02d}"
        driver_off = rng.normal(0.0, 0.25)

        clock = float(rng.uniform(0, 12 * 60))     # staggered first roll-out
        sid = 0
        while clock < horizon - 200:
            comp = str(rng.choice(compounds))
            delta = truth.deg[comp]
            run_len = int(rng.integers(4, 16))

            if fuel_programme:
                # fuel rises through the session, plus genuine cross-car
                # spread at any given minute
                frac = min(clock / horizon, 1.0)
                mean_fuel = (truth.fuel_early_kg
                             + (truth.fuel_late_kg - truth.fuel_early_kg) * frac)
                fuel0 = float(rng.normal(mean_fuel, 12.0))
            else:
                fuel0 = float(rng.uniform(30, 110))
            fuel0 = max(fuel0, 10.0)

            # management mode on longer runs -- contaminates run length as a
            # fuel proxy, exactly as measured on real data
            manage = truth.manage_s_per_lap_of_run * run_len if fuel_programme else 0.0

            for j in range(1, run_len + 1):
                fuel = fuel0 - truth.kappa * (j - 1)
                if fuel < 5:
                    break
                minute = clock / 60.0
                y = (
                    truth.base_lap
                    + driver_off
                    + truth.beta_f * fuel
                    + delta * j
                    - truth.tau_practice * minute      # indexes on CLOCK
                    + manage
                    + rng.normal(0.0, truth.sigma)
                )
                clock += y
                if clock > horizon:
                    break
                rows.append(_row(
                    session_uid=uid, year=year, round=rnd, circuit=circuit,
                    session_type="FP2", driver=code, team=f"T{car // 2:02d}",
                    stint_id=sid, lap_in_run=j, lap_in_session=1,
                    session_clock_s=clock,
                    tyre_age=j, compound_abs=comp,
                    lap_time_s=y, pit_out=bool(j == 1),
                    truth_fuel_kg=fuel, truth_deg_s=delta * j,
                ))

            clock += float(rng.uniform(240, 900))   # garage time between runs
            sid += 1

    df = pl.DataFrame(rows, schema_overrides=schema.LAPS_SCHEMA)
    df = df.sort("driver", "session_clock_s").with_columns(
        (pl.int_range(pl.len()).over("driver") + 1).cast(pl.Int16).alias("lap_in_session")
    )
    return schema.validate(df, source="sim.practice")


def simulate_quali(
    rng: np.random.Generator,
    truth: Truth = Truth(),
    *,
    n_cars: int = 20,
    session_minutes: float = 60.0,
    year: int = 2024,
    rnd: int = 1,
    circuit: str = "SimCircuit",
) -> pl.DataFrame:
    """One qualifying session -- the constant-fuel probe of track evolution.

    Every car runs minimum fuel, so the fuel term is near-constant and the
    only thing moving pace across the session is track evolution. Measured on
    real data this gives tau at z ~ 10 from a single session, against z ~ 0.5
    for the practice arm. Drivers appear in successive segments, so the
    identification comes from within-driver comparisons across the session.
    """
    rows: list[dict] = []
    uid = f"{year}_{rnd:02d}_Q"
    horizon = session_minutes * 60.0

    # segment windows, as fractions of the session
    segments = [(0.02, 0.30), (0.38, 0.60), (0.72, 0.98)]

    for car in range(n_cars):
        code = f"D{car:02d}"
        driver_off = rng.normal(0.0, 0.35)
        # faster cars survive into later segments
        n_seg = 1 + int(driver_off < 0.15) + int(driver_off < -0.15)

        sid = 0
        for seg in range(min(n_seg, len(segments))):
            lo, hi = segments[seg]
            for _ in range(int(rng.integers(1, 3))):        # 1-2 push laps
                clock = float(rng.uniform(lo, hi)) * horizon
                fuel = float(rng.normal(18.0, 2.0))         # minimum fuel
                age = int(rng.integers(2, 6))
                y = (
                    truth.base_lap
                    + driver_off
                    + truth.beta_f * fuel
                    + truth.deg["C4"] * age
                    - truth.tau_practice * (clock / 60.0)
                    + rng.normal(0.0, truth.sigma * 0.5)    # push laps: tighter
                )
                rows.append(_row(
                    session_uid=uid, year=year, round=rnd, circuit=circuit,
                    session_type="Q", driver=code, team=f"T{car // 2:02d}",
                    stint_id=sid, lap_in_run=1, lap_in_session=1,
                    session_clock_s=clock,
                    tyre_age=age, compound_abs="C4", compound_rel="SOFT",
                    lap_time_s=y, truth_fuel_kg=fuel,
                    truth_deg_s=truth.deg["C4"] * age,
                ))
                sid += 1

    df = pl.DataFrame(rows, schema_overrides=schema.LAPS_SCHEMA)
    df = df.sort("driver", "session_clock_s").with_columns([
        (pl.int_range(pl.len()).over("driver") + 1).cast(pl.Int16).alias("lap_in_session"),
        (pl.int_range(pl.len()).over("driver")).cast(pl.Int32).alias("stint_id"),
    ])
    return schema.validate(df, source="sim.quali")
