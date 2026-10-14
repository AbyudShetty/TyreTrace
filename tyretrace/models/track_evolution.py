"""Track evolution from qualifying -- the constant-fuel probe.

WHY THIS EXISTS
---------------
The original plan identified track evolution inside practice, on the argument
that cars carry different fuel loads at the same session minute. Measured on
ten weekends of real data, that argument does not hold:

    corr(run start minute, run pace) = +0.54   [IQR 0.34, 0.78]

Teams run low-fuel quali sims early and high-fuel race sims late, so fuel
state TRENDS with the session clock -- the exact index track evolution rides
on. A practice-only fit returns tau = +0.029 s/min, the wrong sign: the cars
are getting heavier faster than the track is rubbering in.

Run length looked like a way out (a 15-lap run must be fuelled for 15 laps),
but it is contaminated: the measured coefficient is +0.22 s per lap of run
length against +0.029 predicted by fuel alone, because long runs are also
driven in management mode. It proxies programme type, not fuel.

Qualifying does work. Every car runs minimum fuel, so the fuel term is
near-constant across a session spanning ~60 minutes, and within-driver
comparisons across segments isolate the track. Measured:

    tau = -0.0165 s/min   negative in 10/10 sessions   single-session z = 10.2
    robust to tyre age, compound and fresh-tyre controls (-0.0161 to -0.0172)

against the practice arm's z = 0.54 with the wrong sign.

THE ASSUMPTION THIS INTRODUCES
------------------------------
Track evolution measured in qualifying must transport to the other sessions of
the weekend. The rate is not identical -- rubber laid down depends on how much
running has happened -- so this is a real assumption and it is stated, not
hidden. It is weaker than assuming a fuel coefficient outright, and unlike
that assumption it is checkable against the game rig.

WHERE IT SITS IN THE CHAIN
--------------------------
    qualifying  ->  tau, track evolution                    (here)
    race        ->  degradation, and with tau, fuel pace    (A2: passes)
    practice    ->  consumes both; no longer has to identify anything
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

PUSH_LAP_TOLERANCE = 1.02      # within 2% of the driver's session best


@dataclass
class TrackEvolution:
    session_uid: str
    tau_per_min: float          # negative = track getting faster
    se: float
    span_min: float
    n_laps: int
    n_drivers: int

    @property
    def total_s(self) -> float:
        """Pace gained over the whole session, seconds."""
        return self.tau_per_min * self.span_min

    def tau_model(self, lap_time_s: float) -> float:
        """Track evolution in s/lap, in the MODEL sign convention.

        Two conventions exist in this codebase and they are opposite. This
        method is the only sanctioned bridge between them; converting by hand
        is what produced a sign-flipped sensitivity table once already.

            measured here   tau_per_min < 0  means the track is IMPROVING
                            (it is a regression coefficient on the clock, so
                            falling lap times give a negative slope)

            model / sim     enters as  ``- tau * lap``  so tau > 0 means the
                            track is IMPROVING

        So the bridge is a negation, plus s/min -> s/lap via the lap time.
        """
        return -self.tau_per_min * lap_time_s / 60.0

    @property
    def z(self) -> float:
        return abs(self.tau_per_min) / self.se if self.se > 0 else 0.0


def push_laps(df: pl.DataFrame, tol: float = PUSH_LAP_TOLERANCE) -> pl.DataFrame:
    """Green, non-pit, timed laps within `tol` of each driver's session best.

    Qualifying is mostly push laps and cool-down laps; this keeps the former.
    """
    out = df.filter(
        (pl.col("track_status") == "1")
        & ~pl.col("pit_in") & ~pl.col("pit_out")
        & pl.col("lap_time_s").is_finite()
        & pl.col("session_clock_s").is_finite()
    )
    if out.height == 0:
        return out
    return (out.with_columns(
                pl.col("lap_time_s").min().over(["session_uid", "driver"]).alias("_b"))
               .filter(pl.col("lap_time_s") <= tol * pl.col("_b"))
               .drop("_b"))


def fit_session(g: pl.DataFrame, *, controls: bool = True,
                min_laps: int = 40) -> TrackEvolution | None:
    """Within-driver pace against session clock, at (near) constant fuel.

    Driver dummies absorb pace and car; the clock coefficient is what is left.
    With `controls`, tyre age / compound / fresh-tyre are partialled out too --
    on real data this moves the estimate by under 5%, which is the check that
    it is track evolution and not Q3 running fresher rubber than Q1.
    """
    if g.height < min_laps:
        return None

    drv = g["driver"].to_numpy().astype(str)
    t = g["session_clock_s"].to_numpy() / 60.0
    y = g["lap_time_s"].to_numpy()
    levels = sorted(set(drv))
    if len(levels) < 4 or t.std() == 0:
        return None

    cols = [np.column_stack([(drv == k).astype(float) for k in levels]), t[:, None]]
    tau_col = len(levels)          # position of the clock coefficient

    if controls:
        age = g["tyre_age"].cast(pl.Float64).fill_null(0.0).to_numpy()
        cols.append(age[:, None])
        comp = g["compound_rel"].cast(pl.Utf8).fill_null("UNKNOWN").to_numpy().astype(str)
        present = [c for c in ("SOFT", "MEDIUM", "HARD") if (comp == c).sum() >= 3]
        if len(present) > 1:
            cols.append(np.column_stack(
                [(comp == c).astype(float) for c in present[1:]]))
        fresh = g["is_fresh"].cast(pl.Boolean).fill_null(False).to_numpy().astype(float)
        if 0 < fresh.mean() < 1:
            cols.append(fresh[:, None])

    X = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(len(y) - np.linalg.matrix_rank(X), 1)
    s2 = float(resid @ resid) / dof
    se = float(np.sqrt(max(np.linalg.pinv(X.T @ X)[tau_col, tau_col] * s2, 0.0)))

    uid = str(g["session_uid"][0])
    return TrackEvolution(uid, float(coef[tau_col]), se,
                          float(t.max() - t.min()), g.height, len(levels))


def fit(quali: pl.DataFrame, *, controls: bool = True) -> list[TrackEvolution]:
    """One estimate per qualifying session."""
    out = []
    for _, g in push_laps(quali).group_by("session_uid", maintain_order=True):
        r = fit_session(g, controls=controls)
        if r is not None:
            out.append(r)
    return out


def summary_tau_model(fits: list[TrackEvolution], lap_time_s: float) -> tuple[float, float]:
    """Median tau and SE in the MODEL convention, s/lap. See tau_model()."""
    tau = float(np.median([f.tau_model(lap_time_s) for f in fits]))
    se = float(np.median([f.se for f in fits])) * lap_time_s / 60.0
    return tau, se


def summary(fits: list[TrackEvolution]) -> dict:
    if not fits:
        return {"n": 0}
    tau = np.array([f.tau_per_min for f in fits])
    se = np.array([f.se for f in fits])
    return {
        "n": len(fits),
        "median_tau": float(np.median(tau)),
        "median_se": float(np.median(se)),
        "frac_negative": float(np.mean(tau < 0)),
        "median_z": float(np.median(np.abs(tau) / np.where(se > 0, se, np.nan))),
        "median_total_s": float(np.median([f.total_s for f in fits])),
    }
