"""Tier 3 backtest: forecast a race from its weekend's practice.

THE QUESTION THIS ANSWERS
-------------------------
Every comparison so far forecast the END of a race from its START. That only
needs the fused lap slope, so a model with the wrong fuel coefficient forecasts
almost as well (audit/b1_baselines.py) while reporting the wrong degradation
(tests/test_baselines.py). This backtest is where the split has to be right.

For weekend W the forecaster may use, and only use:

    W's practice           degradation per compound, via within-run slopes
    W's qualifying         track evolution (the anchor), and the lap time
    races BEFORE W         the fuel effect and the race/quali evolution scale
    first ANCHOR_LAPS      green race laps per driver: today's pace level
                           (a strategist sees these before deciding anything)

and forecasts every later race lap. The decisive laps are the stints after a
pit stop: their level relative to the anchor is

    -fuel_pace * (laps since anchor)  +  deg_new * age  -  deg_old * age_old

so the fuel effect enters SEPARATELY from degradation -- in both directions.
Practice gives only the fused slope (deg - fuel + track per lap), so a method
that overstates fuel overstates degradation by the same amount, and both
errors land on the post-stop stints. Protocol section 2: rolling origin,
strictly causal, no global fit.

THE METHODS (all share practice slopes, anchors, noise and scoring)
-------------------------------------------------------------------
    identified     fuel pace and race/quali scale s learned from earlier races
    fuel-only      fuel learned from earlier races, no race track term
                   (pre-registered 2026-10-21 after audit/f1_fuel_gap.py)
    no-track       identified fuel, track evolution ignored  (ablation, sec. 6)
    conventional   fuel = 0.030 s/kg x kappa, no track term   (protocol sec. 4)
    naive          no fuel correction; the practice slope IS degradation

Everything is closed-form least squares and conjugate Gaussian updating, so a
weekend takes milliseconds and the result is deterministic.

ASSUMPTIONS, STATED
-------------------
- Practice track evolution runs at the qualifying rate (s_practice = 1).
- kappa = full tank / scheduled race laps, an upper bound (README, L2).
- Degradation per compound = practice evidence updated from a prior built on
  earlier races' pit-stop-identified degradation. A compound practice never
  ran uses the prior alone -- reported per weekend, never hidden.
- Lap-time noise is the pooled within-race residual sd of earlier races.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from tyretrace.data import fastf1_pull as fp
from tyretrace.models import track_evolution as te

FULL_TANK_KG = 110.0

# Race fuel load by regulation era. 2026's power units run on a fuel-ENERGY
# flow limit and the cars start with far less fuel; ~70 kg is the widely
# reported figure. It is NOT verified here, which is why audit/h1_2026.py
# reports every 2026 result under both values instead of trusting one.
TANK_BY_ERA = {2026: 70.0}


def full_tank(year: int, override: dict[int, float] | None = None) -> float:
    table = {**TANK_BY_ERA, **(override or {})}
    return table.get(year, FULL_TANK_KG)
CONVENTIONAL_S_PER_KG = 0.030
ANCHOR_LAPS = 5
FALLBACK_SD = 0.03            # s/lap, extra sd for a compound practice never ran
N_DRAWS = 1000
MIN_EARLIER = 3               # earlier races needed to learn fuel and s

# weakly informative, as in ladder.model_l2 -- wide enough to be overruled
PRIOR_BETA_F = (0.033, 0.020)     # s/kg
PRIOR_S = (0.5, 0.5)              # race / quali track-evolution scale

METHODS = ("identified", "fuel-only", "no-track", "conventional", "naive")


def _ols(X: np.ndarray, y: np.ndarray):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    dof = max(len(y) - np.linalg.matrix_rank(X), 1)
    s2 = float(r @ r) / dof
    cov = np.linalg.pinv(X.T @ X) * s2
    return coef, cov, float(np.sqrt(s2))


def _dummies(labels: np.ndarray) -> np.ndarray:
    levels = sorted(set(labels.tolist()))
    return np.column_stack([(labels == v).astype(float) for v in levels])


# --------------------------------------------------------------------------
# per-weekend inputs
# --------------------------------------------------------------------------

@dataclass
class Weekend:
    key: str                 # "2024_06"
    tau_min: float           # quali track evolution, s/min (negative = faster)
    lap_s: float             # representative lap time, from quali push laps
    kappa: float             # kg/lap, full tank over scheduled race laps

    @property
    def track_per_lap(self) -> float:
        """Quali-rate track evolution in s per lap, signed as a lap-time
        change (negative = getting faster)."""
        return self.tau_min * self.lap_s / 60.0


WET = ("INTERMEDIATE", "WET")


def dry_weekends(race_raw: pl.DataFrame) -> set[str]:
    """Weekends whose race ran with no intermediate or wet tyre at all.

    On a wet or drying track, lap times fall for reasons that are neither
    fuel nor tyre wear -- measured on the lake: 2025 Spa (wet start, then
    drying) gave a race slope of -1.37 s/lap against a typical -0.05, and
    four such races took the cross-race scatter from 0.017 to 0.152 s/lap.
    Every model here is a dry-running model, so wet races are excluded from
    history and from scoring alike. All ten original weekends are dry, so
    this changes nothing on them.
    """
    wet = (race_raw.filter(pl.col("compound_rel").cast(pl.Utf8).is_in(WET))
                   ["session_uid"].str.slice(0, 7).unique().to_list())
    every = race_raw["session_uid"].str.slice(0, 7).unique().to_list()
    return set(every) - set(wet)


def weekend_inputs(quali: pl.DataFrame, race_raw: pl.DataFrame,
                   tank: dict[int, float] | None = None) -> dict[str, Weekend]:
    """Pre-race facts for every weekend that has a qualifying fit.

    `tank` overrides the full-tank mass by year, e.g. {2026: 110.0} to run
    2026 as if the fuel regulations had not changed."""
    out = {}
    laps = dict(race_raw.group_by("session_uid")
                        .agg(pl.col("lap_in_session").max()).iter_rows())
    push = te.push_laps(quali)
    lap_s = dict(push.group_by("session_uid").agg(pl.col("lap_time_s").median())
                     .iter_rows())
    for f in te.fit(quali):
        wk = f.session_uid.rsplit("_", 1)[0]
        n = laps.get(f"{wk}_R")
        if n:
            out[wk] = Weekend(wk, f.tau_per_min, float(lap_s[f.session_uid]),
                              full_tank(int(wk[:4]), tank) / float(n))
    return out


# --------------------------------------------------------------------------
# practice -> fused slope per compound
# --------------------------------------------------------------------------

def _run_slope(lap: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    if len(y) < 4 or np.ptp(lap) == 0:
        return None
    X = np.column_stack([np.ones(len(y)), lap])
    coef, cov, _ = _ols(X, y)
    return float(coef[1]), float(np.sqrt(max(cov[1, 1], 1e-8)))


def random_effects(est: np.ndarray, se: np.ndarray,
                   between: bool = True) -> tuple[float, float]:
    """DerSimonian-Laird pooled mean and its se.

    Runs differ for reasons beyond lap noise -- a push-then-cool run, a mode
    change, a programme that was really a quali sim -- so the between-run
    variance tau^2 is estimated and added to every run's own se^2. Treating
    laps as independent instead is what made three runs at Montreal look
    precise enough to set HARD degradation to -0.23 s/lap.
    """
    w = 1.0 / np.square(se)
    m = float(np.sum(w * est) / np.sum(w))
    if len(est) < 2:
        return m, float(np.sqrt(1.0 / np.sum(w)))
    if not between:                    # fixed-effect pooling: the ablation
        return m, float(np.sqrt(1.0 / np.sum(w)))
    q = float(np.sum(w * np.square(est - m)))
    c = float(np.sum(w) - np.sum(np.square(w)) / np.sum(w))
    tau2 = max((q - (len(est) - 1)) / c, 0.0) if c > 0 else 0.0
    w2 = 1.0 / (np.square(se) + tau2)
    return float(np.sum(w2 * est) / np.sum(w2)), float(np.sqrt(1.0 / np.sum(w2)))


def practice_slopes(prac_lr: pl.DataFrame,
                    between: bool = True) -> dict[str, tuple[float, float, int]]:
    """Fused within-run slope per compound: one slope per run, pooled.

    Only within-run variation counts. Between runs, fuel load trends with the
    session clock (A1's finding, corr +0.54), so comparisons across runs are
    confounded; within a run fuel, age and clock move in lock step and give
    one clean fused slope -- deg - fuel + track per lap.

    Returns {compound: (slope, se, n_runs)}.
    """
    out = {}
    for comp, g in prac_lr.group_by("compound_rel"):
        est, se = [], []
        for _, r in g.group_by("session_uid", "driver", "stint_id"):
            fit = _run_slope(r["lap_in_run"].to_numpy().astype(float),
                             r["lap_time_s"].to_numpy())
            if fit is not None:
                est.append(fit[0])
                se.append(fit[1])
        if est:
            m, sd = random_effects(np.array(est), np.array(se), between)
            out[str(comp[0])] = (m, sd, len(est))
    return out


# --------------------------------------------------------------------------
# earlier races -> fuel pace and the race/quali scale
# --------------------------------------------------------------------------

def race_lap_coef(race_lr: pl.DataFrame) -> tuple[float, float, float]:
    """Fused race-lap coefficient (fuel + track), its se, and residual sd.

    Driver fixed effects and per-compound degradation slopes -- NOT stint
    effects, which would absorb the pit-stop discontinuity that separates
    the lap slope from degradation in the first place.
    """
    drv = race_lr["driver"].to_numpy().astype(str)
    comp = race_lr["compound_rel"].cast(pl.Utf8).to_numpy().astype(str)
    age = race_lr["tyre_age"].to_numpy().astype(float)
    lap = race_lr["lap_in_session"].to_numpy().astype(float)
    X = np.column_stack([_dummies(drv), _dummies(comp) * age[:, None], lap])
    coef, cov, sd = _ols(X, race_lr["lap_time_s"].to_numpy())
    return float(coef[-1]), float(np.sqrt(cov[-1, -1])), sd


def race_degradation(race_lr: pl.DataFrame) -> dict[str, float]:
    """Degradation per compound as one race itself identifies it -- via the
    pit-stop age reset, with a free lap slope. Used only as a PRIOR for
    LATER weekends, never for the race it came from."""
    drv = race_lr["driver"].to_numpy().astype(str)
    comp = race_lr["compound_rel"].cast(pl.Utf8).to_numpy().astype(str)
    age = race_lr["tyre_age"].to_numpy().astype(float)
    lap = race_lr["lap_in_session"].to_numpy().astype(float)
    cs = sorted(set(comp.tolist()))
    X = np.column_stack([_dummies(drv)] + [(comp == c) * age for c in cs] + [lap])
    coef, *_ = np.linalg.lstsq(X, race_lr["lap_time_s"].to_numpy(), rcond=None)
    n = (comp[:, None] == np.array(cs)[None, :]).sum(axis=0)
    # a compound with a handful of laps gives a meaningless slope
    return {c: float(v) for c, v, k in zip(cs, coef[-len(cs) - 1:-1], n) if k >= 100}


def degradation_prior(earlier: list[dict[str, float]]) -> tuple[float, float]:
    """Mean and spread of race-identified degradation over earlier weekends,
    all compounds pooled -- relative labels are not comparable across
    weekends, so this is a prior on 'a race compound', not on SOFT."""
    v = np.array([x for d in earlier for x in d.values()])
    if len(v) < 3:
        return 0.05, 0.03
    return float(np.median(v)), max(float(np.std(v)), 0.01)


@dataclass
class FuelTrack:
    beta_f: float            # s/kg
    s: float                 # race / quali evolution scale
    cov: np.ndarray          # 2x2 posterior covariance of (beta_f, s)
    n_races: int
    slope_sd: float = 0.0     # new-weekend scatter, distinct from parameter covariance

    def lap_coef(self, wk: Weekend) -> tuple[float, float]:
        """Predicted race-lap coefficient at W and its sd:
        -beta_f * kappa + s * track_per_lap."""
        a = np.array([-wk.kappa, wk.track_per_lap])
        return float(a @ [self.beta_f, self.s]), float(np.sqrt(a @ self.cov @ a))

    def fuel_pace(self, wk: Weekend) -> tuple[float, float]:
        return self.beta_f * wk.kappa, float(np.sqrt(self.cov[0, 0])) * wk.kappa

    def predictive_lap_coef(self, wk: Weekend) -> tuple[float, float]:
        """New-race slope uncertainty, including between-weekend variation.

        lap_coef() retains its historical parameter-only semantics so original
        backtests remain reproducible. Do not confuse its SD with this one.
        """
        mean, sd = self.lap_coef(wk)
        return mean, float(np.hypot(sd, self.slope_sd))


def fit_fuel_only(coefs, ses, weekends: list[Weekend]) -> tuple[float, float]:
    """beta_f from earlier races' slopes with NO race track term:

        lap_coef_w = -beta_f * kappa_w + e_w

    Pre-registered 2026-10-21 for the lake backtest, BEFORE the new weekends
    were pulled, on the strength of audit/f1_fuel_gap.py: once between-race
    scatter is allowed, the race/quali scale s is not identified
    (0.33 +/- 0.55 on ten races) and fuel alone predicts unseen race slopes
    better. Returns (beta_f, sd) in s/kg.
    """
    y = np.asarray(coefs, dtype=float)
    se2 = np.square(ses)
    x = -np.array([w.kappa for w in weekends])
    p0 = PRIOR_BETA_F[1] ** -2
    eta2, m, v = 0.0, PRIOR_BETA_F[0], PRIOR_BETA_F[1] ** 2
    for _ in range(50):
        w = 1.0 / (se2 + eta2)
        v = 1.0 / (p0 + np.sum(w * x * x))
        m = v * (p0 * PRIOR_BETA_F[0] + np.sum(w * x * y))
        new = max(float(np.mean(np.square(y - x * m) - se2)), 0.0)
        if abs(new - eta2) < 1e-14:
            break
        eta2 = new
    return float(m), float(np.sqrt(v))


def fit_fuel_track(coefs, ses, weekends: list[Weekend]) -> FuelTrack:
    """Conjugate Gaussian regression of earlier races' lap coefficients on
    their fuel burn and track evolution:

        lap_coef_w = -beta_f * kappa_w + s * track_per_lap_w + e_w

    Each e_w carries the race's own se plus a between-race scatter eta^2,
    estimated by moments -- otherwise ten tiny standard errors would make the
    two-parameter fit absurdly confident about a relationship that holds only
    approximately across circuits.
    """
    y = np.asarray(coefs, dtype=float)
    se2 = np.square(ses)
    X = np.column_stack([[-w.kappa for w in weekends], [w.track_per_lap for w in weekends]])
    m0 = np.array([PRIOR_BETA_F[0], PRIOR_S[0]])
    P0 = np.diag([PRIOR_BETA_F[1] ** -2, PRIOR_S[1] ** -2])

    eta2 = 0.0
    for _ in range(20):                      # moments: resid^2 ~ se^2 + eta^2
        W = np.diag(1.0 / (se2 + eta2))
        cov = np.linalg.inv(P0 + X.T @ W @ X)
        m = cov @ (P0 @ m0 + X.T @ W @ y)
        r2 = np.square(y - X @ m)
        new = max(float(np.mean(r2 - se2)), 0.0)
        if abs(new - eta2) < 1e-12:
            break
        eta2 = new
    return FuelTrack(float(m[0]), float(m[1]), cov, len(y), float(np.sqrt(eta2)))


# --------------------------------------------------------------------------
# the forecast
# --------------------------------------------------------------------------

@dataclass
class Forecast:
    laps: pl.DataFrame          # target laps with labels
    draws: np.ndarray           # (N_DRAWS, n_target)
    deg: dict                   # compound -> (mean, sd, source)
    lap_coef: tuple[float, float]


def degradation(slopes: dict, fuel_pace: tuple[float, float],
                track_per_lap: float, compounds,
                prior: tuple[float, float] | None = None) -> dict:
    """deg_c = practice slope + fuel pace - practice track per lap, updated
    from a prior built on earlier races.

    The update is a precision-weighted average, so practice moves the answer
    exactly as far as it is informative: a compound with 20 consistent runs
    lands near its practice value, one with three erratic runs stays near the
    prior, and one practice never ran IS the prior. Without a prior (v1) a
    missing compound borrowed the mean of the others, which at Monza spread
    a +0.41 s/lap soft outlier onto the hard tyre.
    """
    f, f_sd = fuel_pace
    have = {c: (s + f - track_per_lap, float(np.hypot(se, f_sd)))
            for c, (s, se, _) in slopes.items()}
    out = {}
    for c in compounds:
        if prior is not None:
            pm, ps = prior
            if c in have:
                m, sd = have[c]
                w0, w1 = ps ** -2, sd ** -2
                out[c] = ((w0 * pm + w1 * m) / (w0 + w1), (w0 + w1) ** -0.5,
                          "practice+prior")
            else:
                out[c] = (pm, ps, "prior")
        elif c in have:
            out[c] = (*have[c], "practice")
        elif have:
            m = float(np.mean([v[0] for v in have.values()]))
            sd = float(np.hypot(np.mean([v[1] for v in have.values()]), FALLBACK_SD))
            out[c] = (m, sd, "fallback")
        else:
            out[c] = (0.05, 0.05, "prior")
    return out


def forecast(race_lr: pl.DataFrame, deg: dict, lap_coef: tuple[float, float],
             sigma: float, seed: int = 0,
             traffic: tuple[float, float] | None = None) -> Forecast | None:
    """Anchor each driver on their first ANCHOR_LAPS green laps, forecast the rest.

    Parameter uncertainty is propagated by drawing the lap coefficient and
    every compound's degradation jointly per draw, re-solving each driver's
    anchor level under that draw, then adding lap noise.

    `traffic = (wake, expected_dirty_air)`, both learned from EARLIER races:
    the anchor laps' own dirty air is removed from the pace level, and the
    expected dirty air of a later lap is added back. The post-race
    decomposition (audit/p1_postrace.py) found the early-race anchor laps
    sit in traffic, leaving every forecast +0.11 s too slow for every method.
    Future gaps are unknown before the race, so the forecast carries their
    expectation, not the realised value. Pre-registered 2026-10-21, before
    the lake was pulled.
    """
    rng = np.random.default_rng(seed)
    df = race_lr.sort("driver", "lap_in_session")
    df = df.with_columns(pl.int_range(pl.len()).over("driver").alias("_k"))
    anchor = df.filter(pl.col("_k") < ANCHOR_LAPS)
    target = df.filter(pl.col("_k") >= ANCHOR_LAPS)
    target = target.join(
        anchor.group_by("driver").agg(pl.col("stint_id").first().alias("_anchor_stint")),
        on="driver")
    if target.height == 0:
        return None

    comps = sorted(deg)
    b = lap_coef[0] + lap_coef[1] * rng.standard_normal(N_DRAWS)          # (D,)
    dmat = np.column_stack([deg[c][0] + deg[c][1] * rng.standard_normal(N_DRAWS)
                            for c in comps])                               # (D, C)
    cidx = {c: i for i, c in enumerate(comps)}

    def mean_part(frame):
        lap = frame["lap_in_session"].to_numpy().astype(float)
        age = frame["tyre_age"].to_numpy().astype(float)
        ci = np.array([cidx[str(c)] for c in frame["compound_rel"].cast(pl.Utf8)])
        return b[:, None] * lap[None, :] + dmat[:, ci] * age[None, :]      # (D, n)

    a_part = mean_part(anchor)
    a_resid = anchor["lap_time_s"].to_numpy()[None, :] - a_part
    if traffic is not None and "dirty_air" in anchor.columns:
        a_resid = a_resid - traffic[0] * anchor["dirty_air"].fill_null(0.0).to_numpy()[None, :]
    a_drv = anchor["driver"].to_numpy()
    t_drv = target["driver"].to_numpy()
    level = np.empty((N_DRAWS, target.height))
    for d in np.unique(t_drv):
        lv = a_resid[:, a_drv == d].mean(axis=1)
        lv = lv + sigma / np.sqrt(ANCHOR_LAPS) * rng.standard_normal(N_DRAWS)
        level[:, t_drv == d] = lv[:, None]

    draws = level + mean_part(target) + sigma * rng.standard_normal((N_DRAWS, target.height))
    if traffic is not None and "dirty_air" in anchor.columns:
        draws = draws + traffic[0] * traffic[1]
    labels = target.select(
        "session_uid", "driver", "stint_id", "lap_in_session", "tyre_age",
        pl.col("compound_rel").cast(pl.Utf8).alias("compound"), "lap_time_s",
        (pl.col("stint_id") != pl.col("_anchor_stint")).alias("after_stop"))
    return Forecast(labels, draws, deg, lap_coef)


# --------------------------------------------------------------------------
# one weekend, every method
# --------------------------------------------------------------------------

def run_weekend(wk: Weekend, prac_lr: pl.DataFrame, race_lr: pl.DataFrame,
                ft: FuelTrack, sigma: float, seed: int = 0,
                prior: tuple[float, float] | None = None,
                fuel_only: tuple[float, float] | None = None,
                between: bool = True,
                traffic: tuple[float, float] | None = None,
                methods: tuple[str, ...] | None = None) -> dict[str, Forecast]:
    """All methods on one weekend, from identical inputs -- including the
    same degradation prior, so the only thing that differs is fuel/track."""
    slopes = practice_slopes(prac_lr, between)
    comps = sorted(race_lr["compound_rel"].cast(pl.Utf8).unique().to_list())
    f_id = ft.fuel_pace(wk)
    f_conv = (CONVENTIONAL_S_PER_KG * wk.kappa, 0.0)
    lc_id = ft.lap_coef(wk)

    spec = {
        # (fuel pace for degradation, practice track term, race lap coef)
        "identified": (f_id, wk.track_per_lap, lc_id),
        "no-track": (f_id, 0.0, (-f_id[0], f_id[1])),
        **({"fuel-only": ((fuel_only[0] * wk.kappa, fuel_only[1] * wk.kappa),
                          wk.track_per_lap,
                          (-fuel_only[0] * wk.kappa, fuel_only[1] * wk.kappa))}
           if fuel_only else {}),
        "conventional": (f_conv, 0.0, (-f_conv[0], 0.0)),
        "naive": ((0.0, 0.0), 0.0, (0.0, 0.0)),
    }
    out = {}
    for name, (fuel, track, lc) in spec.items():
        if methods is not None and name not in methods:
            continue
        deg = degradation(slopes, fuel, track, comps, prior)
        fc = forecast(race_lr, deg, lc, sigma, seed=seed, traffic=traffic)
        if fc is not None:
            out[name] = fc
    return out


def prepare(practice: pl.DataFrame, race_raw: pl.DataFrame, trim: bool = True):
    """Long-run filtered laps, keyed by weekend. `trim=False` keeps the slow
    laps the outlier trim would drop -- the lap-filtering ablation.

    Race laps carry `dirty_air`, computed on ALL laps before filtering: the
    running order depends on every car on track (as in the ladder)."""
    from tyretrace.features import traffic as tr

    prac = fp.long_runs(practice, min_laps=4, trim=trim).with_columns(
        pl.col("session_uid").str.slice(0, 7).alias("weekend"))
    race_raw = tr.dirty_air_index(tr.add_gap_ahead(race_raw))
    race = fp.long_runs(race_raw, min_laps=4, trim=trim).with_columns(
        pl.col("session_uid").str.slice(0, 7).alias("weekend"))
    return prac, race


# --------------------------------------------------------------------------
# rolling recalibration
# --------------------------------------------------------------------------

MIN_CAL_LAPS = 500      # history needed before a recalibration is trusted
Z90 = 3.2897            # width of a central 90% interval in standard units


def standardised(fc: Forecast) -> np.ndarray:
    """Each lap's outcome in units of its own forecast spread:
    z = (actual - median) / (90% half-width / 1.645)."""
    y = fc.laps["lap_time_s"].to_numpy()
    q05, q50, q95 = np.quantile(fc.draws, [0.05, 0.5, 0.95], axis=0)
    scale = np.maximum((q95 - q05) / Z90, 1e-6)
    return (y - q50) / scale


def recalibrate(fc: Forecast, z_hist: np.ndarray, seed: int = 0) -> Forecast:
    """Replace each lap's predictive shape with the EMPIRICAL distribution of
    standardised errors from earlier weekends, keeping the lap's own median
    and spread.

    This is quantile recalibration (Kuleshov et al., 2018) done strictly
    causally: z_hist comes only from races before this one. Because it is
    empirical, it learns asymmetry too -- the lopsided, slow-side misses a
    symmetric noise model cannot represent (audit/b1_baselines.py, L6).
    """
    rng = np.random.default_rng(seed)
    q05, q50, q95 = np.quantile(fc.draws, [0.05, 0.5, 0.95], axis=0)
    scale = np.maximum((q95 - q05) / Z90, 1e-6)
    z = rng.choice(z_hist, size=fc.draws.shape[0], replace=True)
    return Forecast(fc.laps, q50[None, :] + scale[None, :] * z[:, None], fc.deg, fc.lap_coef)


def traffic_history(race: pl.DataFrame, keys) -> dict[str, tuple[float, float]]:
    """Per race: the wake cost (s per unit dirty air) the race itself shows,
    and the mean dirty air of laps after each driver's anchor. Used only as
    history for LATER weekends."""
    from tyretrace.backtest.postrace import post_race_fit

    out = {}
    for k in keys:
        r = race.filter(pl.col("weekend") == k)
        if r.height == 0 or "dirty_air" not in r.columns:
            continue
        wake = post_race_fit(r)["wake"]
        later = (r.sort("driver", "lap_in_session")
                  .with_columns(pl.int_range(pl.len()).over("driver").alias("_k"))
                  .filter(pl.col("_k") >= ANCHOR_LAPS))
        out[k] = (wake, float(later["dirty_air"].fill_null(0.0).mean()))
    return out


def traffic_prior(earlier: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Median wake and mean later-lap dirty air over earlier races."""
    if not earlier:
        return None
    return (float(np.median([e[0] for e in earlier])), float(np.mean([e[1] for e in earlier])))


def rolling(practice: pl.DataFrame, quali: pl.DataFrame, race_raw: pl.DataFrame, *,
            prior: bool = True, between: bool = True, trim: bool = True,
            methods=("identified", "fuel-only", "conventional", "naive"),
            score_from: str | None = None, tank: dict[int, float] | None = None,
            traffic: bool = False, recal: bool = False) -> dict:
    """The rolling-origin backtest as a function, with each component
    switchable -- what the ablation table (audit/a6_ablations.py) needs.

    Same procedure as audit/t3_backtest.py: for each weekend with
    MIN_EARLIER races before it, learn fuel/track from those races, the
    degradation prior from their pit stops, degradation per compound from the
    weekend's practice, and forecast every race lap after each driver's
    anchor. Returns {method: pooled metrics over every evaluated lap}.
    """
    from tyretrace.eval import metrics

    prac, race = prepare(practice, race_raw, trim=trim)
    weekends = weekend_inputs(quali, race_raw, tank)
    dry = dry_weekends(race_raw)
    keys = sorted(k for k in weekends if k in dry)
    rc = {k: race_lap_coef(race.filter(pl.col("weekend") == k)) for k in keys}
    rd = {k: race_degradation(race.filter(pl.col("weekend") == k)) for k in keys}
    tf = traffic_history(race, keys) if traffic else {}
    acc: dict[str, list] = {m: [] for m in methods}
    # standardised errors of every EARLIER evaluated weekend, per method
    z_hist: dict[str, list] = {m: [] for m in methods}
    for i, k in enumerate(keys):
        # history always uses every earlier weekend; score_from only limits
        # which weekends are SCORED (e.g. 2026 alone for the headline)
        if i < MIN_EARLIER or (score_from and k < score_from):
            continue
        earlier = keys[:i]
        ft = fit_fuel_track([rc[e][0] for e in earlier], [rc[e][1] for e in earlier],
                            [weekends[e] for e in earlier])
        fo = fit_fuel_only([rc[e][0] for e in earlier], [rc[e][1] for e in earlier],
                           [weekends[e] for e in earlier])
        sigma = float(np.median([rc[e][2] for e in earlier]))
        pr = degradation_prior([rd[e] for e in earlier]) if prior else None
        wk = weekends[k]
        p_w = prac.filter(pl.col("weekend") == k)
        fcs = run_weekend(wk, p_w, race.filter(pl.col("weekend") == k), ft, sigma,
                          seed=i, prior=pr, fuel_only=fo, between=between,
                          traffic=traffic_prior([tf[e] for e in earlier]) if traffic else None)
        for m in methods:
            if m not in fcs:
                continue
            fc = fcs[m]
            raw_z = standardised(fc)
            if recal:
                hist = np.concatenate(z_hist[m]) if z_hist[m] else np.array([])
                if len(hist) >= MIN_CAL_LAPS:
                    fc = recalibrate(fc, hist, seed=i)
            acc[m].append(fc)
            z_hist[m].append(raw_z)              # becomes history for later weekends
    out = {}
    for m, fcs in acc.items():
        if not fcs:
            continue
        y = np.concatenate([f.laps["lap_time_s"].to_numpy() for f in fcs])
        aft = np.concatenate([f.laps["after_stop"].to_numpy() for f in fcs])
        crps = np.concatenate([metrics.crps_ensemble(f.draws, f.laps["lap_time_s"].to_numpy())
                               for f in fcs])
        med = np.concatenate([np.median(f.draws, axis=0) for f in fcs])
        lo = np.concatenate([np.quantile(f.draws, 0.05, axis=0) for f in fcs])
        hi = np.concatenate([np.quantile(f.draws, 0.95, axis=0) for f in fcs])
        out[m] = {"crps": float(crps.mean()), "crps_after": float(crps[aft].mean()),
                  "mae": float(np.abs(med - y).mean()),
                  "bias_after": float((med - y)[aft].mean()),
                  "cov90": float(((y >= lo) & (y <= hi)).mean()),
                  "n": int(len(y)), "n_after": int(aft.sum()), "weekends": len(fcs)}
    return out
