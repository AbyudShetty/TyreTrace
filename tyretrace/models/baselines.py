"""Baselines, as pre-registered in eval/protocol.md section 4.

"All four run on every comparison. Beating only the first proves nothing."

Every baseline here takes the SAME train/test Designs the ladder uses and
returns predictive draws of shape (n_draws, n_test), so it is scored by the
same metrics.score() on the same held-out laps. A baseline that got its own
split, its own lap filter or its own scoring rule would make the comparison
meaningless without anyone noticing.

  B1  naive          within-stint OLS of lap time on lap-in-run
  B2  conventional   subtract 0.030 s/kg x fuel, quadratic in tyre age
  B3  ML             gradient boosting, same inputs, no causal structure
  B4  published      race-simulation lap model after Heilmeier et al. (TUM):
                     logarithmic tyre degradation + fuel mass x sensitivity

Each is implemented the way a competent practitioner would actually run it,
not as a straw man. In particular their predictive spread is the residual
spread they observe in training, which is what anyone using them would quote.
"""
from __future__ import annotations

import warnings

import numpy as np

# joblib probes the physical core count through a Windows tool that is often
# missing and warns about it. Its suggested fix (LOKY_MAX_CPU_COUNT) is only
# honoured below the logical core count, i.e. by giving up a core -- so
# silence exactly this message instead.
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

from tyretrace.models.ladder import Design

N_DRAWS = 1000
FULL_TANK_KG = 110.0
CONVENTIONAL_S_PER_KG = 0.030      # the paddock rule of thumb, protocol section 4
MIN_SD = 0.05                      # floor: a 3-lap stint can fit a line perfectly


def _gaussian_draws(mean: np.ndarray, sd: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return mean[None, :] + np.maximum(sd, MIN_SD)[None, :] * rng.standard_normal(
        (N_DRAWS, len(mean)))


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Min-norm least squares and residual SD. Min-norm, so dummy columns
    for levels absent from training get a zero coefficient rather than an
    arbitrary one."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    dof = max(len(y) - np.linalg.matrix_rank(X), 1)
    return coef, float(np.sqrt(r @ r / dof))


def _onehot(codes: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((len(codes), n))
    out[np.arange(len(codes)), codes] = 1.0
    return out


# --------------------------------------------------------------------------
# B1 -- naive
# --------------------------------------------------------------------------

def naive(tr: Design, te: Design, seed: int = 0) -> np.ndarray:
    """Fit a line to each stint, extrapolate it.

    This is the slope every strategist quotes -- fuel and wear fused, no
    pooling, no structure. A held-out lap whose stint has fewer than three
    training laps (a new set fitted after the training cut) has no line to
    extrapolate, so it falls back to that driver's training mean at the same
    weekend with that driver's spread: the naive model genuinely knows
    nothing about a set it has not seen.
    """
    mean = np.empty(te.n)
    sd = np.empty(te.n)
    fits: dict[int, tuple[float, float, float]] = {}
    for s in np.unique(tr.stint):
        m = tr.stint == s
        if m.sum() >= 3 and np.ptp(tr.age[m]) > 0:
            X = np.column_stack([np.ones(m.sum()), tr.age[m]])
            coef, rsd = _ols(X, tr.y[m])
            fits[int(s)] = (coef[0], coef[1], rsd)

    for i in range(te.n):
        f = fits.get(int(te.stint[i]))
        if f is not None:
            mean[i] = f[0] + f[1] * te.age[i]
            sd[i] = f[2]
        else:
            m = (tr.driver == te.driver[i]) & (tr.weekend == te.weekend[i])
            if not m.any():
                m = tr.weekend == te.weekend[i]
            mean[i] = tr.y[m].mean()
            sd[i] = tr.y[m].std()
    return _gaussian_draws(mean, sd, seed)


# --------------------------------------------------------------------------
# B2 -- conventional fuel correction
# --------------------------------------------------------------------------

def _fuel_kg(d: Design) -> np.ndarray:
    """Fuel on board at the start of each lap: full tank minus kappa per lap."""
    lap_in_session = d.lap + d.lap_mean
    return np.maximum(FULL_TANK_KG - d.kappa * (lap_in_session - 1.0), 0.0)


def _conventional_X(d: Design) -> np.ndarray:
    """Driver-at-weekend level, plus a quadratic in tyre age per
    weekend x compound -- the 'fit a curve to the residual' step."""
    n_dw = d.n_driver * d.n_weekend
    dw = _onehot(d.weekend * d.n_driver + d.driver, n_dw)
    wc = _onehot(d.weekend * d.n_compound + d.compound, d.n_weekend * d.n_compound)
    return np.column_stack([dw, wc * d.age[:, None], wc * (d.age ** 2)[:, None]])


def conventional_fit(tr: Design, s_per_kg: float = CONVENTIONAL_S_PER_KG
                     ) -> tuple[np.ndarray, float]:
    """Coefficients and residual SD of the conventional fit."""
    return _ols(_conventional_X(tr), tr.y - s_per_kg * _fuel_kg(tr))


def conventional_degradation(tr: Design, s_per_kg: float = CONVENTIONAL_S_PER_KG
                             ) -> np.ndarray:
    """The linear age coefficient per weekend x compound: the 'degradation'
    a practitioner using this method would report, in s/lap."""
    coef, _ = conventional_fit(tr, s_per_kg)
    n_dw, n_wc = tr.n_driver * tr.n_weekend, tr.n_weekend * tr.n_compound
    return coef[n_dw:n_dw + n_wc]


def conventional(tr: Design, te: Design, seed: int = 0,
                 s_per_kg: float = CONVENTIONAL_S_PER_KG) -> np.ndarray:
    """Subtract an ASSUMED fuel effect, call the remainder degradation.

    The fuel coefficient is imported, not estimated -- which is the practice
    this project exists to question. Everything after the subtraction is an
    honest least-squares fit.
    """
    coef, rsd = conventional_fit(tr, s_per_kg)
    mean = _conventional_X(te) @ coef + s_per_kg * _fuel_kg(te)
    return _gaussian_draws(mean, np.full(te.n, rsd), seed)


# --------------------------------------------------------------------------
# B3 -- gradient boosting
# --------------------------------------------------------------------------

# Quantile levels at the midpoints of 39 equal-probability bins. Predicting a
# quantile at each and treating the set as an equally weighted ensemble is a
# consistent discretisation of the predictive distribution, so it can be
# scored by the same ensemble CRPS as everything else.
GBM_LEVELS = (np.arange(39) + 0.5) / 39


def _gbm_features(d: Design) -> np.ndarray:
    return np.column_stack([
        d.weekend, d.driver, d.compound,                  # categorical
        d.lap + d.lap_mean, d.age, d.dirty_air,           # numeric
        d.kappa, d.tau_quali,
    ])


def gbm(tr: Design, te: Design, seed: int = 0) -> np.ndarray:
    """Quantile gradient boosting on the same inputs the ladder sees.

    No causal structure: it is free to learn any function of lap, age and the
    rest. What it cannot do is extrapolate -- trees are constant beyond the
    training range, and the held-out block is the END of each race. That is
    the protocol's point: structure is what lets a model predict the part of
    the race it has not seen.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    Xtr, Xte = _gbm_features(tr), _gbm_features(te)
    cat = [True, True, True, False, False, False, False, False]
    qs = np.empty((len(GBM_LEVELS), te.n))
    for k, q in enumerate(GBM_LEVELS):
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=float(q), max_iter=300, learning_rate=0.05,
            max_leaf_nodes=31, min_samples_leaf=20, categorical_features=cat,
            random_state=seed)
        m.fit(Xtr, tr.y)
        qs[k] = m.predict(Xte)
    # independent quantile fits can cross; sorting restores a valid quantile
    # function without moving any level's prediction far
    return np.sort(qs, axis=0)


# --------------------------------------------------------------------------
# B4 -- published: race-simulation lap-time model
# --------------------------------------------------------------------------

# k2 grid for the logarithmic curve; with k2 fixed the model is linear, so
# each candidate is one least-squares fit and the best by residual wins.
LOG_K2 = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5)


def published(tr: Design, te: Design, seed: int = 0,
              s_per_kg: float = CONVENTIONAL_S_PER_KG) -> np.ndarray:
    """Lap-time model of the TUM race simulation family.

    Reimplemented after A. Heilmeier, M. Graf, M. Lienkamp, "A Race
    Simulation for Strategy Decisions in Circuit Motorsports" (IEEE ITSC
    2018) and the open-source TUMFTM race-simulation code: lap time is a base
    pace, plus fuel mass times a fixed time-per-kg sensitivity, plus a tyre
    term per compound. Of the tyre models that work offers, the logarithmic
    one is used here,

        t_tyre(age) = k1 * ln(k2 * age + 1)

    because it differs most from the conventional quadratic (B2): it is
    concave, fast early and flattening, where B2 can curve either way.
    VERIFY the exact equations against the paper before citing them at
    judging -- this follows the model family, not a line-by-line port.

    k2 is shared within a weekend (grid search); k1 is fitted per weekend x
    compound by least squares; levels per driver at each weekend.
    """
    resid_tr = tr.y - s_per_kg * _fuel_kg(tr)
    n_dw = tr.n_driver * tr.n_weekend
    n_wc = tr.n_weekend * tr.n_compound

    def X(d: Design, k2: float) -> np.ndarray:
        dw = _onehot(d.weekend * d.n_driver + d.driver, n_dw)
        wc = _onehot(d.weekend * d.n_compound + d.compound, n_wc)
        return np.column_stack([dw, wc * np.log(k2 * d.age + 1.0)[:, None]])

    best = None
    for k2 in LOG_K2:
        coef, rsd = _ols(X(tr, k2), resid_tr)
        if best is None or rsd < best[2]:
            best = (k2, coef, rsd)
    k2, coef, rsd = best
    mean = X(te, k2) @ coef + s_per_kg * _fuel_kg(te)
    return _gaussian_draws(mean, np.full(te.n, rsd), seed)


BASELINES = {
    "B1 naive": naive,
    "B2 conventional": conventional,
    "B3 gradient boosting": gbm,
    "B4 published": published,
}
