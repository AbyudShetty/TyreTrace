"""Causal calibration of marginal lap-time predictive uncertainty.

Population-parameter uncertainty is not new-weekend predictive uncertainty.
Learn the missing dispersion from forecasts made BEFORE earlier races, with
one equal total weight per weekend. Keep the forecast median unchanged.

This is an empirical Student-t predictive distribution, not a posterior for
fuel mass or tyre wear, and not a distribution-free coverage guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, ndtr, ndtri, stdtr, stdtrit

MIN_WEEKENDS = 8
Z90 = float(ndtri(0.95) - ndtri(0.05))
LEVELS = (0.50, 0.80, 0.90, 0.95)
FEATURES = ("floor_s2", "raw_variance_scale", "race_horizon_s2", "tyre_age_s2")


def _features(t: pl.DataFrame) -> np.ndarray:
    """Variance in s²: intercept, raw variance, race lap², tyre age².

    Fixed scales (50 race laps, 20 tyre laps) condition the optimizer; fitted
    coefficients are predictive discrepancies, not physical wear parameters.
    No target outcome, future gap or post-race fitted parameter is a feature.
    """
    required = {"q05", "q50", "q95", "lap_in_session", "tyre_age"}
    if missing := required - set(t.columns):
        raise ValueError(f"missing prediction columns: {sorted(missing)}")
    lo, mid, hi = (t[c].to_numpy().astype(float) for c in ("q05", "q50", "q95"))
    lap, age = (t[c].to_numpy().astype(float) for c in ("lap_in_session", "tyre_age"))
    if (not all(np.all(np.isfinite(a)) for a in (lo, mid, hi, lap, age))
            or np.any(lo > mid) or np.any(mid > hi) or np.any(lo >= hi)
            or np.any(lap < 0) or np.any(age < 0)):
        raise ValueError("predictions must have finite ordered intervals and nonnegative ages")
    return np.column_stack([np.ones(t.height), ((hi - lo) / Z90) ** 2,
                            (lap / 50.0) ** 2, (age / 20.0) ** 2])


def normal_crps(mean: np.ndarray, sd: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exact CRPS of the stated Gaussian distribution, without Monte Carlo error."""
    mean, sd, y = (np.asarray(a, dtype=float) for a in (mean, sd, y))
    if mean.shape != sd.shape or mean.shape != y.shape or np.any(sd <= 0):
        raise ValueError("mean, positive sd and observations must have matching shapes")
    z = (y - mean) / sd
    return sd * (z * (2 * ndtr(z) - 1) + np.sqrt(2 / np.pi) * np.exp(-z*z/2)
                 - 1 / np.sqrt(np.pi))


def student_crps(mean: np.ndarray, scale: np.ndarray, y: np.ndarray,
                 df: float = 4.0) -> np.ndarray:
    """Exact Student-t CRPS (Jordan et al., 2019; scoringRules).

    Scale is the Student-t scale, NOT its standard deviation. Tests compare
    this expression with numerical CDF integration and Monte Carlo draws.
    """
    mean, scale, y = (np.asarray(a, dtype=float) for a in (mean, scale, y))
    if (df <= 1 or mean.shape != scale.shape or mean.shape != y.shape
            or np.any(scale <= 0)):
        raise ValueError("df > 1 and matching arrays with positive scale are required")
    z = (y - mean) / scale
    pdf = np.exp(gammaln((df+1)/2) - gammaln(df/2) - 0.5*np.log(df*np.pi)
                 - (df+1)/2*np.log1p(z*z/df))
    constant = 2*np.sqrt(df)/(df-1)*np.exp(betaln(0.5, df-0.5)-2*betaln(0.5, df/2))
    return scale * (z*(2*stdtr(df, z)-1) + 2*pdf*(df+z*z)/(df-1) - constant)


@dataclass(frozen=True)
class PredictiveUncertainty:
    coefficients: tuple[float, float, float, float]
    trained_through: str
    n_weekends: int
    n_laps: int
    df: float | None = 4.0
    method: str | None = None

    def __post_init__(self):
        values = np.asarray(self.coefficients, dtype=float)
        if values.shape != (4,) or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("four finite nonnegative scale coefficients are required")
        if self.df is not None and (not np.isfinite(self.df) or self.df <= 2):
            raise ValueError("predictive df must exceed 2 to report a finite standard deviation")

    @classmethod
    def fit(cls, history: pl.DataFrame, *, target: str,
            min_weekends: int = MIN_WEEKENDS,
            family: str = "student_t") -> "PredictiveUncertainty":
        """Fit only strictly earlier forecasts. Reject leakage rather than filter it.

        The objective is predictive log likelihood with equal total
        weight for each race; 1,000 correlated laps are not 1,000 weekends.
        Nonnegative components and an explicit numerical variance floor keep
        the variance valid when extrapolating to longer races or older tyres.
        """
        if min_weekends < MIN_WEEKENDS:
            raise ValueError(f"at least {MIN_WEEKENDS} independent weekends are required")
        if family not in ("student_t", "gaussian"):
            raise ValueError("unknown predictive family")
        df = 4.0 if family == "student_t" else None
        if not {"weekend", "y"} <= set(history.columns):
            raise ValueError("history needs weekend and observed lap times")
        keys = history["weekend"].to_numpy().astype(str)
        if len(keys) == 0:
            raise ValueError("insufficient history")
        if np.any(keys >= target):
            raise ValueError("calibration history must precede the target weekend")
        if "method" in history.columns and history["method"].n_unique() != 1:
            raise ValueError("fit each forecast method separately")
        levels, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
        if len(levels) < min_weekends:
            raise ValueError(f"insufficient history: {len(levels)} weekends; need {min_weekends}")
        x = _features(history)
        y = history["y"].to_numpy().astype(float)
        if not np.all(np.isfinite(y)):
            raise ValueError("history observations must be finite")
        residual2 = (y - history["q50"].to_numpy()) ** 2
        weight = 1.0 / (len(levels) * counts[inv])

        def objective(p):
            var = np.maximum(x @ p, 1e-10)
            if df is None:
                loss = 0.5 * np.sum(weight * (np.log(var) + residual2 / var))
                grad = 0.5 * x.T @ (weight * (1 / var - residual2 / var**2))
            else:
                loss = 0.5 * np.sum(weight * (np.log(var) + (df+1)*np.log1p(residual2/(df*var))))
                grad = 0.5 * x.T @ (weight * (1/var - (df+1)*residual2/(var*(df*var+residual2))))
            return loss, grad

        opt = minimize(objective, np.array([0.1, 1.0, 0.1, 0.1]), jac=True,
                       method="L-BFGS-B", bounds=[(1e-6, None)] * 4,
                       options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-7})
        if not opt.success or not np.all(np.isfinite(opt.x)):
            raise RuntimeError(f"uncertainty fit did not converge: {opt.message}")
        method = str(history['method'][0]) if 'method' in history.columns else None
        return cls(tuple(float(v) for v in opt.x), str(levels[-1]), len(levels), len(keys), df, method)

    def scale(self, predictions: pl.DataFrame, *, target: str) -> np.ndarray:
        if target <= self.trained_through:
            raise ValueError("prediction target must be later than the fitted history")
        if "weekend" in predictions.columns and (
                predictions["weekend"].n_unique() != 1
                or predictions["weekend"][0] != target):
            raise ValueError("predictions must contain only the target weekend")
        if self.method is not None and "method" in predictions.columns and (
                predictions["method"].n_unique() != 1 or predictions["method"][0] != self.method):
            raise ValueError("prediction method does not match calibration history")
        return np.sqrt(_features(predictions) @ np.asarray(self.coefficients))

    def predict(self, predictions: pl.DataFrame, *, target: str) -> pl.DataFrame:
        """Parameters and exact quantiles; independent of target outcomes.

        The caller may omit y entirely. Existing labels are carried through;
        scoring is a separate operation to make accidental leakage harder.
        """
        scale = self.scale(predictions, target=target)
        mean = predictions["q50"].to_numpy().astype(float)
        labels = [c for c in predictions.columns
                  if c not in {"mean", "sd", "scale", "df", "crps"} and not c.startswith("q")]
        sd = scale if self.df is None else scale * np.sqrt(self.df/(self.df-2))
        cols = {"mean": mean, "sd": sd, "scale": scale,
                "df": np.full(len(mean), self.df if self.df is not None else np.inf)}
        for q in (0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975):
            name = f"q{1000*q:03.0f}" if q in (0.025, 0.975) else f"q{100*q:02.0f}"
            quantile = ndtri(q) if self.df is None else stdtrit(self.df, q)
            cols[name] = mean + float(quantile) * scale
        return predictions.select(labels).with_columns([pl.Series(k, v) for k, v in cols.items()])

    def predict_forecast(self, forecast, *, weekend: str, method: str) -> pl.DataFrame:
        """Operational path from a fresh Forecast to exact calibrated intervals.

        The output is a table of MARGINAL lap-time distributions. It does not
        define a joint race trajectory or modify forecast.deg credible intervals.
        """
        table = from_forecast(forecast, weekend=weekend, method=method).drop('y')
        return self.predict(table, target=weekend)


def score(predictions: pl.DataFrame) -> dict:
    """Score the exact stated distribution, including the full calibration curve."""
    y, mean, sd = (predictions[c].to_numpy() for c in ("y", "mean", "scale"))
    if len(y) == 0:
        raise ValueError("cannot score an empty prediction table")
    if not all(np.all(np.isfinite(v)) for v in (y, mean, sd)) or np.any(sd <= 0):
        raise ValueError("scoring needs finite outcomes, locations and positive scales")
    df = float(predictions["df"][0])
    if predictions["df"].n_unique() != 1:
        raise ValueError("score one predictive family at a time")
    quantile = (lambda q: ndtri(q)) if np.isinf(df) else (lambda q: stdtrit(df, q))
    cal = {level: float(np.mean(np.abs(y - mean) <= quantile((1 + level)/2) * sd))
           for level in LEVELS}
    crps = normal_crps(mean, sd, y) if np.isinf(df) else student_crps(mean, sd, y, df)
    return {"crps": float(crps.mean()),
            "mae": float(np.abs(mean-y).mean()), "coverage_90": cal[0.90],
            "calibration": cal, "width90": float((2*quantile(0.95)*sd).mean()), "n": len(y)}


def rolling(predictions: pl.DataFrame, *, score_from: str = "2025",
            score_until: str | None = None,
            family: str = "student_t") -> tuple[pl.DataFrame, pl.DataFrame]:
    """Apply a separately fitted uncertainty model before each target race.

    The input contains ORIGINAL out-of-sample forecasts. Calibrated residuals
    are never fed back as if they came from the original predictor. Early races
    without enough independent history are explicitly omitted, not labelled
    calibrated. score_until permits development without loading a future block.
    """
    if "method" in predictions.columns and predictions["method"].n_unique() != 1:
        raise ValueError("rolling calibration accepts one method at a time")
    keys = sorted(predictions["weekend"].unique().to_list())
    out, fits = [], []
    for i, k in enumerate(keys):
        if i < MIN_WEEKENDS or k < score_from or (score_until and k >= score_until):
            continue
        hist = predictions.filter(pl.col("weekend") < k)
        model = PredictiveUncertainty.fit(hist, target=k, family=family)
        target = predictions.filter(pl.col("weekend") == k)
        calibrated = model.predict(target.drop('y'), target=k).with_columns(target['y'])
        scorer = normal_crps if model.df is None else student_crps
        out.append(calibrated.with_columns(pl.Series("crps", scorer(
            calibrated["mean"].to_numpy(), calibrated["scale"].to_numpy(),
            calibrated["y"].to_numpy()))))
        fits.append({"weekend": k, "trained_through": model.trained_through,
                     "n_weekends": model.n_weekends, "n_laps": model.n_laps,
                     "family": family,
                     **dict(zip(FEATURES, model.coefficients))})
    if not out:
        raise ValueError("no target weekend has enough calibration history")
    return pl.concat(out), pl.DataFrame(fits)


def from_forecast(forecast, *, weekend: str, method: str) -> pl.DataFrame:
    """Summarise a freshly generated Forecast for the same online calibration path."""
    q = np.quantile(forecast.draws, [0.05, 0.5, 0.95], axis=0)
    rename = {"lap_time_s": "y"}
    if 'stint_id' in forecast.laps.columns:
        rename['stint_id'] = 'stint'
    return forecast.laps.rename(rename).with_columns(
        pl.lit(weekend).alias("weekend"), pl.lit(method).alias("method"),
        pl.Series("q05", q[0]), pl.Series("q50", q[1]), pl.Series("q95", q[2]))
