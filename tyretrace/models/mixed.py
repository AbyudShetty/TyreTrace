"""Exact mixed model with one random intercept per run. No JAX, no PyMC.

Why this exists: in practice sessions the starting fuel load is unknown and
varies a lot between runs (30-110 kg is normal). At ~0.033 s/kg that is a
+/- 1.3 s spread sitting on every run's intercept.

  - Ignore it, and it contaminates every slope.
  - Absorb it with run FIXED effects, and you kill the between-run variation
    that track evolution is identified from -- because within a run the
    session clock is nearly collinear with lap-in-run.

Partial pooling is the only thing that threads that. This module is the
smallest correct version, and it doubles as the closed-form check that the
NumPyro L2 model must reproduce.

V is block-diagonal by run, so the inverse and determinant have closed forms
(Sherman-Morrison) and the whole fit is a few hundred microseconds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class MixedFit:
    beta: np.ndarray        # fixed effects
    se: np.ndarray          # their standard errors
    sigma_u: float          # between-run SD (mostly unknown fuel load)
    sigma_e: float          # per-lap residual SD
    loglik: float
    names: list[str]

    def get(self, name: str) -> tuple[float, float]:
        i = self.names.index(name)
        return float(self.beta[i]), float(self.se[i])

    def report(self, name: str, truth: float | None = None, z: float = 1.645) -> str:
        m, s = self.get(name)
        out = f"{name:<16s} {m:+.5f} +/- {s:.5f}"
        if truth is not None:
            lo, hi = m - z * s, m + z * s
            out += (f"   90% [{lo:+.5f},{hi:+.5f}]   true {truth:+.5f}"
                    f"   {'OK' if lo <= truth <= hi else 'MISS'}")
        return out


def _blocks(group: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(group, kind="stable")
    _, starts = np.unique(group[order], return_index=True)
    return np.split(order, starts[1:])


def _gls(X, y, idx, s2u, s2e):
    """GLS given variance components. Returns (beta, XtViX, loglik)."""
    p = X.shape[1]
    XtViX = np.zeros((p, p))
    XtViy = np.zeros(p)
    logdet = 0.0
    quad = 0.0
    # accumulate per block using V_r^-1 = (1/s2e)(I - c J), c = s2u/(s2e + n s2u)
    for b in idx:
        Xb, yb, n = X[b], y[b], len(b)
        c = s2u / (s2e + n * s2u)
        Xs, ys = Xb.sum(0), yb.sum()
        XtViX += (Xb.T @ Xb - c * np.outer(Xs, Xs)) / s2e
        XtViy += (Xb.T @ yb - c * Xs * ys) / s2e
        logdet += (n - 1) * np.log(s2e) + np.log(s2e + n * s2u)
        quad += (yb @ yb - c * ys * ys) / s2e

    beta = np.linalg.solve(XtViX, XtViy)
    # residual quadratic form via the same block structure
    r = y - X @ beta
    quad_r = 0.0
    for b in idx:
        rb, n = r[b], len(b)
        c = s2u / (s2e + n * s2u)
        rs = rb.sum()
        quad_r += (rb @ rb - c * rs * rs) / s2e
    ll = -0.5 * (logdet + quad_r + len(y) * np.log(2 * np.pi))
    return beta, XtViX, ll


def fit(X: np.ndarray, y: np.ndarray, group: np.ndarray,
        names: list[str]) -> MixedFit:
    """Fit y = X beta + Z u + e with u ~ N(0, sigma_u^2) by run."""
    idx = _blocks(np.asarray(group))
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)

    def nll(theta):
        s2u, s2e = np.exp(theta)
        try:
            return -_gls(X, y, idx, s2u, s2e)[2]
        except np.linalg.LinAlgError:
            return 1e12

    v0 = float(np.var(y))
    best = minimize(nll, x0=np.log([max(v0 * 0.5, 1e-4), max(v0 * 0.5, 1e-4)]),
                    method="Nelder-Mead",
                    options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 800})
    s2u, s2e = np.exp(best.x)
    beta, XtViX, ll = _gls(X, y, idx, s2u, s2e)
    se = np.sqrt(np.diag(np.linalg.inv(XtViX)))
    return MixedFit(beta, se, float(np.sqrt(s2u)), float(np.sqrt(s2e)),
                    float(ll), names)
