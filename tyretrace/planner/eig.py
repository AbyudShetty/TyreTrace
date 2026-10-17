"""FP2 run planner v1 -- choose practice runs by expected information gain.

The submission's claim: the best way to deconfound is not to model the
confounding harder, it is to design runs that never confound in the first
place. This module makes that claim computable.

THE MODEL A PLANNED LAP FOLLOWS
-------------------------------
From the team's side of the pit wall, so fuel load is KNOWN (they put it in):

    lap_time = alpha_run + beta_f * fuel_kg + deg_c * tyre_age + tau * minute + noise

alpha_run is a per-run intercept -- each run is its own setup, engine mode
and driver programme, exactly the per-stint offset L4 found carrying half
the residual variance. It is a nuisance with a wide prior, and it is what
makes the problem interesting: within one run, fuel falls by kappa per lap,
tyre age rises by one and the clock advances by one lap time, all in lock
step. A run therefore teaches only the FUSED slope

    -beta_f * kappa  +  deg_c  +  tau * lap_s / 60

Different compounds give different fused slopes, so compound CONTRASTS are
learned; the LEVELS are not (README: "two compounds, one session -> a
contrast, rank deficient by one"). Start fuel does not matter at all -- it
moves only the intercept. What breaks the lock step is a MID-RUN TYRE CHANGE
without refuelling: age resets to one while fuel and the run intercept carry
on. That is a pit stop, run deliberately in practice, and it is the only
candidate here that identifies a degradation level.

WHY THE INFORMATION GAIN IS EXACT
---------------------------------
Linear model, Gaussian prior, Gaussian noise: the posterior covariance does
not depend on the lap times that will be observed, only on the design. So
the expected information gain about target quantities T = A theta is

    EIG = 0.5 * [ logdet(A S0 A') - logdet(A S A') ]     (nats)

with S0 the prior and S = (S0^-1 + X'X / sigma^2)^-1 the posterior
covariance. No Monte Carlo, no sampling noise -- the nested-MC estimator
in the submission is for the non-linear regime model, a later version.

Log-det information gain is submodular in added runs. Under a session-time
budget, plain greedy by gain per minute can do badly (measured: it lost to
a typical FP2), so plan() seeds from two greedy passes and a beam search,
then improves the best by local search. It is checked against exhaustive search, and
fast enough to re-plan between runs.

WHAT V1 DOES NOT MODEL
----------------------
Degradation is linear (no cliff, no graining), tyre allocation is not a
constraint, one car. Those are the regime model's job and v2's.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product

import numpy as np

COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
PARAMS = ("beta_f", "deg_SOFT", "deg_MEDIUM", "deg_HARD", "tau")
P = len(PARAMS)

OUT_IN_LAPS = 2          # out-lap + in-lap: driven, never timed for pace
SWAP_LAPS = 2            # in-lap + out-lap around a mid-run tyre change
TURNAROUND_MIN = 4.0     # garage time between runs


@dataclass(frozen=True)
class Prior:
    """What is believed before FP2. Defaults are deliberately not this
    project's own estimates: beta_f is the paddock convention with a wide sd
    (the identified race value is half of it and unexplained -- README), and
    tau is the qualifying anchor, which IS measured."""
    beta_f: tuple[float, float] = (0.033, 0.010)        # s/kg
    deg: tuple[float, float] = (0.05, 0.04)             # s/lap, each compound
    tau: tuple[float, float] = (-0.0165, 0.003)         # s/min, quali anchor
    run_sd: float = 3.0                                 # run intercepts, s
    sigma: float = 0.30                                 # lap noise, s
    kappa: float = 1.8                                  # kg burned per lap
    lap_s: float = 90.0

    def cov(self) -> np.ndarray:
        sd = [self.beta_f[1]] + [self.deg[1]] * len(COMPOUNDS) + [self.tau[1]]
        return np.diag(np.square(sd))


@dataclass(frozen=True)
class Run:
    compound: str
    laps: int                        # timed laps
    start_fuel_kg: float = 60.0
    swap_to: str | None = None       # tyre change at half distance, no refuel

    @property
    def overhead_laps(self) -> int:
        return OUT_IN_LAPS + (SWAP_LAPS if self.swap_to else 0)

    def minutes(self, lap_s: float) -> float:
        return (self.laps + self.overhead_laps) * lap_s / 60.0

    def label(self) -> str:
        s = f"{self.laps} laps {self.compound} ({self.start_fuel_kg:.0f} kg)"
        if self.swap_to:
            s += f" → {self.swap_to} at half"
        return s


@dataclass
class Programme:
    runs: list[Run] = field(default_factory=list)
    starts: list[float] = field(default_factory=list)     # session minute

    def end(self, prior: Prior) -> float:
        if not self.runs:
            return 0.0
        return self.starts[-1] + self.runs[-1].minutes(prior.lap_s)

    def add(self, run: Run, prior: Prior, first_start: float = 5.0) -> "Programme":
        start = first_start if not self.runs else self.end(prior) + TURNAROUND_MIN
        return Programme(self.runs + [run], self.starts + [start])


# --------------------------------------------------------------------------
# design matrix
# --------------------------------------------------------------------------

def run_rows(run: Run, start_min: float, prior: Prior) -> np.ndarray:
    """Global-parameter columns (PARAMS order) for each timed lap of a run.

    Lap k (1-based, timed) carries fuel for the laps still ahead; the out-lap
    has already burned one lap's fuel. A swap at half distance resets age to
    one and, because the in/out laps are driven, advances fuel and clock.
    """
    rows = []
    lap_min = prior.lap_s / 60.0
    half = run.laps // 2 if run.swap_to else run.laps
    driven = 1                                   # the out-lap
    age = 1
    comp = run.compound
    for k in range(run.laps):
        if run.swap_to and k == half:
            driven += SWAP_LAPS
            age = 1
            comp = run.swap_to
        fuel = run.start_fuel_kg - prior.kappa * driven
        minute = start_min + driven * lap_min
        r = np.zeros(P)
        r[0] = fuel
        r[1 + COMPOUNDS.index(comp)] = age
        r[4] = minute
        rows.append(r)
        driven += 1
        age += 1
    return np.array(rows)


def design(prog: Programme, prior: Prior) -> np.ndarray:
    """Full design: global columns then one intercept column per run."""
    blocks = []
    n = len(prog.runs)
    for i, (run, t0) in enumerate(zip(prog.runs, prog.starts)):
        g = run_rows(run, t0, prior)
        icpt = np.zeros((len(g), n))
        icpt[:, i] = 1.0
        blocks.append(np.hstack([g, icpt]))
    return np.vstack(blocks) if blocks else np.zeros((0, P))


def posterior_cov(prog: Programme, prior: Prior) -> np.ndarray:
    """Posterior covariance of the GLOBAL parameters, run intercepts
    marginalised out. Independent of the lap times that will be observed."""
    n = len(prog.runs)
    S0 = np.zeros((P + n, P + n))
    S0[:P, :P] = prior.cov()
    S0[P:, P:] = np.eye(n) * prior.run_sd ** 2
    X = design(prog, prior)
    prec = np.linalg.inv(S0) + X.T @ X / prior.sigma ** 2
    return np.linalg.inv(prec)[:P, :P]


# --------------------------------------------------------------------------
# targets and information gain
# --------------------------------------------------------------------------

TARGETS = {
    "degradation per compound": np.eye(P)[1:4],
    "SOFT − HARD contrast": (np.eye(P)[1] - np.eye(P)[3])[None, :],
    "degradation + fuel effect": np.eye(P)[0:4],
}


def eig(prog: Programme, prior: Prior, A: np.ndarray) -> float:
    """Expected information gain about A @ theta, in bits."""
    S0 = A @ prior.cov() @ A.T
    S = A @ posterior_cov(prog, prior) @ A.T
    _, ld0 = np.linalg.slogdet(S0)
    _, ld = np.linalg.slogdet(S)
    return 0.5 * (ld0 - ld) / np.log(2.0)


def posterior_sd(prog: Programme, prior: Prior) -> dict[str, float]:
    S = posterior_cov(prog, prior)
    out = {p: float(np.sqrt(S[i, i])) for i, p in enumerate(PARAMS)}
    a = TARGETS["SOFT − HARD contrast"][0]
    out["SOFT−HARD"] = float(np.sqrt(a @ S @ a))
    return out


# --------------------------------------------------------------------------
# candidates and the greedy planner
# --------------------------------------------------------------------------

def candidates(lengths=tuple(range(4, 21)), allow_swap: bool = True,
               fuels=(30.0, 100.0)) -> list[Run]:
    """Every run the planner may choose: compound x length x start fuel, plus
    mid-run tyre changes on runs of six laps or more.

    Start fuel moves only the run intercept, so it carries no information when
    run pace is unknown -- but under the default 3 s prior on run pace a light
    run and a heavy one do say a little about beta_f (measured: two 11-lap
    long runs are worth 1.35 bits at 60 kg and 1.37 at 90 kg). Small, but
    dropping it let a typical FP2 beat the planner, so light and heavy are
    both offered.
    """
    runs = [Run(c, n, f) for c, n, f in product(COMPOUNDS, lengths, fuels)]
    if allow_swap:
        runs += [Run(c, n, f, swap_to=s)
                 for c, n, f, s in product(COMPOUNDS, lengths, fuels, COMPOUNDS) if n >= 6]
    return runs


@dataclass
class Step:
    run: Run
    start: float
    gain_bits: float          # marginal information this run adds
    total_bits: float


QUALI_SIM = Run("SOFT", 3, start_fuel_kg=20.0)


def _schedule(runs: list[Run], prior: Prior, start: Programme | None) -> Programme:
    prog = start or Programme()
    for r in runs:
        prog = prog.add(r, prior)
    return prog


def _greedy(prior, A, session_min, cands, min_gain, start, by_rate) -> list[Run]:
    prog = start or Programme()
    total = eig(prog, prior, A) if prog.runs else 0.0
    added: list[Run] = []
    while True:
        best, best_score = None, 0.0
        for r in cands:
            nxt = prog.add(r, prior)
            if nxt.end(prior) > session_min:
                continue
            g = eig(nxt, prior, A) - total
            score = g / r.minutes(prior.lap_s) if by_rate else g
            if g > min_gain and score > best_score:
                best, best_score = (r, nxt, g), score
        if best is None:
            return added
        r, prog, g = best
        total += g
        added.append(r)


def _beam(prior, A, session_min, cands, start, width=60) -> list[Run]:
    """Keep the `width` best partial programmes at every depth, not one.

    Greedy and single-move local search both get trapped where the better
    programme needs two changes at once -- swapping one long run for two
    shorter ones loses information on either move alone. Measured: with the
    qualifying sim fixed and no tyre changes, they settled 10% BELOW a typical
    FP2 whose own runs were among the candidates. The beam keeps those
    two-step paths alive. Width 12 was still trapped (long single runs crowd
    the beam early); from 50 up the result stops changing, so 60.
    """
    def value(rs):
        prog = _schedule(rs, prior, start)
        return eig(prog, prior, A) if prog.end(prior) <= session_min else -np.inf

    beam, best, best_v = [[]], [], value([]) if start else 0.0
    while beam:
        grown: dict[tuple, tuple[float, list[Run]]] = {}
        for rs in beam:
            for c in cands:
                nxt = rs + [c]
                v = value(nxt)
                if v == -np.inf:
                    continue
                key = tuple(sorted(r.label() for r in nxt))
                if key not in grown or v > grown[key][0]:
                    grown[key] = (v, nxt)
        top = sorted(grown.values(), key=lambda t: t[0], reverse=True)[:width]
        for v, rs in top:
            if v > best_v:
                best, best_v = rs, v
        beam = [rs for _, rs in top]
    return best


def _improve(runs, prior, A, session_min, cands, start, max_rounds=25) -> list[Run]:
    """Local search: replace any one run with any candidate, drop a run, or
    append one, whenever the whole programme gains information and still
    fits. Stops at a local optimum."""
    def value(rs):
        prog = _schedule(rs, prior, start)
        return eig(prog, prior, A) if prog.end(prior) <= session_min else -np.inf

    best = value(runs)
    for _ in range(max_rounds):
        moves = [runs[:i] + [c] + runs[i + 1:] for i in range(len(runs)) for c in cands]
        moves += [runs[:i] + runs[i + 1:] for i in range(len(runs))]
        moves += [runs + [c] for c in cands]
        top = max(moves, key=value, default=None)
        if top is None or value(top) <= best + 1e-9:
            return runs
        runs, best = top, value(top)
    return runs


def plan(prior: Prior, A: np.ndarray, *, session_min: float = 60.0,
         cands: list[Run] | None = None, min_gain: float = 0.01,
         start: Programme | None = None) -> list[Step]:
    """Choose the runs that teach the most about A @ theta before the flag.

    Two greedy passes (by information per minute, and absolute) and a beam
    search, then local search from the best of the three. Rate-greedy alone
    is myopic under a time budget: measured, it took one efficient 16-lap
    run, was left with seven minutes and filled them with a 4-lap run worth
    0.07 bits, LOSING to a typical FP2 by 8%. Taking the better of the two
    greedy passes is the textbook fix for a budgeted submodular objective,
    and the local search closes what remains; tests/test_planner.py checks
    the result against exhaustive search on a short session.

    `start` holds runs the team must do regardless -- a qualifying simulation,
    setup work. They are kept, their information counts, and the planner
    fills the time that remains. Without it the planner wins partly by
    skipping runs a real team cannot skip, which flatters it.
    """
    cands = cands or candidates()

    def value(rs):
        return eig(_schedule(rs, prior, start), prior, A)

    seeds = [_greedy(prior, A, session_min, cands, min_gain, start, by_rate=b)
             for b in (True, False)]
    seeds.append(_beam(prior, A, session_min, cands, start))
    runs = _improve(max(seeds, key=value), prior, A, session_min, cands, start)

    # report each run's marginal contribution in the order it is driven
    base = start or Programme()
    total = eig(base, prior, A) if base.runs else 0.0
    prog, steps = base, []
    for r in runs:
        prog = prog.add(r, prior)
        g = eig(prog, prior, A)
        steps.append(Step(r, prog.starts[-1], g - total, g))
        total = g
    return steps


def programme(steps: list[Step], start: Programme | None = None) -> Programme:
    base = start or Programme()
    return Programme(base.runs + [s.run for s in steps],
                     base.starts + [s.start for s in steps])


def standard_programme(prior: Prior) -> Programme:
    """A typical FP2: a quali simulation on softs, then long runs on the two
    race compounds. Built with the same timing rules as the planner, so the
    comparison is on equal session time."""
    prog = Programme()
    for r in (QUALI_SIM,
              Run("MEDIUM", 11, start_fuel_kg=90.0),
              Run("HARD", 11, start_fuel_kg=90.0)):
        prog = prog.add(r, prior)
    return prog


def with_prior(prior: Prior, **kw) -> Prior:
    return replace(prior, **kw)
