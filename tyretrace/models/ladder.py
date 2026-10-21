"""The model ladder, L0 upward. NumPyro / JAX.

Each rung ships only if it beats the rung below on held-out CRPS while keeping
interval coverage inside [0.85, 0.95] -- eval/protocol.md Âsection 5. That is the
"every piece has to earn its place" promise made structural rather than
aspirational.

  L0  pooled linear-Gaussian on races. Fuel+track fused, degradation split by
      the pit-stop age reset. The identification core: if this does not work,
      nothing built on it can.
  L1  + hierarchy. Partial pooling over drivers and circuits, per-compound
      degradation. Non-centred from the start -- retrofitting it after
      divergences show up wastes days.

WHAT L0 CAN AND CANNOT SEE
--------------------------
The race-lap coefficient is -(beta_f*kappa + tau): fuel and track evolution
are FUSED and no amount of race data separates them. Supply tau from the
qualifying anchor (tyretrace.models.track_evolution) to split it. The model
takes `tau_prior` for exactly that, and defaults to a wide prior so the fused
case is the honest default rather than a hidden assumption.

Sign convention matches the simulator: lap time RISES with tyre age and with
fuel mass; track evolution LOWERS it through a session.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median

numpyro.set_host_device_count(4)


# --------------------------------------------------------------------------
# design
# --------------------------------------------------------------------------

@dataclass
class Design:
    """Model inputs, built once from a canonical lap frame."""
    y: np.ndarray               # lap_time_s
    lap: np.ndarray             # race lap, centred -- fuel + track
    age: np.ndarray             # tyre age, the degradation index
    driver: np.ndarray          # integer codes
    circuit: np.ndarray
    compound: np.ndarray
    n_driver: int
    n_circuit: int
    n_compound: int
    kappa: np.ndarray | None = None      # fuel burned per lap, kg (per row)
    tau_quali: np.ndarray | None = None  # quali track evolution, MODEL units
    dirty_air: np.ndarray | None = None  # bounded 0-1 wake exposure
    stint: np.ndarray | None = None      # (session, driver, stint) codes
    n_stint: int = 0
    stint_circuit: np.ndarray | None = None   # circuit index of each stint
    weekend: np.ndarray | None = None
    n_weekend: int = 0
    weekend_names: list = field(default_factory=list)
    driver_names: list = field(default_factory=list)
    circuit_names: list = field(default_factory=list)
    compound_names: list = field(default_factory=list)
    lap_mean: float = 0.0

    @property
    def n(self) -> int:
        return len(self.y)


def _codes(values: np.ndarray) -> tuple[np.ndarray, list]:
    levels = sorted(set(values.tolist()))
    lookup = {v: i for i, v in enumerate(levels)}
    return np.array([lookup[v] for v in values], dtype=np.int32), levels


def _apply(values: np.ndarray, levels: list) -> np.ndarray:
    """Map onto an EXISTING level set. Unseen levels go to 0 with a warning --
    silently recoding them would make held-out scores meaningless."""
    lookup = {v: i for i, v in enumerate(levels)}
    unseen = sorted(set(values.tolist()) - set(levels))
    if unseen:
        print(f"    ! {len(unseen)} level(s) unseen in training, mapped to "
              f"index 0: {unseen[:4]}")
    return np.array([lookup.get(v, 0) for v in values], dtype=np.int32)


def build_design(df: pl.DataFrame, like: "Design | None" = None) -> Design:
    """Canonical laps -> model arrays. Source-agnostic by construction.

    Pass `like=train_design` to encode a held-out frame on the TRAINING level
    set and lap centring. Building a test design independently gives it its
    own integer codes, so driver 3 in test would silently mean a different
    driver than driver 3 in train -- which quietly corrupts every held-out
    score without failing.
    """
    y = df["lap_time_s"].to_numpy().astype(np.float64)
    lap = df["lap_in_session"].to_numpy().astype(np.float64)
    age = df["tyre_age"].to_numpy().astype(np.float64)

    drivers = df["driver"].to_numpy().astype(str)
    circuits = df["circuit"].to_numpy().astype(str)
    compounds = df["compound_abs"].cast(pl.Utf8).to_numpy().astype(str)

    if like is None:
        # Centre the lap index so the intercept is mid-race pace rather than a
        # lap-zero extrapolation; uncentred, intercept and slope correlate
        # strongly and NUTS has to work much harder.
        lap_mean = float(lap.mean())
        d, dn = _codes(drivers)
        c, cn = _codes(circuits)
        k, kn = _codes(compounds)
    else:
        lap_mean = like.lap_mean
        dn, kn = like.driver_names, like.compound_names
        cn = like.circuit_names
        d = _apply(drivers, dn)
        c = _apply(circuits, cn)
        k = _apply(compounds, kn)

    return Design(y=y, lap=lap - lap_mean, age=age,
                  driver=d, circuit=c, compound=k,
                  n_driver=len(dn), n_circuit=len(cn), n_compound=len(kn),
                  driver_names=dn, circuit_names=cn, compound_names=kn,
                  lap_mean=lap_mean)


# --------------------------------------------------------------------------
# L0 -- pooled linear-Gaussian
# --------------------------------------------------------------------------

def model_l0(d: Design, y=None, tau_prior: tuple[float, float] = (0.0, 0.05)):
    """Per-driver intercepts, one fused lap slope, one degradation slope.

    `tau_prior` is where the qualifying anchor enters: pass (mean, sd) of the
    track-evolution rate in s/lap and the fuel term is what remains of the
    fused coefficient. Left wide, the split is simply not claimed.
    """
    mu_d = numpyro.sample("mu_driver",
                          dist.Normal(90.0, 10.0).expand([d.n_driver]).to_event(1))
    fused = numpyro.sample("lap_coef", dist.Normal(0.0, 0.5))
    deg = numpyro.sample("deg", dist.Normal(0.0, 0.5))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

    # declared, not fitted from race data -- races cannot identify it
    tau = numpyro.sample("tau", dist.Normal(*tau_prior))
    numpyro.deterministic("fuel_pace", -fused - tau)

    mean = mu_d[d.driver] + fused * d.lap + deg * d.age
    numpyro.sample("obs", dist.Normal(mean, sigma), obs=y)


# --------------------------------------------------------------------------
# L1 -- hierarchy, non-centred
# --------------------------------------------------------------------------

def model_l1(d: Design, y=None, tau_prior: tuple[float, float] = (0.0, 0.05)):
    """Partial pooling over drivers and circuits; degradation per compound.

    Non-centred parameterisation throughout: sampling a standard normal and
    scaling it decouples the location from the scale, which is what keeps NUTS
    out of the funnel when a group has few laps.
    """
    # --- pace -----------------------------------------------------------
    # CIRCUITS ARE FIXED EFFECTS, deliberately. Two reasons:
    #
    #   1. They are not exchangeable. Shrinking Monaco's pace toward Spa's is
    #      not regularisation, it is a wrong prior -- circuits differ by tens
    #      of seconds (measured sd_circuit ~8.9 s against a HalfNormal(3)
    #      prior, i.e. prior-data conflict).
    #   2. Each circuit carries ~1250 laps, so its effect is precisely
    #      determined and partial pooling buys nothing while costing geometry.
    #      Non-centring helps sparse groups; with abundant data it makes the
    #      sampler worse (measured ESS 51 out of 1000 draws, r_hat 1.04, with
    #      sd_circuit and z_circuit trading off along a ridge).
    #
    # Drivers ARE partially pooled: some appear for only a handful of laps,
    # which is exactly the case pooling is for. ZeroSumNormal pins them to sum
    # to zero so the overall level is not shared with the circuit intercepts --
    # without that constraint the hierarchy has a redundant additive degree of
    # freedom and r_hat blows up (measured: 2.23).
    circ = numpyro.sample(
        "circuit_pace",
        dist.Normal(90.0, 25.0).expand([d.n_circuit]).to_event(1))

    sd_drv = numpyro.sample("sd_driver", dist.HalfNormal(1.0))
    z_drv = numpyro.sample("z_driver", dist.ZeroSumNormal(1.0, (d.n_driver,)))
    drv = numpyro.deterministic("driver_off", sd_drv * z_drv)

    # --- fused fuel + track slope ---------------------------------------
    fused = numpyro.sample("lap_coef", dist.Normal(0.0, 0.2))
    tau = numpyro.sample("tau", dist.Normal(*tau_prior))
    numpyro.deterministic("fuel_pace", -fused - tau)

    # --- degradation ------------------------------------------------------
    deg_mu = numpyro.sample("deg_mu", dist.Normal(0.05, 0.1))
    if d.n_compound > 1:
        # partial pooling across compounds
        sd_deg = numpyro.sample("sd_deg", dist.HalfNormal(0.02))
        z_deg = numpyro.sample("z_deg",
                               dist.Normal(0, 1).expand([d.n_compound]).to_event(1))
        deg = numpyro.deterministic("deg", deg_mu + sd_deg * z_deg)
    else:
        # With a single compound label there is nothing to pool: sd_deg and
        # z_deg would trade off freely and neither is identified. Real data
        # currently hits this branch because compound_abs is a placeholder
        # until the Pirelli C1-C5 map lands.
        deg = numpyro.deterministic("deg", deg_mu[None])

    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

    mean = (circ[d.circuit] + drv[d.driver]
            + fused * d.lap
            + deg[d.compound] * d.age)
    numpyro.sample("obs", dist.Normal(mean, sigma), obs=y)


MODELS = {"L0": model_l0, "L1": model_l1}


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------

def fit(rung: str, d: Design, *, draws: int = 1000, warmup: int = 1000,
        chains: int = 4, seed: int = 0, progress: bool = False,
        target_accept: float = 0.98, **model_kwargs) -> MCMC:
    # 0.98 rather than the 0.8 default: the compound hierarchy over few groups
    # sits in a funnel, and a smaller step size is what gets NUTS down the neck.
    # Measured on the twin: 0.95 -> 2 divergences, 0.98 -> 0. Do NOT push to
    # 0.99, where the step size collapses and the chains stop mixing entirely
    # (r_hat 2.23, ESS 3) -- more caution is not monotonically better.
    # init_to_median rather than the default uniform draw. With a heavy-tailed
    # likelihood there is a second, degenerate mode -- "everything vague",
    # sigma huge -- and a chain started far from the data can fall into it and
    # never leave. Measured: one chain of two stuck at sigma 52 s on 87 s laps,
    # r_hat 3.6, while a single well-started chain fit cleanly at sigma 0.24.
    kernel = NUTS(MODELS[rung], target_accept_prob=target_accept,
                  init_strategy=init_to_median)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=draws,
                num_chains=chains, progress_bar=progress)
    mcmc.run(jax.random.PRNGKey(seed), d, y=jnp.asarray(d.y), **model_kwargs)
    return mcmc


def predict(rung: str, mcmc: MCMC, d: Design, *, seed: int = 1,
            **model_kwargs) -> np.ndarray:
    """Posterior predictive draws, shape (n_draws, n_laps)."""
    pred = Predictive(MODELS[rung], mcmc.get_samples())
    out = pred(jax.random.PRNGKey(seed), d, y=None, **model_kwargs)
    return np.asarray(out["obs"])


def diagnostics(mcmc: MCMC) -> dict:
    """r_hat and ESS across all sampled sites. Divergences are fatal, not
    cosmetic -- a divergent fit is not a fit."""
    import arviz as az

    idata = az.from_numpyro(mcmc)
    summ = az.summary(idata, round_to=6)
    extra = mcmc.get_extra_fields(group_by_chain=False)
    div = int(np.sum(np.asarray(extra["diverging"]))) if "diverging" in extra else 0
    return {
        "max_rhat": float(summ["r_hat"].max()),
        "min_ess_bulk": float(summ["ess_bulk"].min()),
        "divergences": div,
        "summary": summ,
    }


# NOTE: an earlier predict_new_circuit() drew unseen-circuit effects from the
# circuit hyperprior. It is gone with the hierarchy. The pre-registered task
# (protocol section 2) is FP -> race at the SAME circuit, so the circuit intercept is
# always fitted. Predicting a genuinely new circuit needs circuit-level
# covariates (layout, length, abrasiveness), which is a later phase.


# --------------------------------------------------------------------------
# the race lap slope, shared by every rung from L2 up
# --------------------------------------------------------------------------

def _lap_coef(d: Design):
    """Per-row race lap coefficient: fuel burn and race track evolution, with
    a per-weekend deviation from the cross-circuit relationship.

        lap_coef_w = -(beta_f * kappa_w + s * tau_quali_w) + slope_sd * z_w

    The deviation term was added 2026-10-21. Without it every weekend's slope
    had to equal the formula EXACTLY, so ten weekends' tiny within-race
    standard errors pinned s as if the relationship held perfectly across
    circuits: s = 0.87 [0.79, 0.96]. audit/f1_fuel_gap.py found that with
    between-race scatter allowed, s is not identified (0.33 +/- 0.55) and the
    fuel effect alone (0.029 s/kg -- physics) predicts unseen race slopes
    best. Whatever s takes, it is subtracted from fuel, so the overconfident
    s was the whole fuel-pace gap.

    Non-centred, so the sampler is not caught in the funnel when the data
    pull slope_sd toward zero.
    """
    beta_f = numpyro.sample("beta_f", dist.Normal(0.033, 0.020))
    s = numpyro.sample("tau_scale", dist.Normal(0.5, 0.5))
    slope_sd = numpyro.sample("slope_sd", dist.HalfNormal(0.02))
    z = numpyro.sample("z_slope", dist.Normal(0, 1).expand([d.n_weekend]).to_event(1))
    return numpyro.deterministic(
        "lap_coef_row", -(beta_f * d.kappa + s * d.tau_quali) + slope_sd * z[d.weekend])


# --------------------------------------------------------------------------
# L2 -- split the fused coefficient using cross-circuit variation
# --------------------------------------------------------------------------

def model_l2(d: Design, y=None):
    """Decompose the fused lap coefficient instead of assuming its split.

    L0 and L1 leave  lap_coef = -(beta_f*kappa + tau)  fused, and the only way
    to split it was to declare tau from qualifying. The sensitivity analysis
    showed that declaration is worth a factor of two on beta_f, so it deserves
    to be estimated rather than asserted.

    The decomposition, per circuit c:

        lap_coef_c = -(beta_f * kappa_c  +  s * tau_quali_c)

    beta_f (s/kg) and s (how much of qualifying's track evolution survives into
    the race) are GLOBAL. kappa_c and tau_quali_c are both observed and vary
    across circuits -- kappa from race length, tau from the qualifying anchor.
    Two unknowns, ten weekends of variation, and the two regressors are only
    mildly collinear (measured corr -0.54, VIF 1.42), so both are identified.

    s is the number the earlier sensitivity table could only bracket. s = 1
    means the race evolves exactly as qualifying did; s = 0 means the track is
    already done rubbering in by race time.
    """
    circ = numpyro.sample(
        "circuit_pace",
        dist.Normal(90.0, 25.0).expand([d.n_circuit]).to_event(1))

    sd_drv = numpyro.sample("sd_driver", dist.HalfNormal(1.0))
    z_drv = numpyro.sample("z_driver", dist.ZeroSumNormal(1.0, (d.n_driver,)))
    drv = numpyro.deterministic("driver_off", sd_drv * z_drv)

    # weakly informative around the paddock convention, wide enough to be
    # overruled by data -- this must not become the assumption it replaces
    lap_coef = _lap_coef(d)

    # Degradation per weekend x compound, with NO pooling across weekends:
    # relative labels are not comparable between races ("SOFT" is three
    # different rubbers across a season), so pooling them is exactly the error
    # the Pirelli C1-C5 map exists to prevent. Within a weekend the labels are
    # unambiguous, which is what makes the CONTRAST (estimand E1) valid here.
    deg = numpyro.sample(
        "deg",
        dist.Normal(0.05, 0.08).expand([d.n_weekend, d.n_compound]).to_event(2))

    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

    mean = (circ[d.circuit] + drv[d.driver]
            + lap_coef * d.lap
            + deg[d.weekend, d.compound] * d.age)
    numpyro.sample("obs", dist.Normal(mean, sigma), obs=y)


MODELS["L2"] = model_l2


# --------------------------------------------------------------------------
# L3 -- dirty air as a covariate, not an exclusion
# --------------------------------------------------------------------------

def model_l3(d: Design, y=None):
    """L2 plus wake exposure.

    The submission's position is that traffic is a COVARIATE, never a reason
    to bin a lap. This is that, and it comes free from timing data -- no
    telemetry required.

    It also settles a falsifiable prediction. The field spreads out over a
    race (corr(race lap, gap) = +0.37), so early laps are dirtier and slower,
    which is a NEGATIVE contribution to the lap coefficient on top of fuel.
    Controlling for it must therefore push the identified fuel pace DOWN, away
    from the physics expectation, not toward it. Measured: it does, by 6.5%.
    So dirty air is real (0.57 s/lap in full wake) and is NOT the explanation
    for the fuel-pace shortfall.
    """
    circ = numpyro.sample(
        "circuit_pace",
        dist.Normal(90.0, 25.0).expand([d.n_circuit]).to_event(1))

    sd_drv = numpyro.sample("sd_driver", dist.HalfNormal(1.0))
    z_drv = numpyro.sample("z_driver", dist.ZeroSumNormal(1.0, (d.n_driver,)))
    drv = numpyro.deterministic("driver_off", sd_drv * z_drv)

    lap_coef = _lap_coef(d)

    # positive: sitting in another car's wake costs time
    wake = numpyro.sample("wake_s", dist.Normal(0.5, 0.5))

    deg = numpyro.sample(
        "deg",
        dist.Normal(0.05, 0.08).expand([d.n_weekend, d.n_compound]).to_event(2))

    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

    mean = (circ[d.circuit] + drv[d.driver]
            + lap_coef * d.lap
            + wake * d.dirty_air
            + deg[d.weekend, d.compound] * d.age)
    numpyro.sample("obs", dist.Normal(mean, sigma), obs=y)


MODELS["L3"] = model_l3


# --------------------------------------------------------------------------
# L4 -- calibration: stint offsets and heavy tails
# --------------------------------------------------------------------------

def model_l4(d: Design, y=None):
    """L3 plus the two things the residuals actually asked for.

    L2 and L3 both under-cover on held-out laps (0.843 and 0.849 against a
    0.85-0.95 band), so neither ships. Diagnosing the residuals rather than
    guessing gave two causes:

      1. A PER-STINT OFFSET carrying 52% of residual variance (between-stint
         sd 0.457 s against within-stint 0.442 s). Physically obvious in
         hindsight -- each stint starts at its own fuel level relative to the
         model's linear path, on its own tyre set, with the driver in its own
         mode. Nothing in L0-L3 could represent that, so it leaked into the
         noise term and the model then trusted its mean far too much.

      2. HEAVY, RIGHT-SKEWED TAILS. Excess kurtosis +1.54 and skew +0.58,
         with |z|>3 at 1.12% against a Gaussian's 0.27% -- four times too
         many. Laps can be much slower (traffic, a mistake, a slow zone) but
         not much faster, and a Gaussian likelihood has to inflate sigma to
         cover the tail, which distorts everything else.

    The prior on the stint sd is deliberately tight, HalfNormal(0.5) against a
    measured 0.46 s. That is not timidity: per-stint offsets are in direct
    tension with the pit-stop identification. Races separate fuel from
    degradation precisely BECAUSE stint levels carry information -- fuel runs
    on across a stop while tyre age resets. Let the offsets float freely and
    they eat that signal along with the noise. The prior keeps them to the
    set-to-set variation they are meant to capture.

    So: tightly pooled stint intercepts, and a Student-t likelihood whose
    degrees of freedom are estimated rather than assumed. A stint appearing
    only in the held-out block gets no likelihood contribution, so its offset
    falls back to the population prior and its interval widens -- which is the
    honest answer for a stint the model has never seen.

    This displaces the regime model that the original plan called L4. Regimes
    on a miscalibrated base would inherit the miscalibration; calibration is a
    prerequisite, not a parallel track.
    """
    circ = numpyro.sample(
        "circuit_pace",
        dist.Normal(90.0, 25.0).expand([d.n_circuit]).to_event(1))

    # A stint belongs to exactly one driver, so per-stint offsets SUBSUME
    # per-driver ones; keeping both leaves a ridge. Driver pace is recovered
    # afterwards as the mean of that driver's stint offsets.
    #
    # The offsets are demeaned WITHIN CIRCUIT, because stints also nest inside
    # circuits: without that, each circuit's intercept and the mean of its
    # stint offsets trade off freely. That ridge is what sent r_hat to 3.04
    # and sigma to 53 s on the first attempt.
    sd_stint = numpyro.sample("sd_stint", dist.HalfNormal(0.5))
    z_raw = numpyro.sample(
        "z_stint", dist.Normal(0, 1).expand([d.n_stint]).to_event(1))
    csum = jnp.zeros(d.n_circuit).at[d.stint_circuit].add(z_raw)
    ccount = jnp.zeros(d.n_circuit).at[d.stint_circuit].add(1.0)
    z_stint = z_raw - (csum / jnp.maximum(ccount, 1.0))[d.stint_circuit]
    stint_off = numpyro.deterministic("stint_off", sd_stint * z_stint)

    lap_coef = _lap_coef(d)

    wake = numpyro.sample("wake_s", dist.Normal(0.5, 0.5))

    deg = numpyro.sample(
        "deg",
        dist.Normal(0.05, 0.08).expand([d.n_weekend, d.n_compound]).to_event(2))

    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    # nu low => heavy tails, and the data wants them (fitted ~2.6). Shifted by
    # 2 so the variance stays finite: below nu=2 a Student-t has no variance
    # at all, the likelihood surface goes flat, and that flat region is the
    # degenerate mode chains get trapped in.
    nu = numpyro.deterministic("nu", 2.0 + numpyro.sample("nu_raw",
                                                          dist.Gamma(2.0, 0.2)))

    mean = (circ[d.circuit] + stint_off[d.stint]
            + lap_coef * d.lap
            + wake * d.dirty_air
            + deg[d.weekend, d.compound] * d.age)
    numpyro.sample("obs", dist.StudentT(nu, mean, sigma), obs=y)


MODELS["L4"] = model_l4


# --------------------------------------------------------------------------
# L5 -- variance that grows with tyre age
# --------------------------------------------------------------------------

def model_l5(d: Design, y=None):
    """REJECTED. Kept as a recorded negative result, not a shipping rung.

    Verdict: held-out coverage got WORSE, 0.813 -> 0.759, and the fitted
    gamma came out NEGATIVE (-0.387, sigma shrinking with age) -- the opposite
    of the diagnostic that motivated it.

    Why it failed: the functional form is wrong. Measured residual sd by tyre
    age, with the lap trim off so nothing is censored:

        age  1-10   sd 1.074      warm-up, out-laps, tyres not in
        age 11-20   sd 0.478      steady state
        age 21-60   sd 0.628      degradation

    That is U-SHAPED, not monotone. A log-linear sigma(age) can only be
    increasing or decreasing, so it fit the large early drop and concluded
    variance falls with age -- then gave old tyres intervals that were too
    narrow, making the very problem it was built to fix worse.

    What would actually work: a non-monotone scale (a spline in age, or a
    warm-up term separate from a degradation term), or the regime model the
    original plan called for, where warm-up and cliff are distinct states with
    their own variances. That is the right next attempt.

    L4 remains the best rung: best CRPS (0.344) and MAE (0.463) of anything
    tried, still under-covering at 0.813, and therefore still not shipped.

    ---- original rationale, preserved ----

    L4 with a scale that depends on tyre age.

    L4 got the best CRPS and MAE of any rung and still under-covered (0.813),
    so it did not ship. Splitting held-out coverage by tyre age showed why:

        age  1-5    coverage 0.853
        age  6-10   coverage 0.839
        age 11-15   coverage 0.839
        age 16-25   coverage 0.822
        age 26-60   coverage 0.772

    A clean monotone decline, while median residuals stayed within 0.1 s of
    zero throughout. So this is not bias -- the mean function is fine -- it is
    UNDER-DISPERSION. An old tyre produces more variable lap times (grip comes
    and goes, the driver manages it differently lap to lap), and a constant
    sigma gives those laps the same interval width as a fresh set.

    So sigma grows log-linearly with age. That is also the cliff precursor the
    project set out to detect: variance rising before the mean does is exactly
    what a tyre about to let go looks like, and it is measurable here before
    any regime model exists.

    Scaled by 20 laps so gamma is "log change in sigma per 20 laps of age" and
    the prior is interpretable rather than a raw per-lap number near zero.
    """
    circ = numpyro.sample(
        "circuit_pace",
        dist.Normal(90.0, 25.0).expand([d.n_circuit]).to_event(1))

    sd_stint = numpyro.sample("sd_stint", dist.HalfNormal(0.5))
    z_raw = numpyro.sample(
        "z_stint", dist.Normal(0, 1).expand([d.n_stint]).to_event(1))
    csum = jnp.zeros(d.n_circuit).at[d.stint_circuit].add(z_raw)
    ccount = jnp.zeros(d.n_circuit).at[d.stint_circuit].add(1.0)
    z_stint = z_raw - (csum / jnp.maximum(ccount, 1.0))[d.stint_circuit]
    stint_off = numpyro.deterministic("stint_off", sd_stint * z_stint)

    lap_coef = _lap_coef(d)

    wake = numpyro.sample("wake_s", dist.Normal(0.5, 0.5))
    deg = numpyro.sample(
        "deg",
        dist.Normal(0.05, 0.08).expand([d.n_weekend, d.n_compound]).to_event(2))

    sigma0 = numpyro.sample("sigma0", dist.HalfNormal(1.0))
    gamma = numpyro.sample("gamma_age", dist.Normal(0.0, 0.5))
    sigma = numpyro.deterministic("sigma_row",
                                  sigma0 * jnp.exp(gamma * d.age / 20.0))

    nu = numpyro.deterministic("nu", 2.0 + numpyro.sample("nu_raw",
                                                          dist.Gamma(2.0, 0.2)))

    mean = (circ[d.circuit] + stint_off[d.stint]
            + lap_coef * d.lap
            + wake * d.dirty_air
            + deg[d.weekend, d.compound] * d.age)
    numpyro.sample("obs", dist.StudentT(nu, mean, sigma), obs=y)


MODELS["L5"] = model_l5


# --------------------------------------------------------------------------
# L6 -- compromised laps: a one-sided slow component, likelier on old tyres
# --------------------------------------------------------------------------

def _l4_mean(d: Design):
    """L4's mean function, unchanged: circuit level, within-circuit stint
    offsets, the identified fuel/track split, wake and per-weekend degradation.
    Shared so a rung that only changes the noise model cannot drift the mean."""
    circ = numpyro.sample(
        "circuit_pace",
        dist.Normal(90.0, 25.0).expand([d.n_circuit]).to_event(1))

    sd_stint = numpyro.sample("sd_stint", dist.HalfNormal(0.5))
    z_raw = numpyro.sample(
        "z_stint", dist.Normal(0, 1).expand([d.n_stint]).to_event(1))
    csum = jnp.zeros(d.n_circuit).at[d.stint_circuit].add(z_raw)
    ccount = jnp.zeros(d.n_circuit).at[d.stint_circuit].add(1.0)
    z_stint = z_raw - (csum / jnp.maximum(ccount, 1.0))[d.stint_circuit]
    stint_off = numpyro.deterministic("stint_off", sd_stint * z_stint)

    lap_coef = _lap_coef(d)

    wake = numpyro.sample("wake_s", dist.Normal(0.5, 0.5))
    deg = numpyro.sample(
        "deg",
        dist.Normal(0.05, 0.08).expand([d.n_weekend, d.n_compound]).to_event(2))

    return (circ[d.circuit] + stint_off[d.stint]
            + lap_coef * d.lap
            + wake * d.dirty_air
            + deg[d.weekend, d.compound] * d.age)


SLOW_AGE_CENTRE = 15.0      # p_slow_a is the log-odds of a slow lap at this age
SLOW_AGE_SCALE = 10.0       # p_slow_b is the change in log-odds per 10 laps


def model_l6(d: Design, y=None):
    """REJECTED. Kept as a recorded negative result, not a shipping rung.

    Verdict, on L4's split, 400 draws x 2 chains, three variants declared
    before any were scored (all reported, not just the best):

        variant                       CRPS    cov90   slow miss  fast miss
        L4 (reference, same draws)    0.3454  0.812   11.8%      7.1%
        A  t clean + Normal slow      0.3477  0.763   12.5%     11.2%
        B  Normal clean + t slow      0.3465  0.755   11.7%     12.8%
        C  Normal clean + Normal slow 0.3460  0.760   11.7%     12.4%

    Every variant covers WORSE than L4. The mixture let the clean component
    tighten (sigma 0.224-0.236 vs 0.239), which created fast-side misses,
    while the slow side did not improve. And the age effect on the share of
    compromised laps came out NEGATIVE (p_slow_b -0.8 to -1.9) -- fewer
    compromised laps on old tyres, the opposite of the hypothesis. So the
    one-sided misses on old tyres are not a lap-level contamination process.

    What is left: the misses are lopsided by tyre age while the median stays
    near zero, which points at the MEAN on old tyres (degradation curvature or
    a cliff within some stints) rather than at the noise. That is a regime
    question, and also the place to look once the convergence problem below
    is fixed -- r_hat was 1.27-1.39 for every variant AND for L4 at this draw
    count, so these comparisons are between unconverged fits.

    ---- original rationale, preserved ----

    L4's mean, with a noise model that can be slow but not fast.

    L4's held-out misses are ONE-SIDED. Split by which side of the 90% band a
    missed lap falls on:

        age   1-5    above q95  7.7%   below q05  7.0%
        age 11-15    above q95  9.5%   below q05  6.6%
        age 26-30    above q95 16.1%   below q05  6.8%

    Calibrated would be 5% each side. The fast side is roughly right at every
    age; the slow side is wrong and gets worse with tyre age, while the median
    forecast stays within 0.1 s. A Student-t is symmetric, so it cannot fit
    that -- widening it to catch the slow laps would over-cover the fast side.

    Physically there are two kinds of lap. A CLEAN lap is the driver on the
    car's pace. A COMPROMISED lap -- stuck behind traffic the wake term does
    not capture, a lock-up, a lift, a tyre starting to go -- is slower, never
    faster. So: a two-component mixture,

        clean        StudentT(nu, mean, sigma)                 prob 1 - p
        compromised  Normal(mean + slow_s, sigma + sigma_slow)  prob p

    with slow_s > 0 and the compromised component strictly wider, which also
    pins the labels so the components cannot swap. p depends on tyre age
    through a logistic, so a rising share of compromised laps on old tyres --
    a cliff precursor -- is something the model can find, not something it
    assumes: p_slow_b ~ Normal(0, 1) is centred on no age effect.

    This is the regime model's simplest special case: two states, independent
    across laps. If it calibrates, the HMM has to beat it rather than L4.
    """
    mean = _l4_mean(d)

    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    nu = numpyro.deterministic("nu", 2.0 + numpyro.sample("nu_raw",
                                                          dist.Gamma(2.0, 0.2)))

    slow = numpyro.sample("slow_s", dist.HalfNormal(2.0))
    sigma_slow = numpyro.sample("sigma_slow", dist.HalfNormal(2.0))
    # prior centred on ~8% compromised laps at age 15, wide on the log-odds
    a = numpyro.sample("p_slow_a", dist.Normal(-2.5, 1.0))
    b = numpyro.sample("p_slow_b", dist.Normal(0.0, 1.0))
    p = jax.nn.sigmoid(a + b * (d.age - SLOW_AGE_CENTRE) / SLOW_AGE_SCALE)

    mix = dist.MixtureGeneral(
        dist.Categorical(probs=jnp.stack([1.0 - p, p], axis=-1)),
        [dist.StudentT(nu, mean, sigma),
         dist.Normal(mean + slow, sigma + sigma_slow)])
    numpyro.sample("obs", mix, obs=y)


MODELS["L6"] = model_l6
