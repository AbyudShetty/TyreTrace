# Tyre Trace

Causal tyre-degradation intelligence — recovering true wear rates from confounded
practice data.

**Status: research prototype with a data lake, model audits, rolling backtests,
run planner and initial dashboards.** Predictive lap-time uncertainty has a
rolling calibration layer. Raw Bayesian ladder calibration, latent-parameter
validation and live game proof remain distinct questions.

### Predictive uncertainty update — 2026-10-26

`python -m tyretrace run uncertainty_t` runs the uncertainty audit, including
both 2026 fuel-load assumptions, without overwriting the original backtest.
`python -m tyretrace export` refreshes the React overview's uncertainty table.

The new layer preserves point forecasts and learns predictive dispersion from
strictly earlier forecast errors, weighting weekends equally. A Student-t
distribution accommodates rare large misses, with raw-spread, race-horizon and
tyre-age scale terms. At least eight earlier forecast weekends are required.
A Gaussian variant improved coverage but failed CRPS; its failed results remain.

| 2026 evaluation: 14 weekends | Original CRPS | Repaired CRPS | Original 90% coverage | Repaired coverage |
|---|---:|---:|---:|---:|
| 70 kg, all 11,560 scored laps | 0.51940 | 0.51749 | 82.2% | 93.6% |
| 70 kg, 7,251 laps after a stop | 0.54964 | 0.54653 | 79.7% | 93.1% |
| 110 kg, all scored laps | 0.55578 | 0.54912 | 80.0% | 93.0% |
| 110 kg, after a stop | 0.60170 | 0.59247 | 76.2% | 92.1% |

Both scenarios clear the pooled predictive ship gate: lower CRPS and coverage
inside [0.85, 0.95], overall and after stops. This does not establish a gain
on every race or superiority over conventional correction. The paired-weekend
bootstrap interval for the small CRPS improvement includes zero. Coverage varies
by race and the 95% intervals are conservative.

This is retrospective rolling validation: after the Gaussian attempt failed on
2026, that block was no longer untouched. Prospective replication needs new
races. These are marginal lap-time predictive intervals, not calibrated latent
tyre/fuel intervals or joint strategy simulations. Original L3 remains HOLD
at scale. See [`docs/UNCERTAINTY.md`](docs/UNCERTAINTY.md). Earlier narratives
below preserve historical findings, superseded where a dated update says so.

Also added on 2026-10-26: [game capture/replay and independently verified packet
layouts](docs/GAME_RIG.md), [conditional one-stop decision diagnostics](docs/STRATEGY_EVALUATION.md),
and [verification of the published tyre-curve baseline](docs/PUBLISHED_BASELINE.md).
Live game proof and the full rolling ML/published comparisons remain pending.

---

## The problem in one paragraph

Inside a practice stint, fuel burn and tyre wear both index on lap-in-run, so a
regression of lap time on lap number recovers only the *sum* of their slopes. The
degradation rate every strategist quotes is not identified. The standard fix
imports an unverifiable fuel coefficient and calls the remainder degradation. We
treat it as an identification problem instead.

**What identifies what** (derived in `audit/a3_synthetic_recovery.py`, verified in
`tests/test_recovery.py`):

| Data | Separates | Leaves fused |
|---|---|---|
| **Qualifying** | track evolution, at ~constant minimum fuel | — |
| Race | degradation, via the pit-stop age reset | fuel + track evolution |
| Two compounds, one session | δ_A − δ_B (a **contrast**) | the levels — rank deficient by one |
| Practice | *nothing on its own* — consumes the above | |

The chain: **qualifying → τ; race + τ → fuel pace and degradation; practice consumes
both.** Practice no longer has to identify anything, which is the point — it was the
weakest link and it was carrying the most weight.

---

## Layout

```
tyretrace/
  schema.py      canonical lap table — THE contract. sim and real ingest emit it.
  data/          FastF1 -> canonical laps (timing only; telemetry never stored raw)
  sim/           synthetic twin with known truth
  models/        mixed.py, track_evolution.py (quali anchor), ladder.py (L0/L1 NumPyro)
  eval/          metrics.py — CRPS, calibration, the ship gate
  rig/           F1 UDP decoder + recorder + fake packet generator
  features/      traffic.py — gap to car ahead, dirty-air exposure
  results.py     headline numbers -> out/results.json
  artifacts.py   per-lap predictions, calibration curves, posteriors -> out/artifacts/
  app/           Streamlit dashboard — reads out/, never fits a model
  planner/       eig.py — FP2 run planner, exact expected information gain
  backtest/      fp_race.py — Tier 3: forecast each race from its practice
  infer/                                         (empty — later phases)
audit/           Phase 0 scripts. Each prints a PASS/FAIL gate.
eval/protocol.md pre-registered metrics, splits, baselines. Frozen.
tests/           recovery tests — simulate, fit, assert truth comes back
```

`tyretrace/schema.py` is the decision that lets three people work in parallel.
Because the simulator and the real ingest emit *exactly* the same 37-column table,
no model can tell them apart — which is what makes the recovery tests meaningful
and lets modelling start before ingest finishes.

---

## Running things

```bash
pip install -r requirements.txt
```

One command runs everything, in dependency order, skipping what is already fresh:

```bash
python -m tyretrace run all       # ~20 min cold, seconds when cached
python -m tyretrace status        # gates from the last run
python -m tyretrace list          # the task graph and what is stale
python -m tyretrace run l3        # one task plus whatever it needs
```

Results land in `out/results.json` (headline numbers, gates) and `out/artifacts/`
(per-lap predictive intervals, calibration curves, posterior draws), so nothing has
to parse stdout.

### Dashboard

```bash
python -m tyretrace app           # http://localhost:8501
```

Eight pages: overview and gate board · an interactive explainer of the
identification problem · race explorer (tyre strategy, lap traces, traffic) ·
track evolution (quali anchor vs the practice confound) · model ladder
(calibration curves, posteriors) · held-out predictions with intervals ·
compound contrasts (E1) · the game rig.

The dashboard only **reads** `out/`. Audits fit and write; the dashboard draws. A
page that ran NUTS on load would take minutes to open and could show a number no
audit ever gated. Run `python -m tyretrace run all` first; a page whose inputs are
missing says which task produces them.

Staleness is deliberately conservative: a task reruns if its outputs are missing, its
own script changed, *any* file in `tyretrace/` changed, or a dependency reran. Nothing
survives a code edit.

Two gates are **expected** not to pass, and the runner says so rather than showing them
as breakage — `a1a2` (A1's premise was overturned by measurement; `a1b` replaces it) and
`l3` (improves CRPS but is not yet calibrated). An unexpected gate is reported loudly
and is the only thing that makes the run exit non-zero.

Individual audits still run standalone if you want one in isolation:

```bash
python audit/a3_synthetic_recovery.py
python audit/l3_dirty_air.py --full
python -m pytest tests/ -q
```

Only needed if you delete the cached parquets in `out/`:

```bash
python -m tyretrace run pull      # ~15 min, refetches 40 FastF1 sessions
```

On Windows set `PYTHONIOENCODING=utf-8` (PowerShell: `$env:PYTHONIOENCODING='utf-8'`)
or polars' table borders crash the console. The runner sets it for its subprocesses.

**Compute, as measured here:** JAX 0.6.2 + NumPyro 0.19 run CPU-only on native Windows —
fine for L0–L3. GPU needs WSL2 with a real distro; only `docker-desktop` is installed, so
`wsl --install -d Ubuntu` comes first. The GPU is an **RTX 4050 Laptop, 6 GB**, not the
RTX 5070 the original stack assumed, and 6 GB will be tight for NUTS over a hierarchical
HMM at L4.

---

## Phase 0 gates

| Check | Gate | Status |
|---|---|---|
| A1 practice decorrelation | τ resolvable from one session | **FAIL — premise overturned** |
| A1b quali anchor | τ correct sign, z > 5 | **PASS** — z ≈ 10.7, 10/10 sessions |
| A2 race conditioning | posterior SD < 0.008 s/lap contrast | **PASS** — 0.0026 s/lap |
| A3 synthetic recovery | coverage in [0.85, 0.95] | **PASS** — 0.885–0.920 |
| A4 2026 scope | scoping number, no gate | 12 of 23 rounds run |
| A5 rig recovery | recover hidden state within 3 SE | **PASS** (fake stream) — needs `--self-check` on a live game |
| L0/L1 ladder | recovery, no divergences, ship gate | **PASS** — L1 improves CRPS 65.6% |
| L2 tau-scale | estimate the split, not assert it | **PASS** — s = 0.87 [0.79, 0.96] |
| L3 dirty air | ship gate vs L2 | **HOLD** — CRPS −4.1% but coverage 0.849 |
| L4 calibration | stint offsets + heavy tails | **HOLD** — best CRPS 0.344, coverage 0.813 |
| L5 variance model | non-constant sigma | **REJECTED** — wrong functional form |
| L6 compromised laps | one-sided noise mixture | **REJECTED** — all 3 variants cover worse (0.76) |
| Baselines (§4) | best rung beats every baseline | **PASS** — all 4 built; L4 beats the published race-sim model by only 1.7% |
| F1 fuel-pace gap | fuel alone predicts unseen race slopes best | **PASS** — gap was a track-evolution attribution |
| A6 ablations | every component earns its place | every built component does |
| L3 after the slope fix | ship gate | **SHIP** on the 10 weekends (CRPS −4.0%, coverage 0.858) — **does not replicate** on 61 new weekends (coverage 0.797, no CRPS gain) |
| D1 source parity | OpenF1 agrees with FastF1 | **PASS** — 40 shared sessions |
| T3 on the lake | pre-registered fuel-only beats conventional | **FAIL** — methods tie on 68 weekends |
| F1 on the lake | fuel alone explains race slopes | **FAIL** — small race track term (s 0.35) earns its place |
| H1 2026 | fuel-only beats conventional under both fuel loads | **FAIL** — flips with the unverified 2026 fuel load |
| T3 FP → race backtest | identified beats conventional and naive after stops | **FAIL — finding**: beats naive by 63%, trails fixed 0.030 s/kg by 3.9% |

### What the audit changed

**The practice decoupling argument is wrong.** It assumed cars carry different fuel
loads at the same session minute. They do — but there is also a strong *common trend*,
because teams run low-fuel quali sims early and high-fuel race sims late. Measured
over 10 weekends: `corr(run start minute, run pace) = +0.54` (IQR 0.34–0.78). A
practice-only fit returns `τ = +0.029 s/min`, the wrong sign — cars get heavier faster
than the track rubbers in.

**Run length is not a usable fuel proxy.** A 15-lap run must be fuelled for 15 laps, so
it looked like an instrument. But the measured coefficient is `+0.22 s` per lap of run
length against `+0.029` predicted by fuel alone — long runs are also driven in
management mode. It proxies programme type, not fuel.

**Qualifying is the fix.** Every car runs minimum fuel across a ~60-minute session, so
within-driver pace against the clock isolates the track:

```
τ = -0.0165 s/min   negative in 10/10 sessions   single-session z = 10.7
robust to tyre age, compound and fresh-tyre controls (-0.0161 to -0.0172)
```

Verified both ways: the simulator reproduces the practice failure exactly (wrong sign
with a fuel programme, correct without), and the quali anchor recovers known truth at
z = 14.8 with 10/10 interval coverage. Supplying τ to the race arm then recovers fuel
pace to within 0.00009 s/lap.

This introduces one assumption the proposal did not have: **track evolution measured in
qualifying must transport to the other sessions.** That is weaker than assuming a fuel
coefficient outright, and unlike that assumption it is checkable against the game rig.

---

## The rig

`tyretrace/rig/` decodes fuel mass and per-wheel wear from the F1 game's UDP
stream. Privacy settings can redact those channels; capture diagnostics count
the cars that actually expose both. Controlled game sessions can expose tyre
cliffs and execute candidate practice programmes.

Recorded laps land in the same canonical schema as everything else with the `truth_*`
columns filled, so the estimator sees only real-world-observable channels and the test is
whether it recovers a state it was never shown. On the fake stream it does: degradation
+0.05514 against 0.05500 true, fuel pace +0.05817 against 0.05775.

**Updated 2026-10-26:** independent fixtures from the official EA 2023/24/25
specifications caught and fixed lap-data offsets that the old self-consistent
fake could not detect. F1 23 needs different offsets because fields changed
in the middle, not only at the end. Completed-lap indexing and stale truth
handling are fixed too. Live game verification is still required. Run

```
python -m tyretrace.rig.record --self-check
```

against a live session before trusting a single recorded lap. If it fails, fix the offset
table — do not loosen the check.

Then use `python -m tyretrace.rig.capture --seconds 900 --out out/rig/live_001.udp`
to retain raw evidence, checksums, provenance, canonical laps and a diagnostic
report. Reanalyse with `--replay out/rig/live_001.udp`. A synthetic run is never
reported as live proof. See [`docs/GAME_RIG.md`](docs/GAME_RIG.md) for setup and
the limits of the truth-blind versus fuel-observed reference comparison.

---

## The model ladder

`tyretrace/models/ladder.py`. Each rung ships only if it beats the one below on
held-out CRPS *and* keeps 90% coverage inside [0.85, 0.95] — `eval/protocol.md` §5.

| | recovery on twin | held-out CRPS | cov90 | gate |
|---|---|---|---|---|
| L0 | all params cover truth | 0.3386 | 0.901 | ship |
| L1 | all params cover truth | **0.1166** | 0.891 | **ship, −65.6%** |

Three geometry bugs were found and fixed getting there, all of which produced
*plausible numbers* rather than errors:

- **Redundant additive degree of freedom** — grand mean, circuit intercepts and
  driver offsets all carried the level, so the chains wandered a ridge (r_hat 2.23).
  Fixed with `ZeroSumNormal` on driver offsets.
- **Circuits partially pooled when they shouldn't be** — they are not exchangeable
  (Monaco vs Spa differ by tens of seconds; measured `sd_circuit` 8.9 against a
  `HalfNormal(3)` prior) and each carries ~1250 laps, so non-centring put the sampler
  on a ridge (ESS 51/1000). Now fixed effects.
- **`target_accept` is not monotone** — 0.95 left 2 divergences, 0.98 clears them,
  0.99 collapses the step size entirely (r_hat 2.23, ESS 3).

### L2 — estimating the split instead of asserting it

L0/L1 leave `lap_coef` fused and can only split it by *declaring* τ. L2 decomposes it
using cross-circuit variation instead:

```
lap_coef_c = -(beta_f * kappa_c  +  s * tau_quali_c)
```

κ and τ are both observed and only mildly collinear (corr 0.55, VIF 1.44), so β_f and
the session scaling `s` are separately identified across ten weekends. Result:

- **fuel pace = 0.0320 s/lap** [0.0298, 0.0341] — identified
- **s = 0.869** [0.785, 0.956], P(s<1) = 0.99 — the race evolves *less* than qualifying,
  as expected once the track is rubbered in

Also delivers estimand **E1** (within-weekend compound contrasts) with no Pirelli map,
since relative labels are unambiguous inside a weekend. 3 of 10 weekends show a
SOFT−HARD contrast whose 90% interval excludes zero.

**Two open problems this exposed.** The identified fuel pace of 0.032 s/lap is roughly
*half* what fuel physics implies (β_f 0.030–0.035 s/kg × κ ~1.8 ≈ 0.055–0.06 s/lap).
Degradation curvature is ruled out (adding age² moves `lap_coef` by 0.0004), and
within-stint residual autocorrelation is only ρ=0.19. So something makes late-race laps
slower than fuel burn alone predicts — tyre/fuel management, engine derate, or late-race
track degradation. None is in the model yet. And converting to s/kg needs κ, which was
assumed as (full tank)/(race laps) — an upper bound. **The s/kg figure is a κ assumption
wearing a posterior**; report s/lap, exactly as protocol §1 says.

### L3 — dirty air, and a prediction that held

`gap_ahead_s` comes free from timing data: order cars by session clock within a lap.
Wake exposure is `exp(-gap/2s)`, bounded and nonlinear, since 0.5 s vs 1.5 s behind
matters far more than 10 s vs 11 s.

**Wake costs 0.453 s/lap** [0.410, 0.499] in full wake — a real, usable number, and 45%
of race laps sit within ~2 s of a car ahead. CRPS improves 4.1% over L2.

But L3 **does not ship**: coverage is 0.849, just under the 0.85 band (L2 is 0.843).
Both models are overconfident on held-out laps. That is the gate working — a better
model that is still miscalibrated is not a shippable model.

It also settled a falsifiable prediction. The field spreads out over a race
(corr(race lap, gap) = +0.37), so early laps are dirtier and slower — another *negative*
contribution to `lap_coef`. Controlling for it therefore had to push the identified fuel
pace *away* from physics, not toward it. It did, by −0.0020 s/lap. **Dirty air is real
and worth modelling, and it is not the missing slowdown.**

Still unexplained: identified fuel pace 0.030 s/lap against ~0.055–0.060 implied by
physics. Ruled out so far — degradation curvature (age² moves `lap_coef` by 0.0004),
residual autocorrelation (ρ=0.19), and now dirty air. Remaining: management/engine
derate, late-race track degradation, or β_f genuinely below the convention for these
cars, which would be a result rather than a bug.

### L4 / L5 — calibration, and a rejected hypothesis

No rung has shipped yet, because none is calibrated. Diagnosing the residuals rather
than guessing found two causes, and L4 fixes both:

- **a per-stint offset carrying 52% of residual variance** (between-stint sd 0.457 s vs
  within-stint 0.442 s) — each stint starts at its own fuel level, on its own set, with
  the driver in its own mode
- **heavy right-skewed tails** — excess kurtosis +1.54, `|z|>3` at 1.12% against a
  Gaussian's 0.27%. Fitted `nu = 2.6`, so the tails are genuinely heavy

L4 gets the **best CRPS (0.344) and MAE (0.463) of any rung** and still under-covers at
0.813, so it does not ship.

Two design traps worth recording. Per-stint offsets are in **direct tension with the
pit-stop identification** — races separate fuel from degradation precisely *because*
stint levels are informative, so free offsets eat the signal; the prior is deliberately
tight (HalfNormal(0.5) against a measured 0.46 s). And offsets must be demeaned *within
circuit*, since stints nest inside circuits; without that the first attempt hit r_hat
3.04 with σ = 52 s. A heavy-tailed likelihood also opens a degenerate "everything vague"
mode that a badly-started chain falls into, so fits use `init_to_median`.

**L5 is a recorded negative result.** It made σ log-linear in tyre age, and made coverage
*worse* (0.759) with a negative slope — σ shrinking with age. Measuring σ by age with the
lap trim off shows why:

```
age  1-10   sd 1.074     warm-up, out-laps
age 11-20   sd 0.478     steady state
age 21-60   sd 0.628     degradation
```

That is **U-shaped**, not monotone. A log-linear scale can only rise or fall, so it fit
the large early drop and gave old tyres narrower intervals — worsening the problem it was
built to fix. The next attempt is a non-monotone scale or the regime model, where warm-up
and cliff are distinct states with their own variances.

### Baselines, L6, and a convergence warning

**Baselines** (`tyretrace/models/baselines.py`, `audit/b1_baselines.py`), scored on
the ladder's own held-out laps:

| | CRPS | cov90 |
|---|---|---|
| B1 naive — a line per stint | 0.785 | 0.41 |
| B3 gradient boosting | 0.397 | 0.79 |
| L2 | 0.381 | 0.84 |
| **B2 conventional — assumed 0.030 s/kg** | **0.371** | 0.79 |
| L4 | 0.344 | 0.81 |

L4 beats all three, but **the conventional fuel correction beats L2**. That is
not a contradiction. Forecasting the rest of the *same* race needs only the fused
slope, and a free degradation curve absorbs a wrong fuel assumption. What the
wrong assumption corrupts is the *reported degradation*: `tests/test_baselines.py`
shows a 0.015 s/kg fuel error leaving the fit almost untouched while the reported
degradation moves from 0.060 to 0.089 on one set and **−0.021** on the next. So this
comparison measures forecasting skill, not the split. The FP → race backtest is
where the split gets tested.

**L6 is a recorded negative result.** L4's held-out misses are one-sided (12% of
laps slower than its 95th percentile, 7% faster than its 5th), worst on old tyres.
A clean/compromised-lap mixture, with compromised laps allowed to become likelier
with tyre age, was tried in three pre-declared variants. All covered *worse*
(0.755–0.763 vs 0.812), and the age effect came out negative. So the lopsided misses
are not lap-level noise. The next suspect is the mean on old tyres (curvature or a
cliff), which is the regime model's job.

**L3–L5 do not fully converge.** r_hat is 1.05–1.10 with ESS ~25 on `beta_f`, the
stint offsets and degradation. Centring lap number within circuit was tried and did
not help. It is the stint-offset vs fuel trade-off described under L4. Forecasts
are stable across reruns (coverage 0.812 / 0.813 / 0.813), so the ship-gate verdicts
stand; **fuel estimates from L3+ are not reliable — quote L2's** (r_hat 1.007). The
L4 audit now reports each rung's r_hat and flags unconverged fits.

### Tier 3 — forecasting the race from practice

`tyretrace/backtest/fp_race.py`, `audit/t3_backtest.py`, dashboard page **FP → race
backtest**. Every earlier test forecast the end of a race from its start, which only
needs the fused slope. Here each race is forecast from its own weekend, rolling origin.
The forecast uses W's practice for degradation, W's qualifying for track evolution,
races **before** W for the fuel effect, and each driver's first 5 green laps as a pace
anchor. It then forecasts every later lap. The laps after a pit stop are where fuel
has to be separated from degradation. 7 weekends, 6,475 laps, 4,743 after a stop.

| method | CRPS after a stop | bias after a stop |
|---|---|---|
| naive — practice slope is degradation | 1.606 | +1.87 s |
| no-track (ablation) | 0.679 | +0.49 s |
| identified | 0.589 | +0.19 s |
| **conventional — fixed 0.030 s/kg** | **0.566** | −0.02 s |

**v1 failed for a reason worth recording.** Practice runs are short (4–7 laps) and
mixed with push and cool-down laps. Pooling laps as if independent made three erratic
runs look precise: Montréal HARD came out at −0.23 s/lap and Monza SOFT at +0.41. A
missing compound borrowing the mean of the others spread the outlier further. CRPS
was 1.43. v2 fits a slope per run, pools runs by random effects so disagreement
widens the interval, and updates a degradation prior built from earlier races' pit
stops. CRPS dropped to 0.54. That diagnosis came from v1's failure, not from v2's
gate: v2 still fails.

**Why identified trails conventional.** The race lap slope (fuel plus track) forecast
for each weekend, against what that race did, measured afterwards and never fed back:

| slope forecast | MAE | bias |
|---|---|---|
| conventional, fixed 0.030 s/kg | 0.0101 s/lap | +0.0015 |
| identified, learned from earlier races | 0.0133 s/lap | +0.0072 |
| mean of earlier races, no decomposition | 0.0175 s/lap | +0.0033 |

Decomposing by burn rate and track evolution beats plain averaging, but the learned
fuel effect is **too small**: races speed up more than it predicts. This is the first
out-of-sample evidence on the open fuel-pace gap (identified ~0.03 vs physics
0.055–0.06 s/lap). It says the **race identification is biased low**, not that physics
is wrong. Tightening the fuel prior toward 0.030 s/kg would pass the gate. It would
also be tuning on the test set, so it was not done. The bias is the next thing to explain.

### The run planner

`tyretrace/planner/eig.py` and the dashboard's **Run planner** page: which FP2 runs
would teach the most about degradation? Laps follow a linear-Gaussian model from the
team's side of the wall (fuel known, a free intercept per run), so expected
information gain is **exact**: no Monte Carlo. It's checked against brute-force
simulation, and the optimiser against exhaustive search.

Within a run, fuel, tyre age and track all move in lock step, so a run only reveals
their fused slope. The one move that breaks it is a **tyre change without refuelling**.
On equal session time, with the qualifying simulation kept in both programmes:

| | vs a typical FP2 |
|---|---|
| tyre changes not allowed | **+0%**: the planner rediscovers the standard programme |
| tyre changes allowed | **+21%**: one 20-lap run, medium → hard at half distance |

Two optimiser bugs were found by making the comparison fair. Rate-greedy alone lost by
8%, and greedy plus local search lost by 10%. Both are fixed by a beam search, with
regression tests. v1 assumes linear degradation (no cliff), one car and no tyre
allocation limit.

### What the earlier L1 fit says

Fitted on 9,994 race laps: `lap_coef = -0.0496`, r_hat 1.010, zero divergences. That
coefficient is **fused**: `fuel_pace = -lap_coef - tau`. So every gram of fuel
attributed depends on the τ carried over from qualifying — an assumption, not a
measurement. Its cost, quantified:

| τ carried into the race | implied β_f |
|---|---|
| none | 0.0284 s/kg |
| half the quali rate | 0.0217 s/kg |
| full quali rate | 0.0150 s/kg |

**Sign-convention note.** `track_evolution` reports τ *negative* for an improving track
(it is a regression slope on the clock); the sim and models enter it as `- tau * lap`,
so improving is *positive* there. An earlier version of this table passed the measured
value straight through and was sign-flipped. `TrackEvolution.tau_model()` is now the
only sanctioned bridge, with a regression test. L2 supersedes this table anyway by
estimating the scaling rather than sweeping it.

---

## Closing the core (October 2026)

### The fuel-pace gap, solved

`audit/f1_fuel_gap.py`. A race shows only the fused slope, fuel plus race track
evolution. L2 split it as `-(beta_f*kappa) - s*tau_quali` and found s = 0.87: the
race track keeps improving at 87% of qualifying's rate. Whatever L2 credits to the
track, it subtracts from fuel.

L2 had no room for circuit-to-circuit scatter, so ten races' tiny within-race
standard errors pinned s as if the formula held exactly. With between-race scatter
allowed:

| | estimate |
|---|---|
| fuel effect, fuel alone | **0.0293 ± 0.0024 s/kg** — the physics value |
| race/quali scale s, with fuel | 0.33 ± 0.55, 90% [−0.58, 1.24] — **not identified** |

Leave-one-race-out prediction of each race's slope: physics alone MAE 0.0116,
fitted fuel-only 0.0131, L2's fuel + track 0.0137, plain mean 0.0177. No late-race
slowdown either (second half vs first half +0.006 ± 0.007 s/lap). **The gap was
never about fuel.** It came from overconfidently crediting race track evolution.

Two consequences. `tyretrace/models/ladder.py:_lap_coef` now gives every weekend a
partially pooled slope deviation, so L2–L6 can no longer be overconfident about s.
Refitted, L2's fuel pace is 0.041 s/lap, 90% [0.026, 0.055], and s is 0.61, 90%
[0.09, 1.11]: honest about what ten races cannot pin. The ladder improved too.
**L3 ships**, the first rung to: CRPS 0.346 (−4.0% vs L2), coverage 0.858, inside
the band, converged (r̂ 1.01), and better than all four baselines. L4 (0.354,
coverage 0.807, r̂ 1.10) does not. The fix was motivated by analysis of the same
ten races, so it was re-checked out of sample: on **61 lake weekends** that
analysis never touched, **it does not replicate**. L3's CRPS is 0.626 against L2's
0.624, and coverage is 0.797, overconfident again. The ten-weekend ship was real
on those ten weekends and does not generalise. Calibration at scale remains the
open problem. And a **fuel-only** forecasting method
was **pre-registered on 2026-10-21, before the lake was pulled**, so the lake gives
it a clean out-of-sample test.

### B4 — the published baseline

`tyretrace/models/baselines.py:published`, after Heilmeier et al.'s race simulation
(TUM): fuel mass times a fixed sensitivity, plus a logarithmic tyre curve per
compound. Verify the exact equations against the paper before citing. It is the
strongest baseline: CRPS 0.350 against L4's 0.344, only 1.7% behind. All four
protocol baselines now exist.

### Post-race validation

`tyretrace/backtest/postrace.py`, `audit/p1_postrace.py`. The tool the submission
promised: each lap's forecast error is split into **slope** (fuel + race track,
which a race cannot separate), **tyres**, **traffic** and **noise**. The split is
**exact**: the parts are checked to sum to the error on every lap, and a test
proves that check fires. On the backtest, the largest systematic error for every
method was traffic (+0.11 s): each driver's anchor laps are the crowded opening
laps.

### Traffic-aware anchors

So the anchor is now de-trafficked with the wake cost learned from earlier races,
and the expected traffic of later laps is added back. Future gaps are unknown
before the race, so the forecast carries their expectation. On the ten weekends,
identified CRPS after a stop fell from 0.589 to 0.569 and its bias from +0.19 s to
+0.06 s. That puts it within 0.7% of conventional, down from 3.9%. Pre-registered
for the lake run.

### Recalibration — tried, not adopted

Rolling empirical recalibration reshapes each forecast's interval using
standardised errors from earlier races only, asymmetry included. Coverage rose
(0.78 → 0.81), but CRPS worsened by 8.8%, so it does not ship. Most of the error is
a whole-race shift, not independent lap noise, and reshaping each lap's spread
cannot fix a shift every lap of a race shares.

### Ablations

`audit/a6_ablations.py`, protocol §6. Remove each component and the CRPS change
after a pit stop:

| component | change without it |
|---|---|
| degradation prior (earlier races' pit stops) | +192% |
| hierarchy (L0 → L1, twin) | +190% |
| lap filtering | +31% (also changes which laps are scored) |
| random-effects pooling of practice runs | +18% |
| track evolution | +15% |
| stint offsets + heavy tails (L3 → L4) | +6% |
| dirty air (L2 → L3) | +4% |
| telemetry, regime switching, censoring | not built |

Every built component earns its place.

### The data lake — and why it is OpenF1 now

`tyretrace/data/openf1.py`, `tyretrace/data/bulk.py`. F1's live-timing archive,
which FastF1 reads, now returns **401 Unauthorized**. The original ten weekends
remain as pulled. Everything since 2023 comes from OpenF1, in the same canonical
schema: **71 dry race weekends, 93,861 race laps** (`python -m tyretrace run lake`,
about an hour, resumable). 2018–2022 is not publicly available with tyre data.

`audit/d1_source_parity.py` checks the two sources on all 40 sessions they share.
Lap time agrees to the millisecond on 99.0% of laps, fresh-set tyre age on 99.4%,
per-lap age increments on 99.5%, and the long-run laps the models see overlap
98.7%. The check found two adapter bugs, both fixed and covered by tests. Practice
yellow and red flags carry no lap number and had been ignored (Baku FP1 2023 read
all green). And OpenF1 sometimes starts a race stint one lap early, so boundaries
now snap to the pit-out lap. One difference is real and documented: on **used**
sets OpenF1's tyre age reads about 3–4 laps lower.

Wet and mixed races are excluded everywhere (`fp_race.dry_weekends`): on a drying
track lap times fall for reasons that are neither fuel nor tyres. Four such races
had taken the cross-race slope scatter from 0.017 to 0.152 s/lap. All ten original
weekends are dry, so nothing recorded there changes.

### What 68 more weekends changed

**The fuel-pace gap, refined.** On 71 dry races, fuel alone gives
**0.0324 ± 0.0011 s/kg** — the physics value. A small race track evolution does
earn its place, though: s = 0.35 ± 0.14 of the qualifying rate, against the 0.87
L2 used to claim. With it, fuel is 0.027 s/kg. On ten races, s was not identified
at all. With enough races it is, and it is about **a third** of what L2 asserted.
That overstatement was the gap.

**The pre-registered lake backtest fails, and here is the honest reading.** On 68
dry weekends (39,886 laps after a pit stop), CRPS after a stop is: conventional
0.657, ours without the track term 0.657, identified 0.668, fuel-only 0.674, and
naive 1.404. Paired per weekend, the fuel-corrected methods are **statistically
tied**: identified vs conventional t = +0.52, ours better on 54% of weekends. All of
them beat naive decisively (t = −9.6, on 88% of weekends). Our race-slope forecast
is now the best of the three (MAE 0.0121 vs conventional 0.0126 vs plain mean
0.0153).

So for 2023–2025 the paddock's 0.030 s/kg **is right**. Our data recovers it
independently (0.0324 ± 0.0011), so the assumption no longer needs to be taken on
faith. What limits forecasting a race from practice is **degradation estimated
from practice**. Over 68 weekends the post-race split puts its error at rms
0.92 s, against 0.50 s for the fuel and track slope, 0.34 s for traffic and 0.69 s
for lap noise. That, not the fuel split, is the next thing to improve.

**A finding from ten weekends that did not survive 68.** On the ten, the practice
track-evolution term earned its place (+15%). On 68 weekends, dropping it improves
CRPS by 2.5%. It is recorded as a reversal, not deleted.

### 2026 — the new regulations

`audit/h1_2026.py`. Fifteen dry 2026 races, each forecast from its own weekend
with every earlier race as history. 2026 cars are widely reported to start with
about 70 kg of fuel, a figure not verified here, so every result is run under 70 kg
and under 110 kg:

| method | CRPS, 70 kg | CRPS, 110 kg |
|---|---|---|
| **identified** | **0.580** | 0.572 |
| fuel-only (gated) | 0.623 | 0.576 |
| conventional | 0.630 | 0.569 |
| naive | 1.336 | 1.336 |

The gate fails because the verdict flips with the fuel load. Two results stand
regardless. The identified method is the most robust to the assumption, and it is
best by 8% under the new rules. And the races themselves, which need no fuel-load
assumption, **contradict the 70 kg figure with unchanged fuel sensitivity**. 2026
races speed up by −0.052 s/lap against −0.057 for 2023–25 (difference
+0.006 ± 0.005). At 70 kg the slope should be about −0.037. Either 2026 races
carry well over 70 kg, or each kg costs more time in the new cars. Telling those
apart needs the true fuel load.

## Honest scope

No team-confidential tyre temperatures, pressures or fuel logs. Fuel mass is
**latent throughout** — inferred with a prior, never observed. The method works
without those inputs and states how much it does not know.
