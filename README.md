# Tyre Trace

Causal tyre-degradation intelligence — recovering true wear rates from confounded
practice data.

**Status: Phase 0 (feasibility audit).** No model is being built yet, on purpose.
The audit exists to kill the fatal assumptions cheaply, before any expensive work
depends on them.

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
| Baselines (§4) | best rung beats every baseline | **PASS** (3 of 4 built) — but see caveat below |

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

`tyretrace/rig/` decodes the F1 game's UDP stream, which carries **true fuel mass in kg
and true per-wheel tyre wear for every car on track** — so you run an AI race and harvest
~20 cars of ground truth without driving. It is the only place a tyre cliff is observable
on demand, and the only place the run planner's counterfactual can actually be executed.

Recorded laps land in the same canonical schema as everything else with the `truth_*`
columns filled, so the estimator sees only real-world-observable channels and the test is
whether it recovers a state it was never shown. On the fake stream it does: degradation
+0.05514 against 0.05500 true, fuel pace +0.05817 against 0.05775.

**The offsets are written from the published spec, not verified against a running game
here.** A wrong offset does not crash — it yields plausible garbage, the worst failure
mode for a ground-truth harness. Three defences: per-car stride is derived from packet
length rather than assumed; only leading offsets are read (EA appends new fields at the
end of each block); and `validate()` checks physical plausibility. Run

```
python -m tyretrace.rig.record --self-check
```

against a live session before trusting a single recorded lap. If it fails, fix the offset
table — do not loosen the check.

Game setup: Telemetry → UDP ON, format 2023/2024/2025, port 20777, 20 Hz+, and **"UDP Send
All Cars" ON** — without it only the player's car reports and the 20-cars-per-session
advantage is lost.

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

## Honest scope

No team-confidential tyre temperatures, pressures or fuel logs. Fuel mass is
**latent throughout** — inferred with a prior, never observed. The method works
without those inputs and states how much it does not know.
