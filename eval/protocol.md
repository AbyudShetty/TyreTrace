# Evaluation protocol — pre-registered

**Frozen 2026-08-30. Do not edit after the first backtest runs.**

A project whose thesis is *"everyone else quietly assumes their answer"* cannot pick
its metric after seeing the scores. This file is written before any real fit. If
something here turns out to be wrong, add an amendment section with a date and a
reason — never a silent edit.

---

## 1. Estimands

Two, with different assumption loads. Always reported separately, never blended.

| Estimand | Definition | Assumptions it needs |
|---|---|---|
| **E1 — compound contrast** | δ_A − δ_B, s/lap, between two compounds in the same session | Fuel coefficient is compound-invariant. No cross-session transport. |
| **E2 — absolute degradation** | δ_c, s/lap, for one compound | E1's assumption *plus* the pit-stop design, *plus* β_f and D_c transporting between practice and race. |

Reported in **s/lap**. The s/kg form is derived, and must carry the propagated
uncertainty on κ (fuel burn per lap) from whatever external source supplies it.
`β_f·κ` is what the data identifies; `β_f` alone is not.

## 2. Splits

**Rolling origin, strictly causal.** For a target weekend *W*:

- Priors and pooled parameters may use only weekends strictly earlier than *W*.
- Within *W*: fit on FP1–FP3 only, predict the race.
- No global fit over all seasons, ever, at any stage of model selection.

Held-out set for model selection is a **contiguous earlier block**, never a random
lap-level split. Random splits leak: laps from the same stint are near-duplicates.

**2026 is the headline case.** 12 rounds available as of 2026-08-30. Priors from
2018–2025, evaluate on 2026 round *n* using rounds < *n*. This is the
sample-efficiency claim and it must be run exactly this way.

## 3. Metrics

| Metric | Role | Why |
|---|---|---|
| **CRPS** | Primary | Scores the whole predictive distribution. Point-accuracy alone cannot distinguish a confident wrong answer from an honest uncertain one. |
| **Interval coverage** | Gate | Report the full calibration curve (50/80/90/95), not one level. |
| **Stint-pace MAE** (s/lap) | Legibility | The number a strategist recognises. Reported, never optimised against. |
| **Optimal-stop-lap error** (laps) | Decision | The actual job. A model that improves CRPS but not this has not helped anyone. |

## 4. Baselines

All four run on every comparison. Beating only the first proves nothing.

1. **Naive** — within-stint OLS of lap time on lap-in-run.
2. **Conventional** — subtract 0.030 s/kg/lap assumed fuel effect, fit quadratic to residual.
3. **ML** — gradient boosting on lap-level features, same inputs, no causal structure.
4. **Published** — one academic tyre-degradation model, cited, reimplemented faithfully.

## 5. Ship gate for a model rung

A rung (L0…L5) ships only if, against the rung below:

- held-out **CRPS improves**, and
- **coverage stays inside [0.85, 0.95]** at the 90% level.

A rung that improves the point estimate while breaking calibration is a
**regression**, not an improvement, and gets reverted.

## 6. Ablations

Every component is removed individually and the full metric set re-reported:
hierarchy, track-evolution term, telemetry channel, regime switching, censoring
model, lap-filtering on/off. Published as one table. A component that cannot show
its contribution does not ship.

## 7. Ground-truth tiers

| Tier | Source | What it proves |
|---|---|---|
| T1 | Synthetic twin | Recovery under known truth with calibrated intervals; bias of the conventional fix as a function of its own assumption error. |
| T2 | Game UDP rig | Recovery of a hidden state the estimator is never shown (true fuel kg, per-wheel wear). The only place the cliff is observable on demand, and the only place the run planner's counterfactual can actually be executed. |
| T3 | Real backtest | FP→race across 190+ weekends with the splits in §2. |

## 8. What would falsify the approach

Stated in advance, so it cannot be renegotiated later:

- **Rig recovery fails** — the estimator cannot recover true wear on data where truth
  is known and the inputs are real-world-observable. The method does not work.
- **Coverage is systematically wrong** on T3 — intervals do not mean what they say,
  which destroys the central "knows what it does not know" claim.
- **No ablation gap** — the causal structure performs no better than baseline 3, in
  which case the identification argument is not buying anything in practice.
- **E2 intervals exceed the compound spread** on 2026 data — the absolute estimand is
  not resolvable at the sample sizes that exist, and only E1 should be claimed.

---

## Amendments

*Append dated entries below. Never edit above this line.*

### 2026-08-30 — identification mechanism for track evolution

**What changed.** §1's estimand E2 lists "β_f and D_c transporting between practice and
race" among its assumptions. The route by which track evolution was to be identified has
changed, so that assumption is restated rather than removed.

**Why.** The original plan identified track evolution inside practice, on the argument that
cars carry different fuel loads at the same session minute. Measured over 10 weekends,
`corr(run start minute, run pace) = +0.54` — teams run low-fuel quali sims early and
high-fuel race sims late, so fuel state trends with the session clock and rides the same
index as track evolution. A practice-only fit returns τ = +0.029 s/min, the wrong sign.
Run length was tested as a fuel instrument and rejected: measured +0.22 s per lap of run
length against +0.029 predicted by fuel alone, because long runs are driven in management
mode.

**Replacement.** Track evolution is now estimated from qualifying, where every car runs
minimum fuel: τ = −0.0165 s/min, negative in 10/10 sessions, single-session z = 10.7,
robust to tyre-age/compound/fresh-tyre controls.

**New assumption, added to E2.** Track evolution measured in qualifying transports to the
other sessions of the weekend. This is checkable against the T2 rig and must be reported
as a sensitivity, not assumed silently.

**No change** to metrics, splits, baselines, ship gates, or the falsification conditions
in §8.

### 2026-10-21 — data source, fourth baseline, and two pre-registered methods

**Data (§2, §7 T3).** F1's live-timing archive, which FastF1 reads, now answers
401 Unauthorized. The original ten weekends stay as pulled (`out/audit_*`). Every
weekend from 2023 onward now comes from OpenF1 into `out/lake/`, in the same
canonical schema. Before any lake lap was used, `audit/d1_source_parity.py`
compared the two sources on every shared session. The gate checks lap time,
compound, fresh-set tyre age and per-lap age increments, plus long-run overlap.
The one systematic difference is reported, not hidden: tyre age on USED sets reads
about 4 laps lower in OpenF1 (a definitional difference; slopes are unaffected).
**2018–2022 is not available**: OpenF1 starts in 2023, and Ergast carries no tyre
data. The "190+ weekends" in §2 and §7 is therefore not met.

**Baselines (§4).** Baseline 4, "one academic tyre-degradation model", is the
race-simulation lap model after Heilmeier et al. (TUM): fuel mass times a fixed
sensitivity plus a logarithmic tyre term per compound. Its exact equations must
be checked against the paper before it is cited.

**Pre-registered before the lake was pulled.** Two changes motivated by analysis
of the original ten weekends, so their test must be on data those analyses did
not see:

1. *fuel-only*: the fuel effect is learned from earlier races, with no race
   track-evolution term. Motivation: `audit/f1_fuel_gap.py` shows the race/quali
   evolution scale is not identified once between-race scatter is allowed.
2. *traffic-aware anchors*: the anchor laps' dirty air is removed and the
   expected traffic of later laps is added. Both are learned from earlier races.
   Motivation: `audit/p1_postrace.py` found anchor traffic to be the largest
   systematic error.

The lake backtest's gate judges *fuel-only* with traffic-aware anchors. The
ten-weekend backtest keeps its original gate and its recorded FAIL.

**No change** to metrics, splits, ship gates, or the falsification conditions in §8.

### 2026-10-26 — predictive uncertainty repair and temporal confirmation

The current uncertainty work distinguishes uncertainty in a population mean
from variation of a new weekend, and uncertainty in a practice slope from its
transfer to race conditions. These are predictive discrepancies, not new
claims that fuel mass or absolute degradation is directly observed.

Development uses 2023–2024 forecast residuals, with 2025 for development
validation. Before inspecting new 2026 uncertainty scores, freeze the selected
algorithm and settings in `docs/UNCERTAINTY.md`. Evaluate 2026 once using rolling
history strictly before each target weekend. Previous project work has already
seen aggregate 2026 results; this is retrospective temporal confirmation, not
an untouched prospective experiment. An actually new season is still required
for prospective replication. Changes after the confirmation must be labelled
development and cannot reuse it as an untouched gate.

Report original and repaired distributions on identical laps: CRPS, MAE,
coverage at 50/80/90/95%, interval width, before/after-stop coverage, per-weekend
results, and uncertainty of the paired weekend improvement. Keep original
negative results and gates. A repair ships only if CRPS improves and 90%
coverage is in [0.85, 0.95], including post-stop laps. The requirement is not
relaxed by widening intervals. Calibration of predictive lap times must not be
described as calibration of latent degradation credible intervals.

The operational Tier 3 forecast uses practice, qualifying and each driver's
first five eligible green race laps as pace anchors. It conditions on the
observed tyre strategy for evaluation; it is not an FP2-only pre-race strategy
forecast. A separately reported early-race-to-late-race diagnostic likewise
cannot be presented as Tier 3. No change to the original metrics or ship band.

### 2026-10-27 — completion candidates and baseline extension

Implement and evaluate robust within-run practice slopes and a constrained
left-to-right pace-regime HMM (warm-up/steady/slowdown). State labels describe
latent pace regimes, not verified physical tyre states. Select two versus three
states by held-out sequence likelihood on practice only. Target practice is
available, target race outcomes are not available for estimation except the
five declared anchor laps. Model-selection settings are development choices;
all lake scores are retrospective because previous work inspected these years.

Extend rolling comparisons to include a gradient-boosting residual predictor
and a source-form logarithmic tyre-curve baseline. Both receive the same
available history, practice, qualifying, anchor laps and conditioned race tyre
strategy as existing methods. Predictive spread must be learned before the
target race; no target outcome may tune hyperparameters or intervals. Report
the exact implementation scope instead of claiming a full published simulator.

Nomination lookups require dated primary sources and are keyed by year and
event. Unverified nominations remain explicitly unresolved; never silently
reuse the old C3 placeholder as an observed physical compound. Construction
identifiers across seasons are not assumed exchangeable.

Original metrics, temporal causality, CRPS/coverage gate and negative results
remain unchanged. Candidate implementation is complete even if its gate is
HOLD; it does not become the selected predictor merely by being more complex.
## Amendment — 2026-10-27: retrospective selection and conventional curvature

The rolling comparison now includes a conventional quadratic tyre-age curve
with fuel sensitivity fixed at 0.030 s/kg. Unknown practice run levels are
centred out; the common weak slope prior and noise use earlier weekends. This
is reported separately from the original linear conventional forecast, whose
results are preserved. Source-form published curves remain fitted lap models,
not the entire cited race simulator.

Review also found that the existing long-run eligibility filter trims and
accepts whole stints using their observed lap-time distribution. That affects
which opening laps can serve as anchors and which laps are scored. The current
experiments therefore measure retrospective conditional prediction on this
eligible population. They do not establish a fully prospective, online causal
selection procedure. Current known tyre strategy/distance and qualifying/five
race-anchor inputs must remain explicit. A strict online eligibility study
needs a separately frozen filter and fresh observations; past scores cannot be
renamed as that study. Method selection uses 2025, while the already inspected
2026 check is retrospective and cannot be called untouched confirmation.

The flag-only implementation is now included as `online_eligibility`: retain
valid green non-pit laps after lap one without whole-stint pace/variance/length
selection. Practice is fully observed before the forecast, so its run filtering
is unchanged. Its separately calibrated ML/identified/quadratic comparison is
an expanded-population sensitivity on inspected data, not fresh confirmation.

## Amendment — 2026-10-27: isolated unseen Austin holdout

OpenF1 metadata lists the completed 2026 Austin weekend outside the original
84-weekend lake. Before downloading its lap timing, freeze predictor/scoring
source hashes, original lake hashes, target identity, the 2025-selected ML
method, 70 kg primary 2026 fuel assumption and flag-only eligibility. Keep this
new data separate from the original lake. Compare identified, naive, quadratic
conventional, ML and source-form published forecasts on identical laps; fit
each Student-t calibrator only on its original earlier flag-only forecasts.
The ML claim passes this holdout only if its CRPS beats every comparator and
its marginal 90% coverage is within [0.85,0.95]. Report all scores regardless.
This is one unseen offline weekend, not a forecast issued before the event or
statistically sufficient prospective replication. Do not retune on this result.
