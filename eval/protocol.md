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
