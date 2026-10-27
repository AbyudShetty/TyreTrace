# Tyre Trace: saved research results

Generated 2026-10-27T07:07:28+00:00 from
`out/results.json`. Original failed experiments remain in the artifact folders.
This report describes the current evidence, not the historical completion claims.

## Delivered software

The uncertainty layer, immutable UDP capture/replay and layout checks,
conditional strategy diagnostic, source-form published baseline, robust practice
candidate, two/three-state HMM candidate and rolling ML comparison are built.
The conventional quadratic baseline is evaluated separately from its historical
linear approximation. The nomination registry covers **84
of 84 cached race weekends** using dated primary sources.
New ingests use real mapped construction or null; legacy cached C3 placeholders
are never trusted for physical-compound pooling.

React has nine working pages: overview, identification explainer, race explorer,
track/compounds, model evidence, predictive intervals, backtest, planner and rig.
They read exported artifacts. Opening a page does not run samplers or pull data.
Tables and CSV alternatives accompany the research views. A timed guided tour
is available at `/demo.html`; see [the demo script](DEMO.md).

## Rolling forecast comparison

The displayed 2026 block has 14 forecast weekends, 11,560 eligible scored laps
and 7,251 laps after a stop. Every method uses the same target observations.
Each predictive calibration is fitted separately on original forecasts from at
least eight strictly earlier weekends, with equal total weight per weekend.
All values below are calibrated marginal lap-time distributions.

| Method | All-lap CRPS (s) | 90% coverage | Post-stop CRPS (s) | Post-stop coverage |
|---|---:|---:|---:|---:|
| conventional | 0.54027 | 92.9% | 0.57594 | 92.3% |
| conventional_quadratic | 0.72852 | 93.4% | 0.67514 | 93.6% |
| identified | 0.51749 | 93.6% | 0.54653 | 93.1% |
| ml | 0.50944 | 89.8% | 0.53608 | 89.6% |
| naive | 0.85973 | 96.9% | 1.03805 | 98.2% |
| published | 0.61535 | 88.6% | 0.65532 | 87.7% |
| regime_practice | 0.60481 | 93.0% | 0.59953 | 93.8% |
| robust_practice | 0.60158 | 92.8% | 0.60080 | 93.5% |

The method selected using the 2025 development block is
**ml**. The already inspected 2026
block is a retrospective check. A pooled SHIP on this data does not establish a
statistically decisive gain across independent weekends. For ML versus
identified, inspect the paired-weekend bootstrap intervals in
`out/artifacts/rolling_models/comparison.parquet`; the small 2026 improvement's
interval includes zero. Robust/HMM candidates do not earn adoption merely by
being implemented. Their negative results are retained.

The ML result is material to the original falsification criterion: causal
structure has not demonstrated superiority over the strongest unconstrained
predictor in this experiment. The identification argument can remain useful
without claiming the best predictive performance.

HMM was applicable on 57 of 68 forecast weekends; 54 fits converged. Selected state counts: {2: 42, 3: 15}. Nonconverged fits remain flagged in the diagnostics.

## Flag-only eligibility sensitivity

This separate 2026 population removes whole-stint pace and run-length trimming
from race eligibility and opening-anchor selection. It retains the declared
observed strategy/distance and retrospective dry-race restriction. Calibrators
are fitted separately for this population using strictly earlier weekends.
The original and expanded populations have different denominators; their CRPS
values are not a paired comparison on identical laps.

Identified 90% intervals cover 81.4% of this 2026 population. Inspect the method rows against the coverage band; completing this audit does not imply its models ship. The original filtered-population SHIP cannot establish
reliable operational forecasting without retrospective pace selection.

| Method | Subset | CRPS (s) | 90% coverage | Scored laps |
|---|---|---:|---:|---:|
| identified | all | 2.85961 | 81.4% | 11947 |
| identified | after_stop | 3.13753 | 80.6% | 7860 |
| ml | all | 3.49126 | 72.6% | 11947 |
| ml | after_stop | 3.82667 | 71.7% | 7860 |
| conventional_quadratic | all | 2.98979 | 81.2% | 11947 |
| conventional_quadratic | after_stop | 3.25230 | 80.2% | 7860 |
| published | all | 3.19009 | 83.2% | 11947 |
| published | after_stop | 3.45183 | 81.5% | 7860 |
| naive | all | 3.10241 | 84.6% | 11947 |
| naive | after_stop | 3.49936 | 84.8% | 7860 |

## Isolated Austin 2026 holdout

The original 84-weekend lake remains unchanged. Predictor, protocol and input
hashes were recorded in `out/fresh_holdout/freeze.json` before downloading the
new weekend's lap timing. The isolated practice, qualifying and race files have
their own checksums. Sources are verified again before scoring. The primary
fuel assumption is 70 kg; observed tyre strategy/distance and five green race
anchors remain declared inputs. No method was retuned on the outcome.

The 2025-selected ML method's frozen criterion is **HOLD**. It must beat every comparator on CRPS with 90% coverage in [0.85,0.95]. This is one independent, previously unseen offline weekend. The event predates the algorithm freeze; it is not a forecast issued before the race or a statistically sufficient replication.

| Method | Subset | CRPS (s) | 90% coverage | Scored laps |
|---|---|---:|---:|---:|
| identified | all | 0.68538 | 91.0% | 822 |
| identified | after_stop | 0.63130 | 94.7% | 566 |
| naive | all | 0.81109 | 96.6% | 822 |
| naive | after_stop | 0.83171 | 99.8% | 566 |
| conventional_quadratic | all | 0.60313 | 96.5% | 822 |
| conventional_quadratic | after_stop | 0.57241 | 96.6% | 566 |
| ml | all | 0.77160 | 99.1% | 822 |
| ml | after_stop | 0.69948 | 100.0% | 566 |
| published | all | 1.16448 | 84.5% | 822 |
| published | after_stop | 1.02218 | 89.8% | 566 |

## Scope and limitations

- **E1** is a within-weekend compound contrast in s/lap. **E2** is absolute
  degradation in s/lap and requires extra fuel/track transport assumptions.
  Derived s/kg values require a burn-rate assumption and its uncertainty.
- The original ladder diagnostic trains on early race laps and predicts later
  race laps. It is different from the rolling practice-to-race comparison,
  which also uses qualifying and five opening green race laps per driver.
- The main rolling population retains the historical whole-stint pace filter.
  Its eligibility and anchors are retrospectively selected. The separate
  `online_eligibility` sensitivity removes whole-stint pace trimming and uses
  only observable flags. Both still condition on observed tyre strategy and
  race distance, and dry-race selection is retrospective. Neither is a fresh
  prospective forecasting experiment.
- The original 190+ weekend target is unmet. The available lake contains 84
  weekends from 2023 onward, not the unavailable 2018–2022 tyre history.
- Predictive Student-t calibration clears the pooled CRPS/coverage gate under
  both 70 kg and 110 kg 2026 fuel assumptions. This does not validate latent
  tyre/fuel parameter intervals, every future race, or joint race trajectories.
  The small uncertainty-repair CRPS gain also has a weekend interval including
  zero. Actual 2026 starting fuel remains unverified here.
- The HMM selects two/three descriptive pace states using held-out practice
  likelihood. They are not independently observed warm-up or cliff states.
- The published competitor follows the cited logarithmic tyre-curve equation,
  with fitted within-season construction parameters and within-run practice
  updates. It is not a reproduction of the whole TUM race simulator.
- The stop-decision diagnostic compares against a fitted post-race reference
  with fixed compound order/distance and declared fresh sets. It is not an
  observed optimal counterfactual and uses no independently calibrated joint
  predictive race distribution.
- No live game recovery is established. The example capture is synthetic;
  decoder success and a fuel-observed reference comparison do not prove hidden
  wear recovery or true degradation in seconds.

## Audit board

| Audit | Saved verdict | Interpretation |
|---|---|---|
| a3 | PASS | Identification recovery under the synthetic twin |
| a5 | PASS | Synthetic game stream only |
| ladder | PASS | L0/L1 recovery and real-data diagnostic |
| l2 | PASS | Fuel/track scale and within-weekend compound contrasts |
| l3 | SHIP | Original ten-weekend dirty-air diagnostic |
| l4 | SHIP | L3/L4/L5 calibration comparison; inspect individual rung convergence |
| backtest_lake | FAIL | Original rolling comparison; failed scientific premise is preserved |
| l3_lake_new | HOLD | Independent lake diagnostic relative to original ten-weekend development |
| uncertainty_t | SHIP | Retrospective marginal lap-time calibration only |
| compound_map | PASS | Sourced nomination coverage, not per-lap prototype identification |
| rolling_models | PASS | Experiment completed; candidate gates are separate |
| online_eligibility | PASS | Flag-only population sensitivity; retrospective conditional forecast |
| fresh_holdout | PASS | Unseen Austin offline experiment complete; inspect its separate model_gate |
| strategy_reference | PASS | Conditional stop decision versus fitted reference, not true counterfactual |

### Sampler diagnostics

| Audit | Converged | Saved diagnostics |
|---|---|---|
| ladder | not recorded | `{}` |
| l2 | not recorded | `{"divergences": 0, "max_rhat": 1.030354}` |
| l3 | True | `{"L2": {"divergences": 0, "max_rhat": 1.019565, "min_ess_bulk": 145.903857}, "L3": {"divergences": 0, "max_rhat": 1.012742, "min_ess_bulk": 191.607446}}` |
| l4 | False | `{"rhat_l2": 1.019565, "rhat_l3": 1.012742, "rhat_l4": 1.103228, "rhat_l5": 1.08449}` |
| l3_lake_new | True | `{"L2": {"divergences": 0, "max_rhat": 1.011182, "min_ess_bulk": 133.574162}, "L3": {"divergences": 0, "max_rhat": 1.026101, "min_ess_bulk": 67.903405}}` |

Missing diagnostics are not a convergence pass. Consult each rung's saved
artifact and audit log before interpreting its parameter intervals.

A negative scientific verdict can be a successfully completed experiment.
An absent verdict is incomplete. `python -m tyretrace status` reports saved
gates; `python -m tyretrace list` reports freshness. Inspect
`out/resumed-offline-audits-2.log`, `out/final-freshness-check.log` and
`out/browser-check/` for verification evidence.

## Next manual session

Follow [the manual checklist](TOMORROW.md): start the game and pass the actual
UDP self-check, capture a controlled race including a pit stop, inspect the
live report, execute matched planner programmes, review the product and prepared
demo video, and submit. The user performs Git operations. Future-race replication and latent
recovery remain research evidence, not boxes a build command can tick.

Primary-source nominations and dates are in
[`nominations.json`](../tyretrace/data/nominations.json). See also
[predictive uncertainty](UNCERTAINTY.md), [game validation](GAME_RIG.md),
[published baseline](PUBLISHED_BASELINE.md), and
[conditional strategy evaluation](STRATEGY_EVALUATION.md).
