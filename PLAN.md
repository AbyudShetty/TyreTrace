# Build plan — remainder

Where the project stands against the four-phase plan in the idea submission, and
the order the rest gets built in. Estimates are person-days; double them for
calendar time on a part-time student schedule.

## Status

| Phase | State |
|---|---|
| 0 Feasibility audit | done — gates, protocol, schema, runner |
| 1 Data lake | partial — 10 weekends (2023–25); no C1–C5 map; no per-corner telemetry |
| 2 Identification | mostly done — L0–L3 built; L4 best CRPS but under-covers (0.813) |
| 3 Regimes & proof | Tier 1 done; Tier 2 rig on fake packets only; no backtest, baselines, ablations |
| 4 Planner & UI | dashboard done (2026-10-14); planner, post-race tool not started |

## Critical path

**Calibration.** No rung ships, because every rung's 90% intervals cover too few
laps. The backtest, the ablations and the planner all need a calibrated model
underneath, so this goes first.

## Work list, in order

| # | Work | Days | Notes |
|---|---|---|---|
| ✅ | Runner bug (`calib` vs `l4`), artifacts layer, dashboard | — | done 2026-10-14 |
| 1 | Live rig `--self-check` against the F1 game | 1–3 | needs the game; fix offsets, never loosen the check |
| 2 | Calibration quick win: right-skewed likelihood, then non-monotone σ(age) | 1–3 | L4's misses are one-sided — 10–16% of held-out laps land above q95 vs ~6% below q05, worst at age 26–30 (dashboard → Held-out predictions). Student-t is symmetric; try skew-t / exponentially-modified Gaussian first |
| 3 | Regime / HMM model | 8–15 | only if #2 doesn't reach coverage ≥ 0.85 |
| 4 | Full 2018–2026 pull + Pirelli C1–C5 map | 3–6 | unlocks cross-race compound pooling |
| 5 | Fuel-pace gap investigation | 2–3 | time-boxed; identified 0.030 vs physics 0.055–0.060 s/lap |
| 6 | Four baselines (protocol §4) | 4–7 | naive, conventional, gradient boosting, one published model |
| 7 | Tier 3 rolling-origin backtest, FP → race | 4–7 | ~190 NUTS fits ≈ a day of CPU; run overnight |
| 8 | Ablation table (protocol §6) | 2–3 | mostly compute |
| 9 | 2026 headline evaluation | 1–2 | the sample-efficiency claim |
| 10 | Run planner v1 (expected information gain) | 5 | small candidate grid, Gaussian approximation |
| 11 | Post-race validation view | 3–5 | residual split: fuel / traffic / track / model miss |
| 12 | Demo script, video, write-up | 2–3 | |
| — | Per-corner telemetry features | 8–12 | **cut** — future-work slide |

Lean total ≈ 30 person-days ≈ 3–4 weeks for three people.

## Parallel tracks

```
           Week 1               Week 2                 Week 3                Week 4
Track A    #2 calibration  ──►  #3 HMM if needed  ──►  #10 planner v1        polish
Track B    #4 data, #6 base ──► #7 backtest       ──►  #8 ablations, #9      polish
Track C    #1 rig live      ──► #11 post-race view ──► wire into dashboard   #12 demo
```

## Rules that don't change

- `eval/protocol.md` is frozen. Changes go in a dated amendment, never a silent edit.
- A rung ships only on better CRPS **and** coverage in [0.85, 0.95].
- Audits write to `out/`; the dashboard only reads it.
