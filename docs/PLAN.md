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
| 4 Planner & UI | dashboard done (2026-10-14); run planner v1 done (2026-10-15); post-race tool not started |
| Baselines | 3 of 4 done (2026-10-15): naive, conventional, gradient boosting. Published model not built |

## Critical path

**Calibration.** No rung ships, because every rung's 90% intervals cover too few
laps. The backtest, the ablations and the planner all need a calibrated model
underneath, so this goes first.

## Work list, in order

| # | Work | Days | Notes |
|---|---|---|---|
| ✅ | Runner bug (`calib` vs `l4`), artifacts layer, dashboard | — | done 2026-10-14 |
| ✅ | Baselines B1–B3 + audit + dashboard; L4 reports r_hat | — | done 2026-10-15 |
| 1 | Live rig `--self-check` against the F1 game | 1–3 | needs the game; fix offsets, never loosen the check |
| 2 | Calibration: model the MEAN on old tyres | 2–4 | L4's misses are one-sided (12% above q95, 7% below q05; 16% above at age 26–30). **Tried and rejected 2026-10-15:** a clean/compromised-lap mixture (L6, three variants — all cover worse, 0.76 vs 0.81; age effect came out negative). So it is not lap-level noise. Next: degradation curvature / per-stint cliff in the mean |
| 2b | L3–L5 convergence | 1–2 | r_hat 1.05–1.10, ESS ~25 on beta_f / stint offsets / deg. **Not** circuit centring (tried 2026-10-15, no help). It is the stint-offset vs fuel trade-off the README warns about. Forecasts are stable across runs (cov 0.812/0.813/0.813); fuel estimates from L3+ are not — quote L2's |
| 3 | Regime / HMM model | 8–15 | now the lead candidate for #2 — warm-up and cliff as states with their own mean, not just their own noise |
| 4 | Full 2018–2026 pull + Pirelli C1–C5 map | 3–6 | unlocks cross-race compound pooling |
| 5 | Fuel-pace gap investigation | 2–3 | time-boxed; identified 0.030 vs physics 0.055–0.060 s/lap |
| 6 | Published-model baseline (B4) | 2–4 | B1–B3 done. Result so far: L4 beats all three on CRPS, but conventional (0.371) beats L2 (0.381) — same-race forecasting can't test the fuel split; #7 can |
| 7 | Tier 3 rolling-origin backtest, FP → race | 4–7 | ~190 NUTS fits ≈ a day of CPU; run overnight |
| 8 | Ablation table (protocol §6) | 2–3 | mostly compute |
| 9 | 2026 headline evaluation | 1–2 | the sample-efficiency claim |
| ✅ 10 | Run planner v1 (expected information gain) | — | done 2026-10-15: exact linear-Gaussian EIG, beam + local search, dashboard page. Fair result: +0% without tyre changes (rediscovers the standard programme), **+21% with one mid-run tyre change**. v2: cliff/regimes, tyre allocation, two cars |
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
