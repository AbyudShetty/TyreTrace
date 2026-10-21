# Build plan — remainder

Where the project stands against the four-phase plan in the idea submission, and
the order the rest gets built in. Estimates are person-days; double them for
calendar time on a part-time student schedule.

## Status (2026-10-21)

| Phase | State |
|---|---|
| 0 Feasibility audit | done — gates, protocol, schema, runner |
| 1 Data lake | done for 2023→today via OpenF1 (FastF1's source now returns 401); parity-checked against FastF1. 2018–2022 unavailable (no public tyre data). No C1–C5 map; no per-corner telemetry |
| 2 Identification | done — fuel-pace gap explained (L2 overstated race track evolution ~2.5×); fuel 0.0324 ± 0.0011 s/kg on 71 races; L2–L6 refitted with per-weekend slope scatter |
| 3 Proof | Tier 1 done; Tier 2 rig on fake packets only; Tier 3 backtest done on 10 weekends + lake; all 4 baselines; ablations done; 2026 headline built |
| 4 Planner & UI | run planner v1 done; post-race validation done (backend); React dashboard phase 1 done, phase 2 + redesign pending |

## What is left

| # | Work | Days | Notes |
|---|---|---|---|
| 1 | **Calibration at scale — the main open problem** | 3–5 | L3 ships on the 10 weekends but NOT on 61 new ones (coverage 0.797). Rejected so far: compromised-lap mixture (L6), rolling recalibration (CRPS +8.8%). Most error is weekend-level, not lap noise |
| 2 | **Degradation from practice** | 3–5 | the largest forecast error on 68 weekends (rms 0.92 s vs slope 0.50, traffic 0.34) — better practice-slope estimation is where race forecasting improves next |
| 3 | Live rig `--self-check` against the F1 game | 1–3 | needs the game; fix offsets, never loosen the check |
| 4 | Regime / HMM model | 8–15 | warm-up and cliff as states; time-box |
| 5 | Pirelli C1–C5 compound map | 1–2 | unlocks pooling SOFT/MEDIUM/HARD across weekends |
| 6 | Dashboard phase 2 + redesign | 4–6 | six pages; wire in fuel gap, post-race, ablations, lake, 2026 |
| 7 | Demo script, video, write-up | 2–3 | |
| — | Per-corner telemetry features | 8–12 | **cut** — future-work slide |

## Done

| Work | Date | Result |
|---|---|---|
| Runner bug, artifacts layer, Streamlit dashboard | 2026-10-14 | |
| Baselines B1–B3, r_hat reporting | 2026-10-15 | |
| Run planner v1 | 2026-10-15 | +21% information with one mid-run tyre change; +0% without (rediscovers the standard programme) |
| Tier 3 backtest | 2026-10-17 | FAIL on 10 weekends, recorded as a finding |
| React dashboard phase 1 | 2026-10-17 | overview, explainer, live planner |
| Fuel-pace gap | 2026-10-21 | fuel alone 0.0293 ± 0.0024 s/kg = physics; s not identified once scatter is allowed |
| B4 published baseline | 2026-10-21 | CRPS 0.350 vs L4 0.344 — strongest baseline |
| Post-race validation | 2026-10-21 | exact split into slope / tyres / traffic / noise; traffic at the anchor the largest systematic error |
| Traffic-aware anchors | 2026-10-21 | identified CRPS 0.589 → 0.569, bias +0.19 → +0.06 s |
| Ablation table | 2026-10-21 | every built component earns its place |
| Data lake via OpenF1 + parity audit | 2026-10-21 | PASS; used-set tyre age offset documented |
| 2026 headline audit | 2026-10-21 | gate flips with the unverified 2026 fuel load; identified best under 70 kg (−8%); 2026 race slopes contradict 70 kg at unchanged sensitivity |
| Lake backtest (pre-registered) | 2026-10-21 | FAIL: fuel-corrected methods statistically tied on 68 weekends; all beat naive (t −9.6) |
| Out-of-sample L3 check | 2026-10-21 | the 10-weekend ship does not replicate on 61 new weekends |

## Rules that don't change

- `eval/protocol.md` is frozen. Changes go in a dated amendment, never a silent edit.
- A rung ships only on better CRPS **and** coverage in [0.85, 0.95].
- Audits write to `out/`; the dashboard only reads it.
- A method chosen after seeing results is tested on data it has not seen
  (fuel-only and traffic-aware anchors were pre-registered before the lake).
