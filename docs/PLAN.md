# Build plan — current status

## Autonomous completion — 2026-10-27

The original implementation backlog has been built. All 26 offline audits are
current; 203 Python tests, 17 web tests and 36 browser views pass. The saved gates in `RESULTS.md` and
`out/results.json` determine scientific outcomes. A rejected candidate remains
a completed experiment, not an adopted improvement.

| Area | Current implementation |
|---|---|
| Predictive uncertainty | Earlier-weekend Student-t calibration; both fuel assumptions; negative Gaussian result preserved |
| Practice degradation | Robust within-run slopes and random-effects pooling, with rolling evaluation |
| Regime model | Two/three-state constrained HMM, held-out practice selection, diagnostics and rolling candidate |
| Compound mapping | Dated primary sources for all 84 lake weekends, null for unresolved data, season-specific pooling keys |
| Baselines | Naive, linear/quadratic conventional, ML and source-form published rolling comparisons |
| Eligibility | Original whole-stint filter retained; separate flag-only sensitivity and explicit conditional-input limits |
| New holdout | Austin isolated from the lake, hashes frozen before timing access, experiment scored; preselected ML criterion HOLD |
| Game validation | Independent packet fixtures, live self-check, immutable raw capture, replay and reference comparison |
| Product | Nine React pages, saved-data exports, filters, CSV/table alternatives, desktop/mobile and light/dark design |
| Delivery | Results write-up, timed tour, verified 3:55 narrated MP4 and manual-session checklist |

### Human operations

1. Launch the game, pass its actual UDP self-check and collect controlled live
   data with a pit stop. Then inspect recovery and execute matched planner runs.
2. Review the product and demonstration, and supply competition-specific
   submission details. Synthetic narration can be replaced with your own voice.
3. Perform Git commit/push and separately back up the ignored research assets.

Follow [TOMORROW.md](TOMORROW.md). The assistant can analyse the captured data and
make any subsequent fixes; software work is not assigned back to the user.

### Evidence requiring additional observations

Prospective replication, latent tyre/fuel interval validation and hidden game
wear recovery remain scientific questions. One unseen offline Austin weekend
does not provide enough independent races to settle them. The current data
source cannot supply the original 2018–2022 tyre-history target. Per-corner
telemetry remains deliberately outside scope. Preserve unsuccessful findings.

The original filtered-population uncertainty SHIP does not transport to the
flag-only 2026 population (81.4% identified coverage). The broader operational
calibration problem remains open; neither the completed build nor a new label
turns that negative result into validated forecasting.

## Historical planning record

The estimates and status below describe the 2026-10-21 plan and are superseded
by the current implementation table above.

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

**2026-10-26 update:** marginal lap-time predictive calibration is implemented
and clears the pooled CRPS/coverage gate in retrospective rolling checks on
2025 and 2026, including both fuel assumptions. It is available in the forecast
API, task runner and React overview. The Gaussian attempt remains rejected.
The small gain is uncertain across independent weekends; prospective
replication needs new races. Raw L3 and latent-parameter calibration are not
declared fixed by a predictive wrapper. Details: `docs/UNCERTAINTY.md`.

| # | Work | Days | Notes |
|---|---|---|---|
| 1 | **Prospective calibration replication / latent uncertainty** | research | Predictive layer implemented: 2026 cov90 0.936 / 0.930 under 70 / 110 kg, with lower CRPS. Original L3 remains HOLD at scale; latent tyre/fuel intervals need separate validation |
| 2 | **Degradation from practice** | 3–5 | the largest forecast error on 68 weekends (rms 0.92 s vs slope 0.50, traffic 0.34) — better practice-slope estimation is where race forecasting improves next |
| 3 | Live rig `--self-check` against the F1 game | 1–3 | needs the game; fix offsets, never loosen the check |
| 4 | Regime / HMM model | 8–15 | warm-up and cliff as states; time-box |
| 5 | Pirelli C1–C5 compound map | 1–2 | unlocks pooling SOFT/MEDIUM/HARD across weekends |
| 6 | Dashboard phase 2 + redesign | 4–6 | six pages; wire in fuel gap, post-race, ablations, lake, 2026 |
| 7 | Demo script, video, write-up | 2–3 | |
| — | Per-corner telemetry features | 8–12 | **cut** — future-work slide |

## Done

**2026-10-26 additions:** raw game capture/replay and reference-validation
workflow, independently verified 2023/24/25 packet layouts, completed-lap and
freshness fixes, and UDP loopback tests (`docs/GAME_RIG.md`). Live validation
still requires a running game. A conditional one-stop decision diagnostic now
reports stop-lap error and regret on 520 eligible driver races across the lake;
its post-race fitted reference is not observed counterfactual truth. Four full
rolling baselines and a regime model remain unfinished.

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
