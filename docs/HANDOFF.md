# Claude Code to Codex handoff — 2026-10-26

**Subsequent implementation:** the predictive uncertainty layer is now built,
tested and integrated; see `docs/UNCERTAINTY.md` for current results and limits.
Game capture/replay, independent packet-layout fixtures and a conditional
one-stop decision diagnostic were also added; see `docs/GAME_RIG.md` and
`docs/STRATEGY_EVALUATION.md`. No live game proof is claimed.
The historical assessment below predates these changes. Original L3 findings and
the need for future-race / latent-parameter validation still stand.

## Evidence reviewed

- Claude's main project session `8d5ea450-33a3-4b35-9766-aeff960552af`,
  2026-10-14 through 2026-10-21, and the two smaller adjacent project sessions.
- Project-specific Claude memory: user handles Git operations; commit messages
  stay to one short line. No existing repository AGENTS.md or CLAUDE.md was found.
- Original two-page idea-submission PDF, README, `docs/PLAN.md`, frozen evaluation
  protocol and amendments, implementation, tests, local lake and saved results.
- Current `python -m tyretrace status` and `python -m tyretrace list`.

The last substantive Claude instruction was to finish the core before redesigning
the UI. The last user question asked how long `run all` takes while at L4; the
assistant hit its usage limit without answering. The saved L4 task took 867 seconds
(about 14.5 minutes). L3's lake refit took about 14.5 minutes in its standalone
log. A fresh full run takes tens of minutes; cached tasks take seconds, but
currently two task entries are stale or lack completion metadata.

## What exists

| Area | Implemented state |
|---|---|
| Foundations | Canonical lap schema, frozen protocol, dependency/staleness runner, results/artifact layers |
| Data | Original 10-weekend FastF1 dataset; resumable OpenF1 lake; parity audit and adapter regression checks |
| Identification | Synthetic recovery, qualifying track anchor, pit-stop identification, compound contrasts, hierarchical model ladder |
| Model diagnostics | L0–L6 variants; dirty-air feature; weekend slope scatter; rejected variance/mixture models preserved |
| Evaluation | Four baseline implementations on the ladder split; rolling practice-to-race backtest; ablations; 2026 sensitivity audit |
| Explanation | Backend post-race error decomposition into slope, tyres, traffic and noise |
| Planning | Gaussian expected-information-gain run planner, Python implementation and tested TypeScript port |
| Legacy UI | Ten Streamlit pages |
| Product UI | React overview, problem explainer, interactive planner; six navigation pages are placeholders |

Data counts need an explicit denominator. Current raw lake files contain 84 race
sessions / 93,861 race laps, 86 qualifying sessions / 25,025 laps, and 213 practice
sessions / 91,776 laps. The raw dry-race filter leaves 74 weekends; the saved
fuel-gap analysis uses 71 eligible races, and the rolling backtest evaluates 68
weekends after history/availability requirements. Do not call every raw weekend
an evaluated weekend. The previous assistant's final message said 81 raw weekends;
the present files contain 84.

## Established results and limits

- The original practice-only identification argument failed; the qualifying
  anchor replaces it. Compound contrasts do not identify absolute levels alone.
- Fuel-only analysis over 71 eligible dry races estimates 0.03244 +/- 0.00109
  s/kg under its fuel-load model. Race track evolution is much weaker than the
  earlier ten-weekend model implied.
- Saved post-stop CRPS over 68 backtest weekends: conventional 0.65659,
  identified 0.66845, fuel-only 0.67440, no-track 0.65730, naive 1.40426.
  Earlier analysis reports no decisive difference between fuel-corrected methods;
  there is no established superiority over conventional correction.
- L3 clears the gate on the original ten weekends, coverage 0.85768. On 61 new
  eligible weekends it fails: coverage 0.79745, CRPS 0.62647 versus L2's 0.62353.
  This ladder check predicts later race laps from earlier race laps, not the
  separate rolling Tier 3 forecasting task.
- Rolling forecasts also under-cover: identified 90% intervals contain 76.6% of
  all scored laps and 74.5% after a stop. Calibration is a central research gap.
- The largest identified post-race error component is practice-derived tyre
  degradation (RMS about 0.915 s), ahead of slope (0.497 s), traffic (0.339 s)
  and residual noise (0.689 s).
- The saved 2026 audit evaluates 14 weekends, despite README text saying fifteen.
  Identified wins under the assumed 70 kg load; conventional narrowly wins under
  110 kg. The headline gate fails. Neither this audit nor this handoff verifies
  the actual fuel loads or the original PDF's 2026 regulation/compound claims.
- Tier 2 is validated on fake UDP packets only. There is no live game proof yet.
- Planner v1's roughly 21% information gain requires a mid-run tyre change.
  Without that option it reproduces the standard programme. It is a Gaussian
  design calculation, not the pitch's nonlinear nested-Monte-Carlo planner or a
  demonstrated real-world pace improvement.

## Handoff defects and remaining protocol work

1. `l3_lake_new` has metrics and a completed log ending HOLD, but no `gate` or
   `finished_at_epoch` in `out/results.json`. Status displays it as not run while
   still concluding "all gates as expected". Investigate/re-run through the
   runner; do not invent completion metadata or label this PASS.
2. `headline_2026` has metrics and FAIL but lacks runner completion metadata.
   It is marked stale. Reconcile its 14 evaluated weekends with documentation.
3. `web/public/data/summary.json` was exported before the user's later pipeline
   rerun. React still foregrounds the ten-weekend data and stale fuel-gap prose;
   it does not render the newer lake, post-race, ablation or 2026 sections.
   Re-exporting alone will not fix hard-coded page explanations.
4. README's initial Phase 0/no-model status, expected-L3-HOLD explanation and
   earlier numerical sections disagree with later results. Preserve historical
   findings while clearly marking what is current and what was superseded.
5. Four baselines exist for ladder diagnostics, but the decisive rolling
   backtest currently compares identified/fuel-only/no-track/conventional/naive.
   ML and the published baseline are not included in that comparison.
6. Optimal-stop-lap error is required by the protocol but is not implemented.
   The actual backtest uses qualifying and five early green race laps as anchors,
   unlike the original PDF's FP2-only claim. Document this with a dated protocol
   amendment or implement an additional strict pre-race evaluation.
7. Check the published baseline against its primary paper/code before claiming a
   faithful academic reproduction. This has explicitly remained unverified.
8. The gradient-boosting baseline imports scikit-learn, but `requirements.txt`
   does not declare it. It is installed on this machine; fresh-environment
   reproducibility needs a dependency review.

## Recommended continuation

First reconcile gate recording, stale presentation and evaluation scope so the
status report can be trusted. Then investigate practice-degradation estimation
and weekend-level uncertainty, using fresh temporal evaluation blocks. Build the
physical compound map before interpreting cross-weekend compound pooling as
pooling the same rubber. Time-box the regime/HMM model and require evidence it
earns its complexity. Complete the six React pages and add the new audit views,
then produce the demo, video and final write-up.

The remaining rows in `docs/PLAN.md` add up to 22–39 person-days including the
8–15 day HMM estimate, or 14–24 without it, before newly discovered evaluation
gaps. These are rough research estimates, not promised completion times. The
earlier assistant's "15–20 person-days" does not cover its entire listed scope.
Per-corner telemetry was deliberately cut to future work; it is not implemented.
2018–2022 tyre data is unavailable through the current pipeline, and the original
190+ weekend target has not been met.

## User-dependent work

- Launch a supported F1 game session; enable UDP, port 20777, 20 Hz or higher,
  supported 2023/2024/2025 format, and Send All Cars. Run the self-check before
  recording. Whether the actual game exposes every AI car's truth remains to be
  confirmed live.
- Review the eventual React design and complete the screen recording/submission.
- Run Git add/commit/push personally. Preserve the ignored `out/` and `cache/`
  separately; Git does not back up the research results or raw downloads.
- Provide submission timing and game availability when scheduling the next
  implementation chunk. These do not block the independent code/data work.
- A defensible source for actual 2026 fuel loads would strengthen that headline;
  otherwise retain the sensitivity analysis. Source research and the Pirelli map
  can be done by the coding assistant and are not manual assignments by default.

No dependency reinstall or data download is needed to inspect this machine's
existing outputs. CPU execution works; WSL/GPU setup is optional. A direct disk
query during this review found about 19.9 GiB free on C:; the earlier 96%-full
warning is historical rather than a fresh measurement.

## Commands

From the repository root:

```powershell
$env:PYTHONIOENCODING='utf-8'
python -m tyretrace status
python -m tyretrace list
python -m pytest tests/ -q -m "not slow"
python -m tyretrace app
```

For React, the existing snapshot opens without a refit:

```powershell
cd web
npm run dev
```

Use `python -m tyretrace export` from the repository root when intentionally
refreshing the committed snapshot. `python -m tyretrace run all` is a compute
operation, not a prerequisite for every UI launch. `run lake` explicitly fetches
network data; do not delete the original FastF1 cache to try to reproduce it.

## Verification in this handoff

- React: all 17 Vitest checks pass; TypeScript and production Vite build pass.
- Python: all 118 selected fast checks passed across the initial run and targeted
  rerun. Initially 113 passed and five could not set up because the sandbox
  denied access to pytest's existing temporary directory. The affected artifact
  and pipeline modules then passed all 15 checks with approved access. Four slow
  sampler-recovery tests were excluded. The full research pipeline was not
  re-run for this documentation review.
- No application/model code, evaluation protocol or research results were edited.
