"""Render the final research report from saved audits; never fit a model here."""
from pathlib import Path
from datetime import datetime,timezone
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tyretrace import results
from tyretrace import artifacts
import json
from collections import Counter


def main():
    tasks=results.load()['tasks'];m=lambda n:tasks.get(n,{}).get('metrics',{})
    rolling=m('rolling_models');rows=rolling.get('summary',[])
    coverage=m('compound_map');unc=m('uncertainty_t');strategy=m('strategy_reference')
    fresh=m('fresh_holdout');fresh_text='The isolated Austin holdout has not yet been scored.'
    if fresh.get('summary') and tasks.get('fresh_holdout',{}).get('gate')=='PASS':
        ft=['| Method | Subset | CRPS (s) | 90% coverage | Scored laps |','|---|---|---:|---:|---:|']
        ft.extend(f"| {r['method']} | {r['subset']} | {r['crps']:.5f} | {r['coverage_90']:.1%} | {r['n']} |" for r in fresh['summary'])
        fresh_text=(f"The 2025-selected ML method's frozen criterion is **{fresh['model_gate']}**. "
            'It must beat every comparator on CRPS with 90% coverage in [0.85,0.95]. '
            'This is one independent, previously unseen offline weekend. The event predates the algorithm freeze; '
            'it is not a forecast issued before the race or a statistically sufficient replication.\n\n'+ '\n'.join(ft))
    table=['| Method | All-lap CRPS (s) | 90% coverage | Post-stop CRPS (s) | Post-stop coverage |',
           '|---|---:|---:|---:|---:|']
    for name in sorted({r['method'] for r in rows}):
        a=next((r for r in rows if r['method']==name and r['year']=='2026' and r['subset']=='all'),None)
        b=next((r for r in rows if r['method']==name and r['year']=='2026' and r['subset']=='after_stop'),None)
        if a and b:table.append(f"| {name} | {a['crps']:.5f} | {a['coverage_90']:.1%} | {b['crps']:.5f} | {b['coverage_90']:.1%} |")
    online=['| Method | Subset | CRPS (s) | 90% coverage | Scored laps |','|---|---|---:|---:|---:|']
    online.extend(f"| {r['method']} | {r['subset']} | {r['crps']:.5f} | {r['coverage_90']:.1%} | {r['n']} |"
        for r in m('online_eligibility').get('summary',[]) if r['year']=='2026')
    flag=next((r for r in m('online_eligibility').get('summary',[])
        if r['method']=='identified' and r['year']=='2026' and r['subset']=='all'),None)
    flag_text=(f"Identified 90% intervals cover {flag['coverage_90']:.1%} of this 2026 population. "
        'Inspect the method rows against the coverage band; completing this audit does not imply its models ship.'
        if flag else 'Flag-only sensitivity has no saved result yet.')
    convergence=['| Audit | Converged | Saved diagnostics |','|---|---|---|']
    for name in ('ladder','l2','l3','l4','l3_lake_new'):
        value=m(name)
        diagnostic=value.get('diagnostics') or {k:v for k,v in value.items() if 'rhat' in k or 'diverg' in k}
        convergence.append(f"| {name} | {value.get('converged','not recorded')} | `{json.dumps(diagnostic,sort_keys=True)}` |")
    hmm='No saved regime diagnostics.'
    diag=artifacts.load('rolling_models','diagnostics')
    if diag is not None:
        values=[json.loads(r['detail']) for r in diag.to_dicts() if r['method']=='regime_practice']
        used=[r for r in values if r.get('regimes_used')]
        states=dict(Counter(r['states'] for r in used))
        hmm=f"HMM was applicable on {len(used)} of {len(values)} forecast weekends; {sum(r.get('converged',False) for r in used)} fits converged. Selected state counts: {states}. Nonconverged fits remain flagged in the diagnostics."
    gates=['| Audit | Saved verdict | Interpretation |','|---|---|---|']
    meanings={
        'a3':'Identification recovery under the synthetic twin',
        'a5':'Synthetic game stream only',
        'ladder':'L0/L1 recovery and real-data diagnostic',
        'l2':'Fuel/track scale and within-weekend compound contrasts',
        'l3':'Original ten-weekend dirty-air diagnostic',
        'l4':'L3/L4/L5 calibration comparison; inspect individual rung convergence',
        'backtest_lake':'Original rolling comparison; failed scientific premise is preserved',
        'l3_lake_new':'Independent lake diagnostic relative to original ten-weekend development',
        'uncertainty_t':'Retrospective marginal lap-time calibration only',
        'compound_map':'Sourced nomination coverage, not per-lap prototype identification',
        'rolling_models':'Experiment completed; candidate gates are separate',
        'online_eligibility':'Flag-only population sensitivity; retrospective conditional forecast',
        'fresh_holdout':'Unseen Austin offline experiment complete; inspect its separate model_gate',
        'strategy_reference':'Conditional stop decision versus fitted reference, not true counterfactual',
    }
    for name,meaning in meanings.items():gates.append(f"| {name} | {tasks.get(name,{}).get('gate','INCOMPLETE')} | {meaning} |")
    text=f'''# Tyre Trace: saved research results

Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} from
`out/results.json`. Original failed experiments remain in the artifact folders.
This report describes the current evidence, not the historical completion claims.

## Delivered software

The uncertainty layer, immutable UDP capture/replay and layout checks,
conditional strategy diagnostic, source-form published baseline, robust practice
candidate, two/three-state HMM candidate and rolling ML comparison are built.
The conventional quadratic baseline is evaluated separately from its historical
linear approximation. The nomination registry covers **{coverage.get('mapped','?')}
of {coverage.get('weekends','?')} cached race weekends** using dated primary sources.
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

{chr(10).join(table)}

The method selected using the 2025 development block is
**{rolling.get('selected_on_2025','INCOMPLETE')}**. The already inspected 2026
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

{hmm}

## Flag-only eligibility sensitivity

This separate 2026 population removes whole-stint pace and run-length trimming
from race eligibility and opening-anchor selection. It retains the declared
observed strategy/distance and retrospective dry-race restriction. Calibrators
are fitted separately for this population using strictly earlier weekends.
The original and expanded populations have different denominators; their CRPS
values are not a paired comparison on identical laps.

{flag_text} The original filtered-population SHIP cannot establish
reliable operational forecasting without retrospective pace selection.

{chr(10).join(online)}

## Isolated Austin 2026 holdout

The original 84-weekend lake remains unchanged. Predictor, protocol and input
hashes were recorded in `out/fresh_holdout/freeze.json` before downloading the
new weekend's lap timing. The isolated practice, qualifying and race files have
their own checksums. Sources are verified again before scoring. The primary
fuel assumption is 70 kg; observed tyre strategy/distance and five green race
anchors remain declared inputs. No method was retuned on the outcome.

{fresh_text}

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

{chr(10).join(gates)}

### Sampler diagnostics

{chr(10).join(convergence)}

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
'''
    (ROOT/'docs/RESULTS.md').write_text(text,encoding='utf-8')
    print('Wrote docs/RESULTS.md')
    return 0

if __name__=='__main__':raise SystemExit(main())
