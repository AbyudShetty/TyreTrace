# Tyre Trace working instructions

Recovered from the user's Claude Code conversations and project memory on
2026-10-26. These are project instructions, not a claim that earlier results
have passed every validation. Read `docs/HANDOFF.md` and `docs/PLAN.md` before
continuing work.

## User preferences

- The user performs Git operations. Do not run Git commands unless explicitly
  asked; supply the exact commands when a completed change needs committing.
- Commit messages must be one short line covering the change. No multi-paragraph
  bodies or long bullet lists.
- When asked to build, complete a substantial, coherent chunk and verify it.
  Explain what is complete, what remains, and how the user can inspect the result.
- The user rejected the Streamlit UI and considers the first React UI too bland.
  React is the product UI; redesign and finish it after the core work.
- Do not treat an old assistant's completion claim as evidence. Check code,
  saved metrics, logs and tests, and report limitations plainly.

## Scientific and implementation contracts

- `eval/protocol.md` is frozen. Append a dated, reasoned amendment when the
  protocol changes; never silently rewrite its original requirements.
- Primary metric: CRPS. A model rung ships only if held-out CRPS improves over
  the preceding rung and 90% interval coverage is within [0.85, 0.95]. Check and
  report sampler convergence as well. Preserve negative results.
- E1 (within-weekend compound contrasts) and E2 (absolute degradation) are
  different estimands. Report them separately in s/lap. Derived s/kg values need
  the fuel-burn assumption and its uncertainty.
- Real forecasting uses rolling history: earlier weekends supply priors. A
  method selected after inspecting data requires a fresh evaluation block.
- Distinguish ladder diagnostics (early race laps predict later race laps) from
  Tier 3 practice-to-race forecasting. Current Tier 3 also consumes qualifying
  and five initial green race laps per driver; describe those inputs accurately.
- `tyretrace/schema.py` defines the shared lap-table contract for real data,
  simulation and the UDP rig. Preserve that contract.
- Audits fit models and write `out/results.json` and `out/artifacts/`. Dashboards
  consume saved outputs; do not launch expensive model fits when a page opens.
- `out/` and `cache/` are local, ignored research assets. Preserve them. Network
  pulls are explicit tasks, not prerequisites to opening a cached dashboard.
- Predictive uncertainty uses `tyretrace/infer/uncertainty.py`. Fit on original
  forecasts from strictly earlier weekends, separately per method, with at least
  eight independent weekend forecasts. `uncertainty_t` audits both fuel loads.
  Its retrospective SHIP concerns marginal lap-time intervals; do not describe
  latent-parameter intervals or joint race paths as validated by that result.
- The live rig must pass `python -m tyretrace.rig.record --self-check` against the
  actual game before treating its output as ground truth. Fix packet offsets if
  validation fails; never weaken the validation to get a pass.

## Charts and verification

- Keep entity colors stable across filters. Use one y-axis, visible labels or
  legends, hover details, and accessible table alternatives. Validate categorical
  colors for light/dark contrast and color-vision deficiencies; visually inspect
  charts before calling UI work finished.
- Use `python -m tyretrace status` and `python -m tyretrace list` to inspect saved
  gates and freshness. A missing verdict is incomplete, not a successful gate.
- Fast Python checks: `python -m pytest tests/ -q -m "not slow"`. Statistical model
  changes may also require the relevant slow recovery tests and audits.
- Web checks, from `web/`: `npm test` and `npm run build`.
- In PowerShell use `$env:PYTHONIOENCODING='utf-8'` for audit console output.
- Avoid concurrent heavy sampler fits on this laptop. Native Windows currently
  uses CPU JAX; WSL/GPU setup is optional, not required for existing tasks.
