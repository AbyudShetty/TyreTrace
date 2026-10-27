# Manual session

The remaining human operations are the live game experiment, product/video review,
submission and Git. The code, offline source checks, model candidates,
saved-data dashboard and demo materials are prepared independently.

## 1. Live game data

Start an offline F1 23, F1 24 or F1 25 session. Enable UDP on
`127.0.0.1:20777`, at least 20 Hz, with matching 2023/2024/2025 format and
public telemetry where the game permits it. Use AI cars, no flashbacks, and a
race long enough to include a tyre change. The 2026 Season Pack is not accepted
as a 2025 format.

From the repository root, while the session is unpaused:

```powershell
$env:PYTHONIOENCODING='utf-8'
python -m tyretrace.rig.record --self-check
python -m tyretrace.rig.capture --seconds 900 --out out/rig/live_001.udp
python -m tyretrace.rig.capture --replay out/rig/live_001.udp
```

Only start the capture after the real self-check passes. Stop on a layout
failure; fix the parser against the actual game, never loosen the check. Choose
a fresh filename for each capture. A paused game or restricted AI telemetry
can produce HOLD even with correct settings. More than one car's truth must
actually be observed before claiming multi-car evidence.

The assistant can inspect the report and fit the prepared reference comparison
after capture. A passing layout report establishes usable data. It does not
establish hidden wear recovery or true degradation in seconds. Those claims
need a controlled experiment with an explicit estimand and a genuinely held-out
run. Keep initial fuel, tyre set, car, weather, assists, track and pit-stop
conditions recorded; repeat with changed fuel/tyre programmes to test transport.
Execute a planner-recommended programme and a matched conventional programme
before claiming its counterfactual benefit in the game.

## 2. Review and submission

```powershell
cd web
npm run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173`. Review the nine pages and the prepared
`out/demo/TyreTrace-demo.mp4`. It uses a local synthetic Microsoft David voice;
review the narration before submitting. Supply competition branding and final
submission details where required. If your own voice is required, use
`http://127.0.0.1:5173/demo.html` and `docs/DEMO.md` to replace the narration. The
assistant can incorporate feedback; no UI redesign is assigned back to you.

## 3. Preserve and submit

The user owns Git operations. Suggested commands, after reviewing the files:

```powershell
git add AGENTS.md README.md requirements.txt docs eval tyretrace audit tests web
git commit -m "Complete research candidates, sourced compounds and dashboard"
git push
```

`out/` and `cache/` are ignored. Back them up separately before moving machines:
they contain the raw data, captures, audit results, reports and screenshots.
Commit the exported dashboard JSON with the code so the UI opens without fits.

## Research evidence requiring new observations

Prospective calibration needs genuinely new races after the method is frozen.
Latent tyre/fuel intervals need a separate validation; the predictive wrapper
does not validate them. A sourced actual 2026 starting fuel load would reduce
the current assumption sensitivity. The original 190+ weekend target is unmet
because the current source lacks 2018–2022 tyre data. None of these facts can be
resolved by marking a software task complete or manufacturing another fit.

The broader flag-only calibration test failed (identified 90% coverage 81.4%).
The Austin ML criterion also failed. Further model research requires a new
selection/validation block; these results must not be relabelled as deployment
proof or silently repaired on the same holdout. This is a scientific limitation,
not a request for you to implement the remaining software.
