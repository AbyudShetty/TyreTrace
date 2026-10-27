# Four-minute demonstration

The recorded video is `out/demo/TyreTrace-demo.mp4`: 3 minutes 55 seconds,
1440 × 1000, with local synthetic Microsoft David narration. Review the voice
and competition requirements before submitting. The accompanying
`video-report.json`, `media-check.json` and capture manifest document the
encoding, source frames and checks. This video shows the actual saved dashboard,
including the failed flag-only sensitivity and frozen Austin ML criterion.

Start the saved React dashboard; no model fit or network pull is needed:

```powershell
cd C:\ABYUD\HACKATHONS\TRACKSHIFT\web
npm run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173/demo.html` for the timed tour, or the root URL
for manual navigation. Space pauses the tour and the arrow keys move scenes.
Press F11 for a clean recording view. The tour lasts about four minutes.

| Time | View | Narration / action |
|---|---|---|
| 0:00–0:22 | Overview | We separate what tyre data identifies from what a fuel assumption supplies. The workbench makes the evidence, uncertainty and next practice run inspectable. |
| 0:22–0:45 | The problem | Fuel and wear share the same clock within a stint. Qualifying supplies track evolution; race pit stops reset tyre age while fuel continues to burn. E1 is a compound contrast, E2 is absolute degradation. |
| 0:45–1:05 | Race explorer | Choose a weekend and driver. Inspect actual tyre stints, measured pace and timing-derived traffic. There are 84 cached weekends, with sourced compound nominations. |
| 1:05–1:25 | Track & compounds | Every nomination has a source and date. Some selections skip a compound. Qualifying intervals are parameter estimates, separate from race forecasts. |
| 1:25–1:55 | Models & evidence | Complexity must improve CRPS while preserving coverage. Robust practice and HMM candidates are built and evaluated; unsuccessful additions remain visible. ML is a serious comparison, including when it beats the identified model. |
| 1:55–2:22 | Held-out forecasts | Pick a driver and predictive method. The shaded interval is marginal lap-time uncertainty. Priors use earlier weekends; current inputs also include practice, qualifying and five race anchors. Strategy and distance are conditioned on observed choices. |
| 2:22–2:50 | Practice → race | Show the rolling baseline table, calibration, ablations and conditional stop-decision diagnostic. State explicitly that this is retrospective, not a fresh prospective confirmation. |
| 2:50–3:15 | Run planner | Change the budget or tyre availability. The programme balances information and operating constraints. Counterfactual improvement still needs execution in the game. |
| 3:15–3:40 | Game validation | Raw UDP recording, integrity checks and replay are implemented. The example is synthetic until a real capture passes. Wear percent and degradation in seconds are distinct quantities. |
| 3:40–3:55 | Overview | Close with the evidence: a working research tool, calibrated marginal forecasts, and preserved negative findings. Live game proof and new-race replication remain experimental steps. |

Before recording, read `docs/RESULTS.md` and use the current saved numbers.
Do not narrate a prospective validation, hidden wear recovery, a universal
strategy optimum, or a superior causal model unless the corresponding evidence
has actually been obtained. The unedited original result claims elsewhere in
the repository are historical.

## Reproduce the local recording

The recording scripts capture the actual dashboard through a hidden browser.
They do not generate mock interface frames. Local Microsoft David Desktop
narration is synthetic; review it before submission or replace it with your voice.
The encoder needs the optional `audit/requirements-demo.txt` packages.

First refresh exports and build, then keep that build unchanged during recording:

```powershell
python -m tyretrace export
cd web
npm run build
cd ..
python -m http.server 5179 --bind 127.0.0.1 --directory web/dist *> out/dashboard-preview.log
```

Leave the server running. In another PowerShell at the repository root:

```powershell
Start-Process -FilePath 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe' -WindowStyle Hidden -ArgumentList '--headless','--disable-gpu','--no-first-run','--remote-debugging-port=9223','--user-data-dir=C:/ABYUD/HACKATHONS/TRACKSHIFT/out/browser-qa','about:blank'
cd web
npm run test:browser
npm run demo:record
cd ..
& .\audit\demo_voice.ps1
python audit/render_demo.py
```

Output: `out/demo/TyreTrace-demo.mp4`, with a machine-readable
`video-report.json`, source JPEG frames and narration WAVs beside it.
All are ignored local artifacts; back them up separately. The live game
experiment and competition-specific submission review still require you.
