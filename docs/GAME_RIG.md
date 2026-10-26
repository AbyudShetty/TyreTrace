# Live game validation workflow

## What is ready

Raw UDP capture, checksum verification, deterministic replay, version-specific
decoding, canonical lap export, multi-car truth availability diagnostics and a
truth-blind estimator versus a fuel-observed reference. Independent packed
fixtures check field order without reusing decoder offsets. Loopback tests also
exercise real UDP sockets. Synthetic success is labelled synthetic forever.

No F1 game process was running during implementation. Live layout checks and
hidden-state recovery on an actual game have not been established.

## Corrected defects

- F1 24/25 position, lap number, pit status and validity offsets were one byte
  late. F1 23 needs a different layout because two minute fields were added in
  the middle for F1 24. Packet sizes are 1131 bytes for 2023 and 1285 for 2024/25.
- LastLapTime belongs to the completed lap, `current_lap_num - 1`. The old
  recorder labelled it with the current lap. The fake stream now models that
  boundary and establishes an initial lap before reporting a completed one.
- Incomplete per-car blocks are rejected rather than silently inferred by
  integer division. Invalid session clocks and packet versions are rejected.
- A missing truth sample, unknown compound, status/wear older than two seconds,
  or a gap spanning several unobserved laps cannot produce a fabricated row.
- Completed-lap validity uses the previous lap's observed invalidity, not the
  new lap's flag. Both actual construction and visual compound are retained.
- F1 25 C6 is recognised. Restricted telemetry is reported rather than assuming
  every AI or multiplayer car exposes fuel and wear.

## User steps for the live run

1. Start an offline race in a supported F1 game, preferably with AI cars and
   enough laps for a tyre change. Disable flashbacks during the experiment.
2. Enable UDP telemetry to `127.0.0.1:20777` at 20 Hz or higher. Choose packet
   format 2023, 2024 or 2025 matching the game. Set Your Telemetry to Public
   when permitted. More than one car's fuel/wear must be observed before
   claiming multi-car ground truth; a setting name alone does not establish it.
3. Use the existing layout preflight with the game unpaused:

   ```powershell
   python -m tyretrace.rig.record --self-check
   ```

4. Capture fifteen minutes including a pit stop, with a new output name:

   ```powershell
   python -m tyretrace.rig.capture --seconds 900 --out out/rig/live_001.udp
   ```

   Ctrl+C ends the capture and finalises its evidence file. Socket silence
   produces HOLD, not PASS. Existing captures are never overwritten.

5. Re-run analysis without playing again:

   ```powershell
   python -m tyretrace.rig.capture --replay out/rig/live_001.udp
   ```

The `.udp` file is immutable source evidence; `.udp.json` records provenance
and checksum; `.report.json` records diagnostics. A passing data-readiness check
also emits canonical `.parquet` laps. Mixed sessions, corrupt or incomplete
archives and implausible readings fail. Too little data holds.

The published 2026 Season Pack uses a different protocol/car count. It is not
silently accepted as 2025; support requires independently verified layouts.

## What the reference comparison proves

The blind estimator is explicitly denied every `truth_*` column. It fits lap
time from lap number and tyre age, per car; pit-stop resets are required for
identification. A separate evaluator fits lap time against measured fuel and
tyre age and compares the slopes. This tests consistency with a fuel-observed
reference. It is not observed true degradation in seconds, and linear fake
burn makes that reference comparison algebraically circular.

Wear percent and pace degradation in seconds are different estimands. A decoder
that reads wear percent does not prove the estimator inferred it. The report
therefore retains `tier2_established: false`; a controlled live experiment must
establish the desired hidden-state recovery separately. No criterion is weakened
to manufacture a live PASS.

## Official layout sources

- [EA/Codemasters F1 23 specification and attachment](https://forums.ea.com/discussions/f1-23-en/f1-23-udp-specification/8390745)
- [EA/Codemasters F1 24 specification and attachment](https://forums.ea.com/discussions/f1-24-general-discussion-en/f1-24-udp-specification/8369125)
- [EA F1 25 specification, version 3](https://forums.ea.com/t5/s/tghpe58374/attachments/tghpe58374/f1-games-game-info-hub-en/61/4/Data%20Output%20from%20F1%2025%20v3.pdf)

The offset checks were derived from these primary sources rather than another
parser's constants. Physical validation remains mandatory on the actual build.
