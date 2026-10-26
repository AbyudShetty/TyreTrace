# Published baseline verification — 2026-10-26

The primary implementation is TUMFTM's
[tyre-degradation function](https://github.com/TUMFTM/race-simulation/blob/master/helper_funcs/src/calc_tire_degradation.py).
Its [project README](https://github.com/TUMFTM/race-simulation) cites Heilmeier,
Graf and Lienkamp's 2018 ITSC race-simulation paper (DOI
10.1109/ITSC.2018.8570012) and subsequent work.

The logarithmic tyre contribution includes a compound-specific offset, a
logarithmic amplitude and a curvature parameter. The age passed to that
function is age at the start of the lap. Our historical `published()`
adaptation omitted the offset and passed inclusive tyre age. Its results are
preserved, but it must not be described as a faithful full-simulator port.

`published_verified()` implements the source curve with per-weekend/compound
offsets, start-of-lap age, the existing fixed fuel coefficient and the existing
curvature grid. Additive driver/compound intercepts use min-norm least squares;
their individual physical values are not identified, but fitted predictions
are. A synthetic regression checks recovery when compound offsets and the
start-age convention both matter.

The separate `published_verified` audit uses exactly the ladder's held-out laps.
The old adaptation scores CRPS 0.34979; the source-verified curve scores
0.64971, MAE 0.76538 and 90% coverage 0.81126. Adding the source's terms made
this fitted baseline generalise worse on this split. The result is retained;
it is not a reason to silently substitute the easier adaptation and call it
the same model. Neither result is an evaluation of the complete TUM simulator,
and the main rolling practice-to-race comparison still needs its ML and
published baseline extensions.

```powershell
python -m tyretrace run published_verified
```

Outputs are separate under `out/artifacts/published_verified/`. The React
overview distinguishes the legacy adaptation and source-verified curve.
