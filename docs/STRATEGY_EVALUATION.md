# Conditional one-stop decision evaluation

`python -m tyretrace run strategy_reference` reports stop-lap error and strategy
time regret for the original rolling forecasts. The implementation lives in
`tyretrace/eval/strategy.py`; explicit lap-by-lap simulation tests verify the
cost calculation, pit-lap convention, constraints, ties and parameter draws.

For a race of N laps and a stop after lap p, compare the supplied first/second
compound degradation curves over p and N-p laps. Fuel and track terms common
to every candidate cancel. Fixed pit loss is included but cannot affect which
lap wins when exactly one stop is required. Prior scrub ages can be supplied.
`stop_distribution` accepts JOINT paired compound-parameter draws; marginal
calibrated lap-time intervals are not independent simulated race paths.

The lake audit uses the original saved forecast parameters and a separately
fitted post-race reference with traffic. It evaluates 520 driver races with
exactly two observed stints and different dry compounds, conditioned on that
compound order and completed race distance, assuming fresh sets. Retirements
before 90% of the race distance and more complex strategies are excluded.

| Method | Stop-lap MAE | Mean reference regret (s) | Median regret (s) |
|---|---:|---:|---:|
| Identified | 9.78 | 6.69 | 2.17 |
| Fuel-only | 9.78 | 6.65 | 2.25 |
| Conventional | 10.46 | 7.32 | 2.30 |
| No track term | 10.38 | 7.26 | 2.30 |
| Naive | 13.38 | 11.06 | 6.22 |

This fills in a decision diagnostic, not an observed counterfactual optimum.
It does not establish that executing the proposed stop would have produced
the calculated race time: traffic, nonlinear wear, pit windows, safety cars,
opponent decisions and track position are outside this simple comparison.
No superiority ship gate is attached to these descriptive results. Large
stop-lap errors are retained rather than obscured by better lap-time CRPS.

Results are in `out/artifacts/strategy_reference/decisions.parquet` and
`summary.parquet`; headline summaries are exported with dashboard data.
