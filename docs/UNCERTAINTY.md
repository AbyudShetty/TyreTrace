# Predictive uncertainty repair

## Frozen algorithm — 2026-10-26, before the new 2026 confirmation

The baseline is the identified method with traffic-aware anchors. Preserve its
point prediction. Fit a Gaussian marginal predictive variance:

    variance = a + b * raw_variance + c * (race_lap/50)^2 + d * (tyre_age/20)^2

Raw variance is inferred from the original central 90% interval width. Fit
nonnegative a/b/c/d by Gaussian predictive likelihood on ORIGINAL forecast
errors from strictly earlier weekends. Each weekend contributes equal total
weight; at least eight independent earlier forecasts are required. There is no
intercept or slope correction to the median. The scales 50 and 20 are fixed
conditioning constants, not tuned cutoffs. All earlier weekends are used, with
no recency window, per-circuit tuning, or target-outcome access.

The missing horizon and tyre terms model predictive discrepancies, not precise
physical estimates of fuel or tyre parameters. In particular, the original
fuel/track fit includes between-weekend slope scatter while its prediction
returns only population-parameter uncertainty. Practice slopes also describe
practice means, not their race transport. Learning predictive dispersion from
earlier forecasting errors accounts for errors a conditional parameter posterior
cannot describe. It does not repair identification or prove latent-parameter
credible-interval coverage.

## Development evidence

Only 2023–2025 residual rows were loaded in development. A fit on 2023–2024,
applied to 2025, improved CRPS from 0.60991 to 0.60158, with coverage 0.93299
(0.93731 after stops). Refit causally before each 2025 weekend: CRPS 0.60135,
coverage 0.92850 (0.93389 after stops).

A bias-adjusting variant was examined on the same development/validation years.
It improved CRPS slightly further, but is not selected: this work preserves the
existing point forecasts and addresses uncertainty with four variance terms.
The original rolling empirical-quantile recalibration remains rejected.

## Confirmation and ship rule

The selected settings above are frozen before new uncertainty scores on 2026.
Use strictly earlier original forecast errors before every target weekend,
including earlier 2026 races once they are over. Compare repaired and original
forecasts on exactly the same laps. Require better CRPS and 90% coverage inside
[0.85, 0.95], both on all scored laps and on post-stop laps. Report 50/80/90/95%
coverage, widths, per-weekend results and a paired weekend bootstrap interval.
Repeat the sensitivity to the original 70/110 kg assumptions without choosing
the favourable one. Previous aggregate 2026 results were seen by the project;
this is retrospective temporal confirmation, not prospective replication.

Full results and reproducible commands will be appended after the frozen run.

## Gaussian confirmation — rejected

The frozen Gaussian model improved 2026 coverage from 0.82240 to 0.93400 but
CRPS worsened from 0.51940 to 0.52106. Post-stop CRPS also worsened slightly.
It therefore remains HOLD. Its results are preserved under
`out/artifacts/uncertainty/` at this stage and will be retained separately when
the next model is evaluated. No claim of an untouched 2026 set is made below.

## Revision 2 — heavy-tailed predictive dispersion

The Gaussian fit's central 50% intervals covered about 60%: fitting squared
errors made the centre too wide to account for rare large errors. Revision 2
uses a Student-t predictive distribution with four fixed degrees of freedom,
the same four nonnegative scale components, unchanged median, equal weekend
weights, and strictly earlier history. It uses Student-t predictive likelihood
instead of Gaussian likelihood. No degrees-of-freedom sweep is planned.
This is further development and retrospective rolling validation; the already
inspected 2026 block is not a new untouched confirmation.

Exact Student-t CRPS follows the scoringRules reference implementation:
[official formula and reference](https://scoringrules.readthedocs.io/en/latest/generated/scoringrules.crps_t.html),
Jordan, Krüger and Lerch (2019), DOI 10.18637/jss.v090.i12. Numerical integration
and simulation checks verify the local formula independently.

## Implemented results

The Student-t layer clears the numerical predictive gate in retrospective
rolling validation, with each fit restricted to earlier forecast weekends.
The 2026 set contains 14 eligible weekends, 11,560 scored laps and 7,251
post-stop laps, distinct from total lake coverage.

| Block / fuel assumption | Original CRPS | Repaired CRPS | Original cov90 | Repaired cov90 |
|---|---:|---:|---:|---:|
| 2025 all laps | 0.609909 | 0.597844 | 0.77936 | 0.93206 |
| 2025 after stops | 0.636513 | 0.621224 | 0.76264 | 0.93722 |
| 2026, 70 kg, all | 0.519400 | 0.517494 | 0.82240 | 0.93555 |
| 2026, 70 kg, after stops | 0.549641 | 0.546534 | 0.79672 | 0.93104 |
| 2026, 110 kg, all | 0.555782 | 0.549122 | 0.79965 | 0.93019 |
| 2026, 110 kg, after stops | 0.601702 | 0.592469 | 0.76155 | 0.92056 |

For 2026 under 70 kg, nominal 50/80/90/95% intervals cover
53.24/85.06/93.56/97.79%; under 110 kg, 52.34/82.75/93.02/97.16%.
The 95% intervals remain conservative. Mean 90% width under 70 kg is 3.623 s,
up from 2.402 s. MAE is unchanged because the median is unchanged.

The 70 kg paired-weekend CRPS delta is -0.001342, bootstrap 95% interval
[-0.009165, +0.006248]. Under 110 kg it is -0.006773, interval
[-0.018445, +0.003447]. Both include zero. Pooled improvement is small and not
statistically decisive on independent weekends. Round 14 of 2026 still covers
only 83.1%; several other races over-cover. The pooled gate does not promise
each future race is calibrated. No revision-2 2026 score is untouched confirmation.

The code retains between-weekend slope scatter in `FuelTrack.slope_sd` and
exposes `predictive_lap_coef()`. The historical `lap_coef()` keeps its
parameter-only semantics so original FAIL/HOLD findings remain reproducible.
The wrapper accounts for predictive discrepancy separately; do not combine
both corrections silently and reuse a fit trained on the old distribution.

## Operational API and outputs

```python
from tyretrace.infer.uncertainty import PredictiveUncertainty

# Original out-of-sample forecasts with outcomes, restricted to this method
# and strictly earlier weekends. Eight independent weekends are required.
model = PredictiveUncertainty.fit(history, target=weekend)
intervals = model.predict_forecast(raw_forecast, weekend=weekend, method="identified")
```

`raw_forecast` is the existing `fp_race.Forecast`. The prediction API returns
exact Student-t quantiles, location, scale and standard deviation without
consuming target outcomes. With four degrees of freedom, SD is scale times
sqrt(2). It rejects future history, mixed methods, invalid quantiles, wrong
method reuse and insufficient independent history. Output is marginal lap-time
distributions, not correlated race trajectories or latent wear posteriors.

```powershell
python -m tyretrace run uncertainty_t
python -m tyretrace export
python -m pytest tests/test_uncertainty.py tests/test_backtest.py -q
```

`out/artifacts/uncertainty_t/` contains exact predictive distributions and
quantiles, per-weekend comparisons and the coefficient/history manifest for
every fit, plus corresponding 110 kg sensitivity tables and original forecasts.
`out/results.json` records both scenarios, calibration curves, gate and source/
algorithm hashes. Original Gaussian HOLD outputs stay under
`out/artifacts/uncertainty/`. The runner invalidates calibration when its source
forecast input changes. The React overview displays saved coverage comparisons;
its ordinary backtest panel now uses the lake instead of only ten weekends.
