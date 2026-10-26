import { Card, GateBadge } from "./ui";
import { fx, int, pct } from "../lib/format";
import type { UncertaintySummary } from "../lib/data";

/** Saved audit only: every interval and verdict was scored before export. */
export function UncertaintyPanel({ data }: { data?: UncertaintySummary | null }) {
  const result = data?.phases.confirmation;
  if (!data || !result) return null;
  const scenarios = [
    { label: "70 kg assumption", comparison: result },
    ...(data.fuel_sensitivity_110
      ? [{ label: "110 kg assumption", comparison: data.fuel_sensitivity_110 }] : []),
  ];
  return (
    <Card title="Predictive intervals, checked across race weekends"
      sub={`2026 · ${result.paired_weekends.n} weekends · ${int(result.all.calibrated.n)} scored laps · rolling history`}
      right={<GateBadge gate={data.gate} />}>
      <p className="note" style={{ marginBottom: 16 }}>
        The interval learns how forecast errors grow through a race and on older tyres, including rare large
        misses. Every fit uses earlier weekends. Point forecasts stay the same.
      </p>
      <div style={{ overflowX: "auto" }}>
        <table className="table">
          <thead><tr><th>Evaluation</th><th className="num">Original CRPS</th><th className="num">Calibrated CRPS</th>
            <th className="num">Original 90% coverage</th><th className="num">Calibrated coverage</th></tr></thead>
          <tbody>{scenarios.flatMap(({ label, comparison }) => (["all", "after_stop"] as const).map((subset) => {
            const row = comparison[subset];
            return <tr key={`${label}-${subset}`}>
              <td>{label} · {subset === "all" ? "all laps" : "after a stop"}</td>
              <td className="num">{fx(row.original.crps, 4)}</td><td className="num">{fx(row.calibrated.crps, 4)}</td>
              <td className="num">{pct(row.original.coverage_90, 1)}</td><td className="num">{pct(row.calibrated.coverage_90, 1)}</td>
            </tr>;
          }))}</tbody>
        </table>
      </div>
      <details style={{ marginTop: 16 }}>
        <summary style={{ cursor: "pointer" }}>Calibration at every reported interval level</summary>
        <table className="table" style={{ marginTop: 10 }}>
          <thead><tr><th>Nominal interval</th><th className="num">Observed coverage · 70 kg</th><th className="num">Observed coverage · 110 kg</th></tr></thead>
          <tbody>{[0.5, 0.8, 0.9, 0.95].map((level) => <tr key={level}>
            <td>{pct(level)}</td><td className="num">{pct(result.all.calibrated.calibration?.[String(level)] ?? 0, 1)}</td>
            <td className="num">{data.fuel_sensitivity_110?.all.calibrated.calibration
              ? pct(data.fuel_sensitivity_110.all.calibrated.calibration[String(level)], 1) : "—"}</td>
          </tr>)}</tbody>
        </table>
      </details>
      <p className="note" style={{ marginTop: 14 }}>
        The ship band is 85–95% coverage at the 90% level, with lower CRPS on all laps and after stops.
        Wider intervals account for forecast error that the original parameter intervals missed.
        At least {data.min_history_weekends} earlier forecast weekends are required.
      </p>
      <p className="note" style={{ marginTop: 8 }}>
        This is retrospective validation. The small CRPS gain is uncertain across weekends: its bootstrap
        interval includes zero. The 95% intervals remain conservative. These are marginal lap-time
        intervals; fuel and tyre-parameter intervals need separate validation.
      </p>
    </Card>
  );
}
