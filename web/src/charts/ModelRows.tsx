import { scaleLinear } from "d3-scale";
import { useState } from "react";
import { hbarPath } from "./axes";
import { Frame } from "./Frame";
import { Tooltip } from "./Tooltip";

export interface ModelRow {
  id: string;
  label: string;
  sub?: string;
  crps: number;
  cov90: number;
  mae?: number;
  rhat?: number | null;
  color: string;
  group: string;
}

const ROW = 34;
const BAR = 14;

/**
 * One row per model, two aligned panels: forecast error (CRPS, a magnitude,
 * so bars from zero) and 90% coverage (a position against the ship band, so
 * a dot -- a bar from zero would imply 0% coverage is a baseline). Two
 * panels, never two axes on one plot.
 */
export function ModelRows({ rows, band }: { rows: ModelRow[]; band: [number, number] }) {
  const [hover, setHover] = useState<number | null>(null);
  const groups = rows.map((r, i) => (i > 0 && r.group !== rows[i - 1].group ? 1 : 0));
  const offsets = groups.reduce<number[]>((acc, g, i) => [...acc, (acc[i - 1] ?? 0) + g], []);
  const height = rows.length * ROW + offsets[offsets.length - 1] * 14 + 56;
  const margin = { top: 24, right: 16, bottom: 32, left: 150 };

  return (
    <Frame height={height} margin={margin} label="Forecast error and calibration for every model"
      overlay={(inner) => {
        if (hover === null) return null;
        const r = rows[hover];
        const y = margin.top + hover * ROW + offsets[hover] * 14;
        return (
          <Tooltip x={margin.left + inner.width * 0.3} y={y} title={`${r.label}${r.sub ? ` · ${r.sub}` : ""}`}
            rows={[
              { value: r.crps.toFixed(3), name: "CRPS, seconds", color: r.color },
              { value: `${(r.cov90 * 100).toFixed(1)}%`, name: "inside the 90% interval" },
              ...(r.mae != null ? [{ value: r.mae.toFixed(3), name: "median error, s" }] : []),
              ...(r.rhat != null ? [{ value: r.rhat.toFixed(3), name: r.rhat > 1.05 ? "r̂ — not converged" : "r̂" }] : []),
            ]} />
        );
      }}>
      {({ width, height: h }) => {
        const gap = 40;
        const wA = Math.round((width - gap) * 0.56);
        const wB = width - gap - wA;
        const xA = scaleLinear().domain([0, Math.max(...rows.map((r) => r.crps)) * 1.18]).range([0, wA]).nice();
        const xB = scaleLinear().domain([0.3, 1]).range([0, wB]);
        const yOf = (i: number) => i * ROW + offsets[i] * 14;
        const bx = wA + gap;
        return (
          <g>
            {/* panel titles */}
            <text x={0} y={-10} style={{ fontSize: 11.5, fontWeight: 600, fill: "var(--ink-2)" }}>Forecast error (CRPS, s) — lower is better</text>
            <text x={bx} y={-10} style={{ fontSize: 11.5, fontWeight: 600, fill: "var(--ink-2)" }}>Share inside the 90% interval</text>

            {/* grids */}
            {xA.ticks(4).map((t) => (
              <g key={`a${t}`} transform={`translate(${xA(t)},0)`}>
                <line y1={0} y2={h} style={{ stroke: "var(--grid)" }} shapeRendering="crispEdges" />
                <text y={h + 18} textAnchor="middle" style={{ fontSize: 11.5, fill: "var(--ink-3)" }} className="tnum">{t.toFixed(1)}</text>
              </g>
            ))}
            <rect x={bx + xB(band[0])} width={xB(band[1]) - xB(band[0])} y={0} height={h} style={{ fill: "var(--band-fill)" }} />
            <text x={bx + xB((band[0] + band[1]) / 2)} y={h + 18} textAnchor="middle" style={{ fontSize: 11.5, fill: "var(--ink-2)", fontWeight: 600 }}>ship band</text>
            {[0.4, 0.6, 0.8, 1].map((t) => (
              <g key={`b${t}`} transform={`translate(${bx + xB(t)},0)`}>
                <line y1={0} y2={h} style={{ stroke: "var(--grid)" }} shapeRendering="crispEdges" />
                {t !== 0.8 && t !== 1 && <text y={h + 18} textAnchor="middle" style={{ fontSize: 11.5, fill: "var(--ink-3)" }} className="tnum">{t * 100}%</text>}
              </g>
            ))}
            <line x1={bx + xB(0.9)} x2={bx + xB(0.9)} y1={0} y2={h} style={{ stroke: "var(--good)" }} strokeOpacity={0.6} />

            {rows.map((r, i) => {
              const y = yOf(i);
              const cy = y + ROW / 2;
              const on = hover === i;
              return (
                <g key={r.id} onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)}>
                  <rect x={-margin.left} y={y} width={width + margin.left} height={ROW} fill="transparent" />
                  {on && <rect x={-margin.left + 4} y={y + 2} width={width + margin.left - 4} height={ROW - 4} rx={6} style={{ fill: "var(--surface-2)" }} />}
                  <text x={-14} y={cy} dy="0.32em" textAnchor="end" style={{ fontSize: 13, fontWeight: 560, fill: "var(--ink)" }}>{r.label}</text>
                  {r.sub && <text x={-14} y={cy + 13} dy="0.32em" textAnchor="end" style={{ fontSize: 10.5, fill: "var(--ink-3)" }}>{r.sub}</text>}
                  <path d={hbarPath(0, cy - BAR / 2, xA(r.crps), BAR)} style={{ fill: r.color }} />
                  <text x={xA(r.crps) + 6} y={cy} dy="0.32em" style={{ fontSize: 12, fontWeight: 600, fill: "var(--ink-2)" }} className="tnum">{r.crps.toFixed(3)}</text>
                  <line x1={bx} x2={bx + xB(r.cov90)} y1={cy} y2={cy} style={{ stroke: "var(--grid)" }} strokeWidth={2} />
                  <circle cx={bx + xB(r.cov90)} cy={cy} r={5.5} style={{ fill: r.color, stroke: "var(--surface)" }} strokeWidth={2} />
                  <text x={bx + xB(r.cov90) + (r.cov90 > 0.9 ? -10 : 10)} y={cy} dy="0.32em"
                        textAnchor={r.cov90 > 0.9 ? "end" : "start"}
                        style={{ fontSize: 12, fill: "var(--ink-2)" }} className="tnum">{(r.cov90 * 100).toFixed(1)}%</text>
                </g>
              );
            })}
          </g>
        );
      }}
    </Frame>
  );
}
