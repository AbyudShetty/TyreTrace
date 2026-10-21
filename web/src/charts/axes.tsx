import type { ScaleLinear, ScaleBand } from "d3-scale";

const tick = { fontSize: 11.5, fill: "var(--ink-3)", fontVariantNumeric: "tabular-nums" } as const;

/** Horizontal gridlines + left tick labels. Solid hairlines, never dashed. */
export function YAxis({ y, width, ticks = 5, format = (v: number) => String(v), label }: {
  y: ScaleLinear<number, number>;
  width: number;
  ticks?: number;
  format?: (v: number) => string;
  label?: string;
}) {
  return (
    <g>
      {y.ticks(ticks).map((t) => (
        <g key={t} transform={`translate(0,${y(t)})`}>
          <line x1={0} x2={width} style={{ stroke: "var(--grid)" }} strokeWidth={1} shapeRendering="crispEdges" />
          <text x={-8} dy="0.32em" textAnchor="end" style={tick}>{format(t)}</text>
        </g>
      ))}
      {label && (
        <text transform={`translate(${-44},${(y.range()[0] + y.range()[1]) / 2}) rotate(-90)`}
              textAnchor="middle" style={{ ...tick, fill: "var(--ink-3)", fontSize: 11.5 }}>
          {label}
        </text>
      )}
    </g>
  );
}

export function XAxis({ x, height, ticks = 6, format = (v: number) => String(v), label, grid = false }: {
  x: ScaleLinear<number, number>;
  height: number;
  ticks?: number;
  format?: (v: number) => string;
  label?: string;
  grid?: boolean;
}) {
  const [r0, r1] = x.range();
  return (
    <g>
      <line x1={r0} x2={r1} y1={height} y2={height} style={{ stroke: "var(--axis)" }} shapeRendering="crispEdges" />
      {x.ticks(ticks).map((t) => (
        <g key={t} transform={`translate(${x(t)},0)`}>
          {grid && <line y1={0} y2={height} style={{ stroke: "var(--grid)" }} shapeRendering="crispEdges" />}
          <text y={height + 18} textAnchor="middle" style={tick}>{format(t)}</text>
        </g>
      ))}
      {label && (
        <text x={(r0 + r1) / 2} y={height + 38} textAnchor="middle" style={{ ...tick, fontSize: 11.5 }}>
          {label}
        </text>
      )}
    </g>
  );
}

export function BandAxis({ x, height }: { x: ScaleBand<string>; height: number }) {
  return (
    <g>
      <line x1={0} x2={x.range()[1]} y1={height} y2={height} style={{ stroke: "var(--axis)" }} shapeRendering="crispEdges" />
      {x.domain().map((d) => (
        <text key={d} x={(x(d) ?? 0) + x.bandwidth() / 2} y={height + 18} textAnchor="middle" style={tick}>{d}</text>
      ))}
    </g>
  );
}

/** A bar with a 4px rounded data-end and a square baseline. */
export function barPath(x: number, y: number, w: number, h: number, r = 4): string {
  if (h <= 0 || w <= 0) return "";
  const rr = Math.min(r, w / 2, h);
  return `M${x},${y + h}V${y + rr}Q${x},${y} ${x + rr},${y}H${x + w - rr}Q${x + w},${y} ${x + w},${y + rr}V${y + h}Z`;
}

/** Horizontal bar, data-end on the right. */
export function hbarPath(x: number, y: number, w: number, h: number, r = 4): string {
  if (h <= 0 || w <= 0) return "";
  const rr = Math.min(r, h / 2, w);
  return `M${x},${y}H${x + w - rr}Q${x + w},${y} ${x + w},${y + rr}V${y + h - rr}Q${x + w},${y + h} ${x + w - rr},${y + h}H${x}Z`;
}
