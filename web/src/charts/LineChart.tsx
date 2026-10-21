import { scaleLinear } from "d3-scale";
import { line as d3line } from "d3-shape";
import { useState, type ReactNode } from "react";
import { XAxis, YAxis } from "./axes";
import { Frame, type Inner } from "./Frame";
import { Tooltip } from "./Tooltip";

export interface Series {
  name: string;
  color: string;
  points: [number, number][];
  kind?: "line" | "dots";
  width?: number;
  opacity?: number;
  dash?: string;          // only for projections / counterfactuals, never grids
  hideInTip?: boolean;
}

/**
 * Multi-series line chart with a snapping crosshair. The tooltip lists every
 * series at the hovered x, so the pointer never has to land on a 2px line.
 */
export function LineChart({ series, height = 320, xLabel, yLabel, xFormat, yFormat, yDomain,
                            annotate, label, tipValue = (v) => v.toFixed(3) }: {
  series: Series[];
  height?: number;
  xLabel?: string;
  yLabel?: string;
  xFormat?: (v: number) => string;
  yFormat?: (v: number) => string;
  yDomain?: [number, number];
  annotate?: (s: { x: (v: number) => number; y: (v: number) => number; inner: Inner }) => ReactNode;
  label: string;
  tipValue?: (v: number) => string;
}) {
  const [hx, setHx] = useState<number | null>(null);
  const all = series.flatMap((s) => s.points);
  const xs = all.map((p) => p[0]);
  const ys = all.map((p) => p[1]);
  const xDom: [number, number] = [Math.min(...xs), Math.max(...xs)];
  const yLo = Math.min(...ys), yHi = Math.max(...ys);
  const pad = (yHi - yLo) * 0.08 || 1;
  const yDom = yDomain ?? [yLo - pad, yHi + pad];
  const margin = { top: 12, right: 18, bottom: xLabel ? 46 : 30, left: yLabel ? 58 : 46 };
  const xValues = [...new Set(xs)].sort((a, b) => a - b);

  return (
    <Frame height={height} margin={margin} label={label}
      overlay={(inner) => {
        if (hx === null) return null;
        const x = scaleLinear().domain(xDom).range([0, inner.width]);
        const rows = series.filter((s) => !s.hideInTip).flatMap((s) => {
          const p = s.points.find((q) => q[0] === hx);
          return p ? [{ color: s.color, value: tipValue(p[1]), name: s.name }] : [];
        });
        if (!rows.length) return null;
        return <Tooltip x={margin.left + x(hx)} y={margin.top + 8} title={xFormat ? xFormat(hx) : `x = ${hx}`} rows={rows} />;
      }}>
      {(inner) => {
        const x = scaleLinear().domain(xDom).range([0, inner.width]);
        const y = scaleLinear().domain(yDom).range([inner.height, 0]).nice();
        const path = d3line<[number, number]>().x((p) => x(p[0])).y((p) => y(p[1]));
        const onMove = (e: React.PointerEvent<SVGRectElement>) => {
          const rect = (e.currentTarget as SVGRectElement).getBoundingClientRect();
          const v = x.invert(e.clientX - rect.left);
          let best = xValues[0];
          for (const xv of xValues) if (Math.abs(xv - v) < Math.abs(best - v)) best = xv;
          setHx(best);
        };
        return (
          <g>
            <YAxis y={y} width={inner.width} format={yFormat} label={yLabel} />
            <XAxis x={x} height={inner.height} format={xFormat} label={xLabel} />
            {annotate?.({ x, y, inner })}
            {series.map((s) =>
              s.kind === "dots" ? (
                <g key={s.name}>
                  {s.points.map((p, i) => (
                    <circle key={i} cx={x(p[0])} cy={y(p[1])} r={4} style={{ fill: s.color, stroke: "var(--surface)" }} strokeWidth={2} opacity={s.opacity ?? 1} />
                  ))}
                </g>
              ) : (
                <path key={s.name} d={path(s.points) ?? ""} fill="none" style={{ stroke: s.color }}
                      strokeWidth={s.width ?? 2} strokeLinejoin="round" strokeLinecap="round"
                      strokeDasharray={s.dash} opacity={s.opacity ?? 1} />
              ),
            )}
            {hx !== null && (
              <line x1={x(hx)} x2={x(hx)} y1={0} y2={inner.height} style={{ stroke: "var(--ink-3)" }} strokeWidth={1} shapeRendering="crispEdges" />
            )}
            <rect width={inner.width} height={inner.height} fill="transparent"
                  onPointerMove={onMove} onPointerLeave={() => setHx(null)} />
          </g>
        );
      }}
    </Frame>
  );
}
