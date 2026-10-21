import { useLayoutEffect, useRef, useState, type ReactNode } from "react";

export interface Margin { top: number; right: number; bottom: number; left: number }

export interface Inner { width: number; height: number; margin: Margin }

/**
 * A responsive SVG frame. The container's height INCLUDES the axis bands, so
 * the x-axis labels never fall outside it (dataviz anti-pattern: a fixed
 * height that clips the axis and spawns a nested scroll).
 */
export function Frame({ height, margin, children, overlay, label }: {
  height: number;
  margin: Margin;
  children: (inner: Inner) => ReactNode;
  overlay?: (inner: Inner) => ReactNode;
  label: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    // Measure synchronously before paint: relying on the first ResizeObserver
    // callback alone left charts in grid cells blank (observed in headless
    // Edge), and costs a blank frame in every browser.
    setWidth(Math.floor(el.getBoundingClientRect().width));
    const ro = new ResizeObserver(([e]) => setWidth(Math.floor(e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const inner: Inner = {
    width: Math.max(width - margin.left - margin.right, 0),
    height: height - margin.top - margin.bottom,
    margin,
  };
  return (
    <div ref={ref} style={{ position: "relative", width: "100%", height }}>
      {width > 0 && (
        <>
          <svg width={width} height={height} role="img" aria-label={label} style={{ display: "block", overflow: "visible" }}>
            <g transform={`translate(${margin.left},${margin.top})`}>{children(inner)}</g>
          </svg>
          {overlay?.(inner)}
        </>
      )}
    </div>
  );
}
