import type { CSSProperties, ReactNode } from "react";
import type { Gate } from "../lib/data";

export function PageHead({ eyebrow, title, children }: { eyebrow: string; title: string; children?: ReactNode }) {
  return (
    <header className="page-head">
      <div className="eyebrow">{eyebrow}</div>
      <h1 className="page-title">{title}</h1>
      {children && <p className="lead">{children}</p>}
    </header>
  );
}

export function Card({ title, sub, right, children, style, className = "" }: {
  title?: ReactNode;
  sub?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  style?: CSSProperties;
  className?: string;
}) {
  return (
    <section className={`card ${className}`} style={style}>
      {(title || right) && (
        <div className="card-head">
          <div>
            {title && <h3 className="card-title">{title}</h3>}
            {sub && <div className="card-sub">{sub}</div>}
          </div>
          {right}
        </div>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, unit, foot }: { label: string; value: ReactNode; unit?: string; foot?: ReactNode }) {
  return (
    <div className="card tile">
      <div className="tile-label">{label}</div>
      <div className="tile-value">
        {value}
        {unit && <span className="tile-unit">{unit}</span>}
      </div>
      {foot && <div className="tile-foot">{foot}</div>}
    </div>
  );
}

export function Slider({ label, value, min, max, step, onChange, format = String, help }: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  help?: string;
}) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <label className="control">
      <span className="control-row">
        <span className="control-label">{label}</span>
        <span className="control-value">{format(value)}</span>
      </span>
      <input type="range" min={min} max={max} step={step} value={value}
             style={{ "--pct": `${pct}%` } as CSSProperties}
             onChange={(e) => onChange(Number(e.target.value))} />
      {help && <span className="control-help">{help}</span>}
    </label>
  );
}

export function Switch({ label, checked, onChange, help }: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  help?: string;
}) {
  return (
    <label className="switch">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="switch-track" aria-hidden />
      <span style={{ display: "flex", flexDirection: "column", gap: 3 }}>
        <span className="control-label" style={{ color: "var(--ink)" }}>{label}</span>
        {help && <span className="control-help">{help}</span>}
      </span>
    </label>
  );
}

export function Segmented<T extends string>({ options, value, onChange, label }: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} aria-pressed={o.value === value} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** Status always ships with an icon AND a label -- never colour alone. */
const GATE_STYLE: Record<string, { color: string; icon: ReactNode; label: string }> = {
  PASS: { color: "var(--good)", label: "Pass", icon: <path d="M3.5 7.2 5.8 9.5 10.5 4.5" /> },
  SHIP: { color: "var(--good)", label: "Ship", icon: <path d="M3.5 7.2 5.8 9.5 10.5 4.5" /> },
  HOLD: { color: "var(--warning)", label: "Hold", icon: <path d="M5 4v6M9 4v6" /> },
  FAIL: { color: "var(--critical)", label: "Fail", icon: <path d="M4.5 4.5l5 5M9.5 4.5l-5 5" /> },
};

export function GateBadge({ gate }: { gate: Gate }) {
  const s = gate ? GATE_STYLE[gate] : null;
  if (!s) return <span className="badge">Not run</span>;
  return (
    <span className="badge" style={{ color: "var(--ink)" }}>
      <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden>
        <circle cx="7" cy="7" r="7" style={{ fill: s.color }} opacity={0.18} />
        <g style={{ stroke: s.color }} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" fill="none">{s.icon}</g>
      </svg>
      {s.label}
    </span>
  );
}

const COMPOUND: Record<string, { v: string; l: string }> = {
  SOFT: { v: "var(--soft)", l: "S" },
  MEDIUM: { v: "var(--medium)", l: "M" },
  HARD: { v: "var(--hard)", l: "H" },
};

/** Pirelli-style compound marker: coloured ring + letter. The letter is the
 *  secondary encoding that the white hard tyre needs. */
export function Tyre({ compound, size = 20 }: { compound: string; size?: number }) {
  const c = COMPOUND[compound] ?? { v: "var(--ink-3)", l: "?" };
  return (
    <span className="tyre" title={compound}
          style={{ width: size, height: size, fontSize: size * 0.5, color: "var(--ink)",
                   boxShadow: `inset 0 0 0 ${Math.max(2, size * 0.14)}px ${c.v}` }}>
      {c.l}
    </span>
  );
}

export const compoundColor = (c: string) => COMPOUND[c]?.v ?? "var(--ink-3)";

export function Legend({ items }: { items: { label: string; color: string; kind?: "line" | "rect" }[] }) {
  return (
    <div className="legend">
      {items.map((i) => (
        <span className="legend-item" key={i.label}>
          <span className={i.kind === "rect" ? "legend-rect" : "legend-line"} style={{ background: i.color }} />
          {i.label}
        </span>
      ))}
    </div>
  );
}

export function Loading() {
  return <div className="empty">Loading…</div>;
}

export function Missing({ what, cmd }: { what: string; cmd: string }) {
  return (
    <div className="empty">
      No data for {what} yet. Generate it with <code>{cmd}</code>, then <code>python -m tyretrace export</code>.
    </div>
  );
}
