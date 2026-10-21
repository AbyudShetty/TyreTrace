export interface TipRow { color?: string; value: string; name: string }

/** Values lead, names follow; line keys, not boxes. Rendered with React
 *  text nodes, so labels from data can never inject markup. */
export function Tooltip({ x, y, title, rows }: { x: number; y: number; title?: string; rows: TipRow[] }) {
  return (
    <div className="tip" style={{ left: x, top: y }} role="status">
      {title && <div className="tip-title">{title}</div>}
      {rows.map((r, i) => (
        <div className="tip-row" key={i}>
          {r.color && <span className="tip-key" style={{ background: r.color }} />}
          <span className="tip-val">{r.value}</span>
          <span className="tip-name">{r.name}</span>
        </div>
      ))}
    </div>
  );
}
