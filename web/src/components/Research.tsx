import type { ComponentProps, ReactNode } from "react";
import { useData } from "../lib/data";
import { Card, Legend, Loading, Missing, PageHead } from "./ui";
import { LineChart, type Series } from "../charts/LineChart";

export type Row = Record<string, unknown>;
export function DataPage<T>({ file, children }: { file: string; children: (data: T) => ReactNode }) {
  const load = useData<T>(file);
  if (load.state === "loading") return <div className="page"><Loading /></div>;
  if (load.state === "error") return <div className="page"><Missing what="this analysis" cmd="python -m tyretrace export" /></div>;
  return <>{children(load.data)}</>;
}
export function ResearchHead({ area, title, children }: { area: string; title: string; children: ReactNode }) {
  return <PageHead eyebrow={area} title={title}>{children}</PageHead>;
}
function text(value: unknown) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(4);
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}
export function DataTable({ rows, columns, caption, limit = 100 }: {
  rows: Row[]; columns?: { key: string; label: string }[]; caption?: string; limit?: number;
}) {
  const cols = columns ?? [...new Set(rows.flatMap(Object.keys))].map((key) => ({ key, label: key.replaceAll("_", " ") }));
  function download() {
    const csv = [cols.map((c) => c.label), ...rows.map((r) => cols.map((c) => {
      const v = r[c.key]; const s = v===null||v===undefined?'':typeof v==='object'?JSON.stringify(v):String(v);
      return typeof v === "string" && /^[=+@-]/.test(s) ? `'${s}` : s;
    }))].map((row) => row.map((v) => `"${v.replaceAll('"', '""')}"`).join(",")).join("\r\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const a = document.createElement("a"); a.href = url; a.download = "tyre-trace-analysis.csv"; a.click(); URL.revokeObjectURL(url);
  }
  return <div className="data-table-wrap">
    <div className="table-toolbar"><span className="note">{rows.length.toLocaleString()} rows{rows.length > limit ? ` · first ${limit} shown` : ""}</span>
      <button className="icon-btn" onClick={download} disabled={!rows.length}>Download CSV ↗</button></div>
    <div className="table-scroll"><table className="table">{caption && <caption>{caption}</caption>}
      <thead><tr>{cols.map((c) => <th key={c.key}>{c.label}</th>)}</tr></thead>
      <tbody>{rows.slice(0, limit).map((r, i) => <tr key={i}>{cols.map((c) => <td key={c.key} className={typeof r[c.key] === "number" ? "num" : ""}>{typeof r[c.key]==='string'&&/^https:\/\//.test(String(r[c.key]))?<a href={String(r[c.key])} target="_blank" rel="noopener noreferrer">{text(r[c.key])} ↗</a>:text(r[c.key])}</td>)}</tr>)}</tbody>
    </table></div>{!rows.length && <p className="empty">This audit has no saved rows yet.</p>}
  </div>;
}
export function Plot({ title, sub, series, xLabel = "Lap", yLabel, height = 300, children, ...props }: {
  title: string; sub?: string; series: Series[]; xLabel?: string; yLabel: string; height?: number; children?: ReactNode;
} & Pick<ComponentProps<typeof LineChart>, "annotate" | "yDomain" | "xFormat" | "yFormat">) {
  return <Card title={title} sub={sub}>
    {series.some((s) => s.points.length) ? <><LineChart series={series} height={height} xLabel={xLabel} yLabel={yLabel} label={title} {...props} />
      {series.length > 1 && <Legend items={series.filter((s) => !s.hideInTip).map((s) => ({ label: s.name, color: s.color }))} />}</> : <p className="empty">No valid points in this selection.</p>}
    {children}
  </Card>;
}
export function Filter({ label, value, onChange, options }: { label: string; value: string; onChange: (v: string) => void; options: { value: string; label: string }[] }) {
  return <label className="filter-control"><span>{label}</span><select value={value} onChange={(e) => onChange(e.target.value)}>
    {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}</select></label>;
}
