import { useState } from "react";
import { Card, Legend, Stat, compoundColor } from "../components/ui";
import { DataPage, DataTable, Filter, Plot, ResearchHead } from "../components/Research";

interface Lap { driver: string; lap_in_session: number; lap_time_s: number | null; tyre_age: number; compound_rel: string; stint_id: number; gap_ahead_s: number | null; dirty_air: number | null; pit_in: boolean; pit_out: boolean; is_valid: boolean; track_status: string }
interface Weekend { weekend: string; label: string; year: number; circuit: string; laps: number; drivers: number; dry: boolean; nomination: Record<string,string|null> }
interface Index { weekends: Weekend[]; race_laps: number; mapped_weekends: number }
interface Race { weekend: string; label: string; laps: Lap[]; nomination: Record<string,string|null> }

function Stints({ laps }: { laps: Lap[] }) {
  const drivers = [...new Set(laps.map((r) => r.driver))].sort(); const max = Math.max(...laps.map((r) => r.lap_in_session));
  return <svg viewBox={`0 0 960 ${drivers.length * 25 + 35}`} role="img" aria-label="Tyre strategies by driver and race lap" className="stint-plot">
    {Array.from({ length: Math.ceil(max / 10) }, (_, i) => (i + 1) * 10).map((lap) => <g key={lap}>
      <line x1={70 + lap / max * 860} x2={70 + lap / max * 860} y1={5} y2={drivers.length * 25} stroke="var(--grid)" />
      <text x={70 + lap / max * 860} y={drivers.length * 25 + 23} fill="var(--ink-3)" fontSize={11}>{lap}</text></g>)}
    {drivers.map((driver, i) => <g key={driver}><text x={4} y={i * 25 + 17} fill="var(--ink-2)" fontSize={12}>{driver}</text>
      {[...new Set(laps.filter((r) => r.driver === driver).map((r) => r.stint_id))].map((id) => {
        const stint = laps.filter((r) => r.driver === driver && r.stint_id === id); const start = Math.min(...stint.map((r) => r.lap_in_session)); const end = Math.max(...stint.map((r) => r.lap_in_session)); const comp = stint[0].compound_rel;
        return <g key={id}><rect x={70 + (start - 1) / max * 860} y={i * 25 + 4} width={Math.max((end - start + 1) / max * 860 - 2, 3)} height={17} rx={3} fill={compoundColor(comp)} opacity={.85}>
          <title>{driver} · {comp} · laps {start}–{end}</title></rect><text x={75 + (start - 1) / max * 860} y={i * 25 + 16} fill="var(--page)" fontSize={10} fontWeight={700}>{comp.slice(0, 1)}</text></g>;
      })}</g>)}
  </svg>;
}
function RaceBody({ weekend }: { weekend: string }) {
  const [driver, setDriver] = useState(""); const [mode, setMode] = useState("racing");
  return <DataPage<Race> file={`races/${weekend}.json`}>{(d) => {
    const drivers = [...new Set(d.laps.map((r) => r.driver))].sort(); const selected = drivers.includes(driver) ? driver : drivers[0];
    const rows = d.laps.filter((r) => r.driver === selected && r.lap_time_s !== null && (mode === "all" || (r.is_valid && !r.pit_in && !r.pit_out && r.track_status === "1")));
    const stints = drivers.flatMap((driver) => [...new Set(d.laps.filter((r) => r.driver === driver).map((r) => r.stint_id))].map((stint) => {
      const laps = d.laps.filter((r) => r.driver === driver && r.stint_id === stint);
      return { driver, stint, compound: laps[0].compound_rel, first_lap: Math.min(...laps.map((r) => r.lap_in_session)), last_lap: Math.max(...laps.map((r) => r.lap_in_session)) };
    }));
    return <><div className="filter-bar"><Filter label="Driver" value={selected} onChange={setDriver} options={drivers.map((v) => ({ value: v, label: v }))} />
      <Filter label="Lap selection" value={mode} onChange={setMode} options={[{ value: "racing", label: "Valid green laps" }, { value: "all", label: "All timed laps" }]} /></div>
      <div className="grid g2"><Plot title={`${selected} · lap-time trace`} sub={`${rows.length} laps shown · pit and neutralized laps depend on your selection`} yLabel="Lap time · s" series={[{ name: selected, color: "var(--s1)", points: rows.map((r) => [r.lap_in_session, r.lap_time_s!]) }]} />
      <Plot title="Traffic exposure" sub="Derived from public timing gaps · one unit denotes full wake" yLabel="Dirty-air exposure" yDomain={[0, 1]} series={[{ name: "Dirty air", color: "var(--s2)", points: rows.filter((r) => r.dirty_air !== null).map((r) => [r.lap_in_session, r.dirty_air!]) }]} /></div>
      <Card title="The race, one tyre set at a time" sub="Hover a stint for compound and lap range"><Stints laps={d.laps} /><Legend items={[...new Set(d.laps.map((r) => r.compound_rel))].sort().map((c) => ({ label: c, color: compoundColor(c), kind: "rect" }))} /><details><summary>Tyre stints · table and CSV</summary><DataTable rows={stints} /></details></Card>
      <Card title="Selected laps · source table"><DataTable rows={rows.map((r) => ({ ...r }))} columns={[{ key: "lap_in_session", label: "Lap" }, { key: "lap_time_s", label: "Time · s" }, { key: "compound_rel", label: "Compound" }, { key: "tyre_age", label: "Age" }, { key: "stint_id", label: "Stint" }, { key: "gap_ahead_s", label: "Gap ahead · s" }, { key: "dirty_air", label: "Dirty-air exposure" }]} /></Card>
    </>;
  }}</DataPage>;
}
export default function Races() {
  const [chosen, setChosen] = useState("");
  return <DataPage<Index> file="races.json">{(d) => {
    const weekend = d.weekends.find((w) => w.weekend === chosen) ?? d.weekends[0];
    return <div className="page"><ResearchHead area="Race explorer" title="Every stint tells a story.">Explore tyre choices, measured pace and traffic across the grid. The lake includes wet races; dry-model evaluations exclude them.</ResearchHead>
      <div className="filter-bar"><Filter label="Race weekend" value={weekend.weekend} onChange={setChosen} options={d.weekends.map((w) => ({ value: w.weekend, label: w.label }))} /><span className={`status-pill ${weekend.dry ? "" : "warn"}`}>{weekend.dry ? "Dry race" : "Wet / mixed · excluded from model scoring"}</span></div>
      <div className="grid g4"><Stat label="Race laps" value={weekend.laps.toLocaleString()} foot="All recorded drivers" /><Stat label="Drivers" value={weekend.drivers} foot={weekend.circuit} />
        <Stat label="Nomination" value={Object.values(weekend.nomination).filter(Boolean).join(" / ") || "Unresolved"} foot="Hard / medium / soft · supplier sourced" /><Stat label="Lake" value={d.weekends.length} unit="weekends" foot={`${d.race_laps.toLocaleString()} race laps`} /></div>
      <RaceBody key={weekend.weekend} weekend={weekend.weekend} /></div>;
  }}</DataPage>;
}
