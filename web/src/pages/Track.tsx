import { useState } from "react";
import { Card, Stat } from "../components/ui";
import { DataPage, DataTable, Filter, Plot, ResearchHead, type Row } from "../components/Research";
interface Session { session_uid: string; weekend: string; tau_per_min: number; se: number; z: number; total_s: number; n_laps: number }
interface TrackData { sessions: Session[]; nominations: Row[] }
export default function Track() {
  const [year, setYear] = useState("all");
  return <DataPage<TrackData> file="track.json">{(d) => {
    const rows = d.sessions.filter((r) => year === "all" || r.weekend.startsWith(year)).sort((a,b) => a.weekend.localeCompare(b.weekend));
    return <div className="page"><ResearchHead area="Track evolution" title="Separate the track from the tyre.">Qualifying supplies a low-fuel anchor for track evolution. Transporting that rate to practice or the race remains an assumption, with a separately estimated race scaling.</ResearchHead>
      <div className="filter-bar"><Filter label="Season" value={year} onChange={setYear} options={[{ value: "all", label: "All seasons" }, ...["2023","2024","2025","2026"].map((v) => ({ value:v,label:v }))]} /></div>
      <div className="grid g3"><Stat label="Qualifying sessions" value={rows.length} foot="Usable within-driver push laps" /><Stat label="Negative evolution" value={`${Math.round(100*rows.filter((r) => r.tau_per_min<0).length/Math.max(rows.length,1))}%`} foot="Negative means the track gets faster" /><Stat label="Sourced nominations" value={d.nominations.length} foot="Season-specific constructions" /></div>
      <Plot title="Qualifying track-evolution estimates" sub="Point estimates and 90% parameter intervals · this is not a race forecast" xLabel="Weekend" yLabel="Evolution · s / min" xFormat={(i) => rows[Math.round(i)]?.weekend ?? ""} series={[
        { name:"Estimate",color:"var(--s1)",points:rows.map((r,i) => [i,r.tau_per_min]) },
        { name:"90% lower",color:"var(--ink-3)",dash:"4 4",points:rows.map((r,i) => [i,r.tau_per_min-1.645*r.se]) },
        { name:"90% upper",color:"var(--s2)",dash:"4 4",points:rows.map((r,i) => [i,r.tau_per_min+1.645*r.se]) },
      ]} />
      <Card title="Session estimates"><DataTable rows={rows.map((r) => ({...r}))} columns={[{key:"weekend",label:"Weekend"},{key:"tau_per_min",label:"τ · s/min"},{key:"se",label:"Standard error"},{key:"z",label:"z"},{key:"total_s",label:"Session change · s"},{key:"n_laps",label:"Push laps"}]} /></Card>
      <Card title="Physical tyre nominations" sub="Unresolved weekends are never silently assigned the legacy C3 placeholder"><DataTable rows={d.nominations.filter((r) => year === "all" || String(r.year) === year)} columns={[{key:"year",label:"Season"},{key:"event",label:"Event"},{key:"hard",label:"Hard"},{key:"medium",label:"Medium"},{key:"soft",label:"Soft"},{key:"source_date",label:"Source date"},{key:"source",label:"Primary source"}]} /></Card>
    </div>;
  }}</DataPage>;
}
