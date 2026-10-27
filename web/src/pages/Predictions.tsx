import { useState } from "react";
import { Card, Stat } from "../components/ui";
import { DataPage, DataTable, Filter, Plot, ResearchHead, type Row } from "../components/Research";
interface Prediction extends Row { driver:string; lap_in_session:number; y:number; q05:number; q50:number; q95:number; after_stop:boolean; method?:string }
interface PredictionData { weekend:string; original:Prediction[]; calibrated:Prediction[]; competitors:Prediction[] }
function Body({weekend}:{weekend:string}) {
  const [driver,setDriver]=useState(""); const [kind,setKind]=useState("calibrated");
  return <DataPage<PredictionData> file={`predictions/${weekend}.json`}>{(d) => {
    const drivers=[...new Set(d.original.map((r) => r.driver))].sort();const selected=drivers.includes(driver)?driver:drivers[0];
    const methods=[...new Set(d.competitors.map((r) => r.method!))].filter((v) => v!=='identified');
    const source=kind==="raw"?d.original:kind==="calibrated"?(d.calibrated.length?d.calibrated:d.original):d.competitors.filter((r) => r.method===kind);
    const rows=source.filter((r) => r.driver===selected).sort((a,b) => a.lap_in_session-b.lap_in_session);
    const coverage=rows.filter((r) => r.y>=r.q05 && r.y<=r.q95).length/Math.max(rows.length,1);
    return <><div className="filter-bar"><Filter label="Driver" value={selected} onChange={setDriver} options={drivers.map((v) => ({value:v,label:v}))} />
      <Filter label="Predictive distribution" value={kind} onChange={setKind} options={[{value:"calibrated",label:d.calibrated.length?"Calibrated identified":"Original · calibration unavailable"},{value:"raw",label:"Original identified"},...methods.map((v) => ({value:v,label:v.replaceAll("_"," ")}))]} /></div>
      <div className="grid g3"><Stat label="Scored laps" value={rows.length} foot="After the five declared anchor laps" /><Stat label="Observed 90% coverage" value={`${(100*coverage).toFixed(1)}%`} foot="For this driver and selection" /><Stat label="Mean interval width" value={(rows.reduce((s,r) => s+r.q95-r.q05,0)/Math.max(rows.length,1)).toFixed(2)} unit="s" /></div>
      <Plot title={`${selected} · forecasts and race-day pace`} sub="Shaded band is the central 90% marginal predictive interval" yLabel="Lap time · s" yDomain={[Math.min(...rows.map((r) => Math.min(r.q05,r.y)))-.2,Math.max(...rows.map((r) => Math.max(r.q95,r.y)))+.2]} series={[
        {name:"Observed",color:"var(--s2)",points:rows.map((r) => [r.lap_in_session,r.y])},
        {name:"Forecast median",color:"var(--s1)",points:rows.map((r) => [r.lap_in_session,r.q50])},
      ]} annotate={({x,y}) => rows.length?<path d={`M${rows.map((r) => `${x(r.lap_in_session)},${y(r.q95)}`).join("L")}L${[...rows].reverse().map((r) => `${x(r.lap_in_session)},${y(r.q05)}`).join("L")}Z`} fill="var(--accent-soft)" />:null} />
      <div className="callout">Intervals describe individual lap times, not joint race trajectories. Race strategy is conditioned on observed tyre choices. A well-covered selection does not establish latent tyre-wear recovery.</div>
      <Card title="Predictive intervals · source rows"><DataTable rows={rows} columns={[{key:"lap_in_session",label:"Lap"},{key:"y",label:"Observed · s"},{key:"q05",label:"5th percentile"},{key:"q50",label:"Median"},{key:"q95",label:"95th percentile"},{key:"after_stop",label:"After stop"}]} /></Card>
    </>;
  }}</DataPage>;
}
export default function Predictions() {
  const [weekend,setWeekend]=useState("");
  return <DataPage<{weekends:{weekend:string;calibrated:boolean;fresh?:boolean}[]}> file="predictions.json">{(d) => {
    const chosen=d.weekends.some((w) => w.weekend===weekend)?weekend:d.weekends[0]?.weekend;
    return <div className="page"><ResearchHead area="Held-out forecasts" title="See the uncertainty, lap by lap.">Forecasts use practice, qualifying, strictly earlier race history and five opening green laps per driver. Scores are retrospective on eligible filtered laps, conditional on observed tyre strategy and race distance.</ResearchHead>
      <div className="filter-bar"><Filter label="Weekend" value={chosen} onChange={setWeekend} options={d.weekends.map((w) => ({value:w.weekend,label:`${w.weekend}${w.fresh?' · unseen Austin':''}${w.calibrated?" · calibrated":""}`}))} /></div>
      {chosen && <Body key={chosen} weekend={chosen} />}</div>;
  }}</DataPage>;
}
