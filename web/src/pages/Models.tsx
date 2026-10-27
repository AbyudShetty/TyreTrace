import { useState } from "react";
import { Card, GateBadge, Stat } from "../components/ui";
import { DataPage, DataTable, Filter, Plot, ResearchHead, type Row } from "../components/Research";
interface Calibration { rung: string; level: number; coverage: number }
interface Posterior { rung:string; parameter:string; mean:number; q05:number; q95:number; bins:[number,number][] }
interface ModelsData { calibration:Calibration[]; posterior:Posterior[]; contrasts:Row[]; rolling:{ summary?:Row[]; comparison?:Row[]; selected_on_2025?:string }; rolling_curves?: (Calibration & {method:string;year:string;subset:string})[]; online_eligibility?:{summary?:Row[]};online_curves?:Row[]; diagnostics:Row[] }
export default function Models() {
  const [rung,setRung]=useState("L3"); const [parameter,setParameter]=useState("tau_scale");
  return <DataPage<ModelsData> file="models.json">{(d) => {
    const post=d.posterior.find((r) => r.rung===rung && r.parameter===parameter);
    return <div className="page"><ResearchHead area="Models & evidence" title="Complexity has to earn its place.">Inspect calibration, parameter uncertainty and rolling model comparisons. Early-to-late race diagnostics and practice-to-race forecasts answer different questions.</ResearchHead>
      <Plot title="Raw ladder calibration" sub="Original ten-weekend diagnostic · nominal coverage on the diagonal" xLabel="Nominal interval" yLabel="Observed coverage" yDomain={[0,1]} xFormat={(v) => `${Math.round(v*100)}%`} yFormat={(v) => `${Math.round(v*100)}%`} series={[
        {name:"Ideal",color:"var(--ink-3)",dash:"4 4",points:[[0,0],[1,1]]},
        ...["L2","L3","L4","L5"].map((r,i) => ({name:r,color:`var(--s${i+1})`,points:d.calibration.filter((v) => v.rung===r).map((v):[number,number] => [v.level,v.coverage])})),
      ]} />
      <div className="callout"><GateBadge gate="HOLD" /> Raw L3 did not replicate on 61 additional weekends. The marginal predictive calibration layer is evaluated separately; it does not validate latent fuel or tyre parameters.</div>
      <div className="filter-bar"><Filter label="Model" value={rung} onChange={setRung} options={["L2","L3","L4","L5"].map((v) => ({value:v,label:v}))} />
        <Filter label="Parameter" value={parameter} onChange={setParameter} options={[...new Set(d.posterior.map((p) => p.parameter))].map((v) => ({value:v,label:v.replaceAll("_"," ")}))} /></div>
      {post && <><div className="grid g3"><Stat label="Posterior mean" value={post.mean.toFixed(4)} /><Stat label="90% lower" value={post.q05.toFixed(4)} /><Stat label="90% upper" value={post.q95.toFixed(4)} /></div>
        <Plot title={`${rung} · ${parameter.replaceAll("_"," ")}`} yLabel="Posterior draw count" xLabel="Parameter value" series={[{name:parameter,color:"var(--s1)",points:post.bins}]}><details><summary>Histogram values</summary><DataTable rows={post.bins.map(([value,count])=>({value,count}))} /></details></Plot></>}
      <Card title="Rolling practice models and baseline families" sub="Same temporal inputs · retrospective scores · candidate names do not imply a ship verdict"><DataTable rows={d.rolling.summary ?? []} /></Card>
      <Card title="Candidate gates against calibrated identified forecasts" sub={`Selected using 2025: ${d.rolling.selected_on_2025??'audit pending'}. The 2026 check is retrospective; prospective replication still needs new races.`}><DataTable rows={d.rolling.comparison??[]} /></Card>
      {!!d.rolling_curves?.length && <Plot title="Rolling predictive calibration · 2026" sub="All laps · each method uses its own earlier-weekend calibration" xLabel="Nominal interval" yLabel="Observed coverage" yDomain={[0,1]} series={[
        {name:"Ideal",color:"var(--ink-3)",dash:"4 4",points:[[0,0],[1,1]]},
        ...["identified","ml","published","robust_practice","regime_practice"].map((m,i) => ({name:m.replaceAll('_',' '),color:`var(--s${i+1})`,dash:['','5 3','2 2','7 3 2 3','3 5'][i],points:(d.rolling_curves??[]).filter((r) => r.method===m&&r.year==='2026'&&r.subset==='all').sort((a,b) => a.level-b.level).map((r):[number,number]=>[r.level,r.coverage])})),
      ]} />}
      <details className="card"><summary>Calibration values · accessible tables</summary><DataTable rows={d.calibration as unknown as Row[]} /><DataTable rows={(d.rolling_curves??[]) as unknown as Row[]} limit={100} /></details>
      <Card title="Flag-only race eligibility sensitivity" sub="Retrospective expanded population; no whole-stint pace trimming can change opening anchors. Known tyre strategy and distance remain conditioned inputs."><DataTable rows={d.online_eligibility?.summary??[]} /><details><summary>Full calibration values</summary><DataTable rows={d.online_curves??[]} /></details></Card>
      <Card title="Within-weekend compound contrasts" sub="E1 contrasts need fewer assumptions than E2 absolute degradation"><DataTable rows={d.contrasts} /></Card>
      <details className="card"><summary>Regime-model diagnostics and state selection</summary><p className="note">Two versus three pace states is selected using held-out practice likelihood. Labels are not verified physical wear states.</p><DataTable rows={d.diagnostics} limit={25} /></details>
    </div>;
  }}</DataPage>;
}
