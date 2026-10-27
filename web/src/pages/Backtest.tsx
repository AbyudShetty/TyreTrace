import { Card, Stat, GateBadge } from "../components/ui";
import { DataPage, DataTable, Plot, ResearchHead, type Row } from "../components/Research";
interface Evaluation { backtest:Record<string,number>; weekends:(Row & {weekend:string;method:string;crps_after:number})[]; ablations:Row[]; postrace:Record<string,number>; strategy:{methods?:Row[]}; rolling:{summary?:Row[]}; headline:Record<string,unknown>; fresh_holdout?:{summary?:Row[];model_gate?:string;scope?:string;audit_gate?:string};fresh_curves?:Row[] }
export default function Backtest() {
  return <DataPage<Evaluation> file="evaluation.json">{(d) => {
    const keys=[...new Set(d.weekends.map((r) => r.weekend))].sort();
    const methods=["identified","fuel-only","no-track","conventional","naive"];
    const rows=methods.map((m) => {const k=m.replaceAll("-","_");return {method:m,crps:d.backtest[`crps_after_${k}`],coverage:d.backtest[`cov90_after_${k}`],bias:d.backtest[`bias_after_${k}`]};});
    return <div className="page"><ResearchHead area="Practice → race" title="The test that counts.">Rolling forecasts use only earlier weekends for priors. All comparisons use the same target laps and declared early-race anchors. Lower CRPS means a better predictive distribution.</ResearchHead>
      <div className="grid g3"><Stat label="Evaluated weekends" value={d.backtest.weekends??"—"} foot="Dry races with sufficient earlier history" /><Stat label="Post-stop laps" value={d.backtest.n_after_identified?.toLocaleString()??"—"} foot="Where the fuel / wear split matters" /><Stat label="Primary metric" value="CRPS" unit="s" foot="Coverage remains a separate ship gate" /></div>
      <Card title="Original rolling comparison · laps after a stop"><DataTable rows={rows} /></Card>
      <Plot title="Performance across weekends" sub="Original uncalibrated methods · individual races vary" xLabel="Weekend" yLabel="Post-stop CRPS · s" xFormat={(v) => keys[Math.round(v)]??""} series={[
        {name:"Identified",color:"var(--s1)",points:d.weekends.filter((r) => r.method==="identified").sort((a,b) => a.weekend.localeCompare(b.weekend)).map((r) => [keys.indexOf(r.weekend),r.crps_after])},
        {name:"Conventional",color:"var(--s2)",points:d.weekends.filter((r) => r.method==="conventional").sort((a,b) => a.weekend.localeCompare(b.weekend)).map((r) => [keys.indexOf(r.weekend),r.crps_after])},
      ]} />
      <Card title="All baseline families and new practice candidates" sub="Calibrated comparisons are retrospective; more complex methods are not automatically adopted"><DataTable rows={d.rolling.summary??[]} /></Card>
      {!!d.fresh_holdout?.summary?.length && <Card title="Unseen Austin holdout · frozen before timing access" sub="One independent offline weekend · flag-only eligibility · original lake preserved">
        <div className="callout"><GateBadge gate={d.fresh_holdout.audit_gate==='PASS'&&d.fresh_holdout.model_gate==='SHIP'?'SHIP':'HOLD'} /> {d.fresh_holdout.audit_gate==='PASS'?'The 2025-selected ML method must beat every comparator on all-lap CRPS and keep 90% coverage in range. One weekend does not establish prospective replication.':'The audit is incomplete; these saved scores do not establish a current model verdict.'}</div>
        <DataTable rows={d.fresh_holdout.summary} />
        <details><summary>Full marginal calibration values</summary><DataTable rows={d.fresh_curves??[]} /></details>
      </Card>}
      <Card title="Where forecast errors come from" sub="RMS contributions are correlated; they should not be added as independent variances"><div className="grid g4">{["tyres","slope","traffic","noise"].map((part) => <Stat key={part} label={part} value={d.postrace[`identified_rms_${part}`]?.toFixed(3)??"—"} unit="s" />)}</div></Card>
      <Card title="Every component has to contribute" sub="Ablations preserve reversals as well as improvements"><DataTable rows={d.ablations} /></Card>
      <Card title="Conditional one-stop decision diagnostic" sub="Fitted post-race reference, fixed compound order and fresh sets; not observed counterfactual truth"><DataTable rows={d.strategy.methods??[]} /></Card>
      <details className="card"><summary>2026 fuel-load sensitivity · full audit numbers</summary><DataTable rows={Object.entries(d.headline).map(([metric,value]) => ({metric,value}))} /></details>
    </div>;
  }}</DataPage>;
}
