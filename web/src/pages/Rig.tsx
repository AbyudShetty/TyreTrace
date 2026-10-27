import { useState } from "react";
import { Card, GateBadge, Stat } from "../components/ui";
import { DataPage, DataTable, Filter, Plot, ResearchHead, type Row } from "../components/Research";
interface RigLap extends Row { driver:string; lap_in_session:number; truth_fuel_kg:number; truth_wear_fl:number; truth_wear_fr:number; truth_wear_rl:number; truth_wear_rr:number }
interface Report extends Row { file:string;origin:string;layout_gate:"PASS"|"FAIL"|"HOLD";laps:number;packets:number;cars_with_fuel_and_wear:number[];problems:string[] }
interface RigData { reports:Report[];example_origin:string;laps:RigLap[];live_established:boolean }
export default function Rig() {
  const [driver,setDriver]=useState("");
  return <DataPage<RigData> file="rig.json">{(d) => {
    const drivers=[...new Set(d.laps.map((r) => r.driver))].sort();const selected=drivers.includes(driver)?driver:drivers[0];
    const rows=d.laps.filter((r) => r.driver===selected).sort((a,b) => a.lap_in_session-b.lap_in_session);
    return <div className="page"><ResearchHead area="Ground-truth rig" title="Measure what the model cannot see.">The game can expose fuel mass and per-wheel wear. Raw capture, replay and a blind estimator make the validation chain inspectable.</ResearchHead>
      <div className="callout"><GateBadge gate={d.live_established?"PASS":"HOLD"} /><span>{d.live_established?"Live validation established":"Live validation pending. Synthetic capture and reference comparisons do not establish hidden-state recovery on the game."}</span></div>
      <div className="grid g3"><Stat label="Example origin" value={d.example_origin==="synthetic"?"Synthetic":"Live capture"} foot="Provenance survives replay" /><Stat label="Example cars" value={drivers.length} foot="Telemetry availability is measured" /><Stat label="Canonical laps" value={d.laps.length.toLocaleString()} foot="Status and wear must be fresh" /></div>
      <div className="filter-bar"><Filter label="Example car" value={selected} onChange={setDriver} options={drivers.map((v) => ({value:v,label:v}))} /></div>
      <div className="grid g2"><Plot title={`${selected} · hidden fuel mass`} sub={`${d.example_origin} evidence · decoder observation`} yLabel="Fuel · kg" series={[{name:"Fuel",color:"var(--s1)",points:rows.map((r) => [r.lap_in_session,r.truth_fuel_kg])}]} />
      <Plot title="Hidden per-wheel wear" sub="Wheel order and packet offsets are independently tested" yLabel="Wear · %" series={["fl","fr","rl","rr"].map((wheel,i) => ({name:wheel.toUpperCase(),color:`var(--s${i+1})`,dash:i%2?"5 3":undefined,points:rows.map((r):[number,number] => [r.lap_in_session,Number(r[`truth_wear_${wheel}`])])}))} /></div>
      <Card title="Capture reports"><DataTable rows={d.reports} columns={[{key:"file",label:"Evidence file"},{key:"origin",label:"Origin"},{key:"layout_gate",label:"Data readiness"},{key:"laps",label:"Laps"},{key:"packets",label:"Packets"},{key:"cars_with_fuel_and_wear",label:"Truth car indices"}]} /></Card>
      <Card title="Tomorrow's live run"><ol className="checklist"><li>Start an offline race with enough laps for a tyre change; disable flashbacks.</li><li>Enable UDP to 127.0.0.1:20777, 20 Hz or higher, using supported format 2023/2024/2025.</li><li>Set telemetry privacy to Public where permitted; check which cars actually report fuel and wear.</li><li>Pass the layout preflight, then capture a session including a pit stop.</li></ol>
        <pre className="command-block">{`python -m tyretrace.rig.record --self-check\npython -m tyretrace.rig.capture --seconds 900 --out out/rig/live_001.udp\npython -m tyretrace.rig.capture --replay out/rig/live_001.udp`}</pre>
        <p className="note">Use a new capture filename. The 2026 Season Pack has a different protocol and is not silently decoded as 2025. Wear percent and degradation in seconds are different quantities.</p>
      </Card><Card title="Example source rows"><DataTable rows={rows} columns={[{key:"lap_in_session",label:"Lap"},{key:"truth_fuel_kg",label:"Fuel · kg"},{key:"truth_wear_fl",label:"Front-left · %"},{key:"truth_wear_fr",label:"Front-right · %"},{key:"truth_wear_rl",label:"Rear-left · %"},{key:"truth_wear_rr",label:"Rear-right · %"}]} /></Card>
    </div>;
  }}</DataPage>;
}
