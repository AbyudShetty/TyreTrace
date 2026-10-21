import { scaleBand, scaleLinear } from "d3-scale";
import { useDeferredValue, useMemo, useState } from "react";
import { barPath, YAxis } from "../charts/axes";
import { Frame } from "../charts/Frame";
import { Tooltip } from "../charts/Tooltip";
import { Card, compoundColor, Legend, PageHead, Segmented, Slider, Switch, Tyre } from "../components/ui";
import { fx, pct } from "../lib/format";
import {
  DEFAULT_PRIOR, eig, EMPTY, OUT_IN_LAPS, plan, posteriorSd, programmeOf, runLabel,
  runMinutes, standardProgramme, SWAP_LAPS, TARGETS, type Prior, type Programme,
} from "../lib/planner/eig";

const TARGET_LABEL: Record<string, string> = {
  "degradation per compound": "Degradation per compound",
  "SOFT − HARD contrast": "Soft vs hard difference",
  "degradation + fuel effect": "Degradation + fuel effect",
};

interface Seg { t0: number; t1: number; compound: string; label: string; run: number }

function segments(g: Programme, p: Prior): Seg[] {
  const lapMin = p.lap_s / 60;
  return g.runs.flatMap((r, i) => {
    const t0 = g.starts[i];
    const t1 = t0 + runMinutes(r, p);
    if (!r.swap_to) return [{ t0, t1, compound: r.compound, label: runLabel(r), run: i }];
    const mid = t0 + (OUT_IN_LAPS / 2 + Math.floor(r.laps / 2) + SWAP_LAPS / 2) * lapMin;
    return [{ t0, t1: mid, compound: r.compound, label: runLabel(r), run: i },
            { t0: mid, t1, compound: r.swap_to, label: runLabel(r), run: i }];
  });
}

function Timeline({ rows, session, p }: { rows: { name: string; prog: Programme }[]; session: number; p: Prior }) {
  const [hover, setHover] = useState<{ s: Seg; row: number } | null>(null);
  const end = Math.max(session, ...rows.map((r) => (r.prog.runs.length ? r.prog.starts.at(-1)! + runMinutes(r.prog.runs.at(-1)!, p) : 0)));
  const margin = { top: 8, right: 16, bottom: 34, left: 104 };
  const H = 34;
  return (
    <Frame height={rows.length * (H + 14) + margin.top + margin.bottom} margin={margin} label="Session timeline"
      overlay={(inner) => {
        if (!hover) return null;
        const x = scaleLinear().domain([0, end]).range([0, inner.width]);
        return <Tooltip x={margin.left + x((hover.s.t0 + hover.s.t1) / 2)} y={margin.top + hover.row * (H + 14)}
                        title={`Minute ${hover.s.t0.toFixed(0)}–${hover.s.t1.toFixed(0)}`}
                        rows={[{ value: hover.s.label, name: "", color: compoundColor(hover.s.compound) }]} />;
      }}>
      {({ width, height }) => {
        const x = scaleLinear().domain([0, end]).range([0, width]);
        return (
          <g>
            {x.ticks(6).map((t) => (
              <g key={t} transform={`translate(${x(t)},0)`}>
                <line y1={0} y2={height} style={{ stroke: "var(--grid)" }} shapeRendering="crispEdges" />
                <text y={height + 18} textAnchor="middle" className="tnum" style={{ fontSize: 11.5, fill: "var(--ink-3)" }}>{t}</text>
              </g>
            ))}
            <line x1={x(session)} x2={x(session)} y1={-4} y2={height} style={{ stroke: "var(--ink-2)" }} strokeWidth={1.5} />
            {rows.map((r, ri) => {
              const y = ri * (H + 14);
              return (
                <g key={r.name}>
                  <text x={-12} y={y + H / 2} dy="0.32em" textAnchor="end" style={{ fontSize: 12.5, fontWeight: 560, fill: "var(--ink)" }}>{r.name}</text>
                  {segments(r.prog, p).map((s, i) => {
                    const w = Math.max(x(s.t1) - x(s.t0) - 2, 1);
                    return (
                      <g key={i} onPointerEnter={() => setHover({ s, row: ri })} onPointerLeave={() => setHover(null)}>
                        <rect x={x(s.t0) + 1} y={y} width={w} height={H} rx={5}
                              style={{ fill: `color-mix(in srgb, ${compoundColor(s.compound)} 22%, var(--surface))`, stroke: compoundColor(s.compound) }} strokeWidth={1.5} />
                        {w > 26 && (
                          <text x={x(s.t0) + 1 + w / 2} y={y + H / 2} dy="0.34em" textAnchor="middle"
                                style={{ fontSize: 12, fontWeight: 700, fill: "var(--ink)" }}>{s.compound[0]}</text>
                        )}
                      </g>
                    );
                  })}
                </g>
              );
            })}
          </g>
        );
      }}
    </Frame>
  );
}

const SD_KEYS = [["deg_SOFT", "Soft"], ["deg_MEDIUM", "Medium"], ["deg_HARD", "Hard"], ["SOFT−HARD", "Soft − hard"]] as const;

function Uncertainty({ sds }: { sds: { name: string; color: string; sd: Record<string, number> }[] }) {
  const [hover, setHover] = useState<string | null>(null);
  const margin = { top: 10, right: 12, bottom: 30, left: 52 };
  return (
    <Frame height={280} margin={margin} label="Uncertainty left after the session"
      overlay={(inner) => {
        if (!hover) return null;
        const x0 = scaleBand<string>().domain(SD_KEYS.map((k) => k[1])).range([0, inner.width]).padding(0.28);
        const key = SD_KEYS.find((k) => k[1] === hover)!;
        return <Tooltip x={margin.left + (x0(hover) ?? 0) + x0.bandwidth() / 2} y={margin.top + 6} title={`${hover} degradation`}
                        rows={sds.map((s) => ({ color: s.color, value: `±${fx(s.sd[key[0]], 4)}`, name: s.name }))} />;
      }}>
      {({ width, height }) => {
        const x0 = scaleBand<string>().domain(SD_KEYS.map((k) => k[1])).range([0, width]).padding(0.28);
        const x1 = scaleBand<string>().domain(sds.map((s) => s.name)).range([0, x0.bandwidth()]).padding(0.12);
        const y = scaleLinear().domain([0, Math.max(...sds.flatMap((s) => SD_KEYS.map((k) => s.sd[k[0]])) ) * 1.1]).range([height, 0]).nice();
        const bw = Math.min(x1.bandwidth(), 24);
        return (
          <g>
            <YAxis y={y} width={width} ticks={4} format={(v) => v.toFixed(2)} label="sd, s/lap" />
            <line x1={0} x2={width} y1={height} y2={height} style={{ stroke: "var(--axis)" }} />
            {SD_KEYS.map(([k, name]) => (
              <g key={k} transform={`translate(${x0(name)},0)`} onPointerEnter={() => setHover(name)} onPointerLeave={() => setHover(null)}>
                <rect x={-6} width={x0.bandwidth() + 12} height={height} fill="transparent" />
                {hover === name && <rect x={-6} width={x0.bandwidth() + 12} height={height} rx={6} style={{ fill: "var(--surface-2)" }} />}
                {sds.map((s) => {
                  const bx = (x1(s.name) ?? 0) + (x1.bandwidth() - bw) / 2;
                  return <path key={s.name} d={barPath(bx, y(s.sd[k]), bw, height - y(s.sd[k]))} style={{ fill: s.color }} />;
                })}
                <text x={x0.bandwidth() / 2} y={height + 18} textAnchor="middle" style={{ fontSize: 12, fill: "var(--ink-2)" }}>{name}</text>
              </g>
            ))}
          </g>
        );
      }}
    </Frame>
  );
}

export default function Planner() {
  const [session, setSession] = useState(60);
  const [sigma, setSigma] = useState(0.3);
  const [bfSd, setBfSd] = useState(0.01);
  const [degSd, setDegSd] = useState(0.04);
  const [target, setTarget] = useState<keyof typeof TARGETS>("degradation per compound");
  const [reserve, setReserve] = useState(true);
  const [swaps, setSwaps] = useState(true);

  const settings = useDeferredValue({ session, sigma, bfSd, degSd, target, reserve, swaps });
  const stale = settings.session !== session || settings.sigma !== sigma || settings.bfSd !== bfSd ||
                settings.degSd !== degSd || settings.target !== target || settings.reserve !== reserve || settings.swaps !== swaps;

  const out = useMemo(() => {
    const p: Prior = { ...DEFAULT_PRIOR, sigma: settings.sigma, beta_f: [0.033, settings.bfSd], deg: [0.05, settings.degSd] };
    const A = TARGETS[settings.target];
    const { steps, start } = plan(p, A, { session: settings.session, allowSwap: settings.swaps, reserveQualiSim: settings.reserve });
    const planned = programmeOf(steps, start);
    const std = standardProgramme(p);
    const stdEnd = std.starts.at(-1)! + runMinutes(std.runs.at(-1)!, p);
    return {
      p, steps, planned, std, stdFits: stdEnd <= settings.session,
      gPlan: eig(planned, p, A), gStd: eig(std, p, A),
      sd: { before: posteriorSd(EMPTY, p), std: posteriorSd(std, p), plan: posteriorSd(planned, p) },
    };
  }, [settings]);

  const gain = out.gStd > 0 ? out.gPlan / out.gStd - 1 : null;

  return (
    <div className="page">
      <PageHead eyebrow="Decide · Bayesian experimental design" title="Plan the FP2 runs that teach the most">
        Every model estimates from whatever running happened. This turns it around: score every candidate run by its
        <strong> exact expected information gain</strong> about what a race strategy needs, then fill the session with
        the best combination. It recomputes live as you change the assumptions.
      </PageHead>

      <div className="grid g-side">
        <Card title="Assumptions">
          <Slider label="Session length" value={session} min={30} max={90} step={5} onChange={setSession} format={(v) => `${v} min`} />
          <Slider label="Lap-time noise" value={sigma} min={0.1} max={0.6} step={0.05} onChange={setSigma} format={(v) => `${fx(v, 2)} s`}
                  help="Noisier laps favour longer runs over clever ones." />
          <Slider label="Fuel-effect uncertainty" value={bfSd} min={0.002} max={0.02} step={0.001} onChange={setBfSd} format={(v) => `±${fx(v)} s/kg`}
                  help="Prior sd around the paddock's 0.033 s/kg." />
          <Slider label="Degradation uncertainty" value={degSd} min={0.01} max={0.08} step={0.005} onChange={setDegSd} format={(v) => `±${fx(v)} s/lap`} />
          <div className="control" style={{ marginTop: 20 }}>
            <span className="control-label">Optimise for</span>
            <Segmented label="Optimisation target" value={target} onChange={setTarget}
                       options={Object.keys(TARGETS).map((k) => ({ value: k, label: TARGET_LABEL[k] }))} />
          </div>
          <div style={{ marginTop: 20, display: "flex", flexDirection: "column", gap: 16 }}>
            <Switch label="Keep a qualifying simulation" checked={reserve} onChange={setReserve}
                    help="Teams need it regardless. Turning it off flatters the planner." />
            <Switch label="Allow mid-run tyre changes" checked={swaps} onChange={setSwaps}
                    help="New tyres without refuelling: a pit stop, run on purpose in practice." />
          </div>
        </Card>

        <div style={{ display: "flex", flexDirection: "column", gap: 16, opacity: stale ? 0.55 : 1, transition: "opacity 120ms ease" }}>
          <div className="grid g3">
            <div className="card tile">
              <div className="tile-label">Planned programme</div>
              <div className="tile-value">{fx(out.gPlan, 2)}<span className="tile-unit">bits</span></div>
              <div className="tile-foot">information about the target</div>
            </div>
            <div className="card tile">
              <div className="tile-label">Typical FP2</div>
              <div className="tile-value">{out.stdFits ? fx(out.gStd, 2) : "—"}<span className="tile-unit">{out.stdFits ? "bits" : ""}</span></div>
              <div className="tile-foot">{out.stdFits ? "quali sim + medium and hard long runs" : "doesn't fit this session"}</div>
            </div>
            <div className="card tile">
              <div className="tile-label">Extra information</div>
              <div className="tile-value" style={{ color: gain != null && gain > 0.005 ? "var(--accent-ink)" : "var(--ink)" }}>
                {out.stdFits && gain != null ? `${gain >= 0 ? "+" : "−"}${pct(Math.abs(gain))}` : "—"}
              </div>
              <div className="tile-foot">{swaps ? "with tyre changes allowed" : "without tyre changes"}</div>
            </div>
          </div>

          <Card title="Session timeline" sub="Session minutes · hover a stint for details · the vertical line is the chequered flag">
            <Timeline rows={[{ name: "Typical FP2", prog: out.std }, { name: "Planned", prog: out.planned }]} session={session} p={out.p} />
            <div style={{ marginTop: 6 }}>
              <span className="legend">
                {["SOFT", "MEDIUM", "HARD"].map((c) => (
                  <span key={c} className="legend-item"><Tyre compound={c} size={16} />{c[0] + c.slice(1).toLowerCase()}</span>
                ))}
              </span>
            </div>
          </Card>

          <Card title="Uncertainty left after the session" sub="Posterior sd of each degradation rate · lower is better">
          <Uncertainty sds={[
            { name: "Before FP2", color: "var(--muted-series)", sd: out.sd.before },
            { name: "Typical FP2", color: "var(--s2)", sd: out.sd.std },
            { name: "Planned", color: "var(--s1)", sd: out.sd.plan },
          ]} />
          <div style={{ marginTop: 8 }}>
            <Legend items={[{ label: "Before FP2", color: "var(--muted-series)", kind: "rect" }, { label: "Typical FP2", color: "var(--s2)", kind: "rect" }, { label: "Planned", color: "var(--s1)", kind: "rect" }]} />
          </div>
        </Card>
        </div>
      </div>

      <div>
        <Card title="The plan, run by run" sub={reserve ? "After the fixed 3-lap soft qualifying simulation at minute 5" : undefined}>
          {out.steps.length ? (
            <table className="table">
              <thead><tr><th>Start</th><th>Run</th><th className="num">Adds</th><th className="num">Total</th></tr></thead>
              <tbody>
                {out.steps.map((s, i) => (
                  <tr key={i}>
                    <td className="tnum muted">{s.start.toFixed(1)} min</td>
                    <td>
                      <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                        <Tyre compound={s.run.compound} size={18} />
                        {s.run.swap_to && <><span className="muted">→</span><Tyre compound={s.run.swap_to} size={18} /></>}
                        <span>{s.run.laps} laps{s.run.swap_to ? ", change at half" : ""}</span>
                        <span className="muted">· {s.run.start_fuel_kg.toFixed(0)} kg</span>
                      </span>
                    </td>
                    <td className="num">+{fx(s.gain, 2)}</td>
                    <td className="num muted">{fx(s.total, 2)} bits</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <div className="empty">No run fits in this session.</div>}
        </Card>

      </div>

      <div className="callout">
        <span aria-hidden>ⓘ</span>
        <span>
          <strong>Why the planner likes tyre changes.</strong> Within one run, fuel falls, tyre age rises and the track
          improves in lock step, so a run only reveals their combined effect. A tyre change <strong>without refuelling</strong>{" "}
          resets tyre age while fuel carries on: the one move that separates wear from fuel. With tyre changes off and
          the quali sim kept, the planner arrives at the teams' standard programme by itself. All of the gain comes from
          that one move.
        </span>
      </div>
      <p className="note">
        Exact for a linear-Gaussian lap model; numbers match the Python planner to 7 decimal places
        (<code>web/src/lib/planner/eig.test.ts</code>). v1 assumes linear degradation (no cliff), known fuel loads,
        one car, no tyre-allocation limit, untimed out/in laps and 4 minutes between runs.
      </p>
    </div>
  );
}
