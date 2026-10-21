import { useMemo, useState } from "react";
import { LineChart, type Series } from "../charts/LineChart";
import { Card, Legend, PageHead, Slider } from "../components/ui";
import { fx, signed } from "../lib/format";

/** Seeded normal draws, so the "observed" laps are stable while you drag. */
function rng(seed: number) {
  let a = seed >>> 0;
  const u = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return () => Math.sqrt(-2 * Math.log(u() + 1e-12)) * Math.cos(2 * Math.PI * u());
}

/** Least squares for y ~ 1 + lap + age, with standard errors. */
function fit(lap: number[], age: number[], y: number[]) {
  const X = lap.map((l, i) => [1, l, age[i]]);
  const XtX = [0, 1, 2].map((a) => [0, 1, 2].map((b) => X.reduce((s, r) => s + r[a] * r[b], 0)));
  const Xty = [0, 1, 2].map((a) => X.reduce((s, r, i) => s + r[a] * y[i], 0));
  const inv = invert3(XtX);
  const beta = inv.map((row) => row.reduce((s, v, j) => s + v * Xty[j], 0));
  const resid = y.map((v, i) => v - X[i].reduce((s, x, j) => s + x * beta[j], 0));
  const s2 = resid.reduce((s, r) => s + r * r, 0) / Math.max(y.length - 3, 1);
  return { beta, se: inv.map((row, i) => Math.sqrt(row[i] * s2)) };
}

function invert3(m: number[][]) {
  const [[a, b, c], [d, e, f], [g, h, i]] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [
    [A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
    [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
    [C / det, -(a * h - b * g) / det, (a * e - b * d) / det],
  ];
}

export default function Problem() {
  // part 1
  const [net, setNet] = useState(0.01);
  const [assumed, setAssumed] = useState(0.055);

  const stint = useMemo(() => {
    const k = Array.from({ length: 21 }, (_, i) => i);
    const s: Series[] = [];
    // your assumption, full strength; two equally consistent alternatives, faint
    for (const [f, o] of [[assumed - 0.025, 0.35], [assumed + 0.025, 0.35], [assumed, 1]] as const) {
      s.push({ name: `tyre wear ${signed(net + f)}`, color: "var(--soft)", opacity: o, points: k.map((i) => [i, (net + f) * i]), hideInTip: o < 1 });
      s.push({ name: `fuel burn ${signed(-f)}`, color: "var(--s1)", opacity: o, points: k.map((i) => [i, -f * i]), hideInTip: o < 1 });
    }
    s.push({ name: "what you observe", color: "var(--ink)", width: 3, points: k.map((i) => [i, net * i]) });
    return s;
  }, [net, assumed]);

  // part 2
  const [deg, setDeg] = useState(0.06);
  const [pit, setPit] = useState(22);
  const [noise, setNoise] = useState(0.15);
  const [seed, setSeed] = useState(7);
  const fuel = 0.058;

  const race = useMemo(() => {
    const laps = Array.from({ length: 50 }, (_, i) => i + 1);
    const age = laps.map((l) => (l <= pit ? l : l - pit));
    const net1 = deg - fuel;
    const rivalDeg = deg + 0.04, rivalFuel = rivalDeg - net1;
    const pace = (d: number, f: number) => laps.map((l, i) => d * age[i] - f * l);
    const z = rng(seed);
    const obs = pace(deg, fuel).map((v) => v + noise * z());
    const f = fit(laps, age, obs);
    return {
      series: [
        { name: `rival story · deg ${fx(rivalDeg)}`, color: "var(--s7)", dash: "6 5", points: laps.map((l, i) => [l, pace(rivalDeg, rivalFuel)[i]]) },
        { name: `truth · deg ${fx(deg)}`, color: "var(--soft)", points: laps.map((l, i) => [l, pace(deg, fuel)[i]]) },
        { name: "observed lap", color: "var(--ink)", kind: "dots", points: laps.map((l, i) => [l, obs[i]]) },
      ] as Series[],
      deg: f.beta[2], degSe: f.se[2], fuel: -f.beta[1], fuelSe: f.se[1],
    };
  }, [deg, pit, noise, seed]);

  return (
    <div className="page">
      <PageHead eyebrow="Why this is hard" title="One stint can't tell fuel from tyres">
        As fuel burns off, the car gets <strong>lighter and faster</strong>. As tyres wear, it gets <strong>slower</strong>.
        Both move by the same amount every lap, so a stint's lap times only show the difference between them.
      </PageHead>

      <Card title="1 · Infinitely many explanations for the same stint"
            sub="Three different splits of fuel and tyre wear, each producing the identical observed line">
        <div className="grid g-side">
          <div>
            <Slider label="Observed slope in the stint" value={net} min={-0.05} max={0.08} step={0.005}
                    onChange={setNet} format={(v) => `${signed(v)} s/lap`}
                    help="What a regression of lap time on lap-in-stint returns." />
            <Slider label="Your assumed fuel effect" value={assumed} min={0} max={0.1} step={0.005}
                    onChange={setAssumed} format={(v) => `${fx(v)} s/lap`}
                    help="The coefficient everyone imports and nobody verifies." />
            <div className="card" style={{ marginTop: 20, background: "var(--surface-2)", boxShadow: "none" }}>
              <div className="tile-label">You would report tyre degradation of</div>
              <div className="tile-value" style={{ fontSize: 26, marginTop: 4 }}>{signed(net + assumed)}<span className="tile-unit">s / lap</span></div>
              <div className="tile-foot" style={{ marginTop: 6 }}>Move the fuel slider: the data hasn't changed, but the answer moves one-for-one with your assumption.</div>
            </div>
          </div>
          <div>
            <LineChart series={stint} height={340} xLabel="lap in stint" yLabel="Δ lap time (s)"
                       label="Three explanations of one stint" tipValue={(v) => signed(v, 2)} xFormat={(v) => `Lap ${v}`} />
            <div style={{ marginTop: 10 }}>
              <Legend items={[{ label: "tyre wear (slower)", color: "var(--soft)" }, { label: "fuel burn (faster)", color: "var(--s1)" }, { label: "what you observe", color: "var(--ink)" }]} />
            </div>
          </div>
        </div>
      </Card>

      <Card title="2 · The pit stop breaks the tie"
            sub="New tyres, same fuel. Two stories that matched perfectly now predict different jumps.">
        <div className="grid g-side">
          <div>
            <Slider label="True degradation" value={deg} min={0.02} max={0.12} step={0.005} onChange={setDeg} format={(v) => `${fx(v)} s/lap`} />
            <Slider label="Pit stop on lap" value={pit} min={10} max={40} step={1} onChange={setPit} format={String} />
            <Slider label="Lap-time noise" value={noise} min={0} max={0.4} step={0.05} onChange={setNoise} format={(v) => `${fx(v, 2)} s`} />
            <Slider label="Random seed" value={seed} min={1} max={50} step={1} onChange={setSeed} format={String} />
            <div className="grid g2" style={{ marginTop: 20, gap: 10 }}>
              <div className="card" style={{ background: "var(--surface-2)", boxShadow: "none", padding: 14 }}>
                <div className="tile-label">Recovered degradation</div>
                <div style={{ fontSize: 19, fontWeight: 640, marginTop: 2 }} className="tnum">{fx(race.deg)}</div>
                <div className="tile-foot">± {fx(2 * race.degSe)} · truth {fx(deg)}</div>
              </div>
              <div className="card" style={{ background: "var(--surface-2)", boxShadow: "none", padding: 14 }}>
                <div className="tile-label">Recovered fuel effect</div>
                <div style={{ fontSize: 19, fontWeight: 640, marginTop: 2 }} className="tnum">{fx(race.fuel)}</div>
                <div className="tile-foot">± {fx(2 * race.fuelSe)} · truth {fx(fuel)}</div>
              </div>
            </div>
          </div>
          <div>
            <LineChart series={race.series} height={360} xLabel="race lap" yLabel="Δ lap time (s)"
                       label="A race with one pit stop" tipValue={(v) => signed(v, 2)} xFormat={(v) => `Lap ${v}`}
                       annotate={({ x, inner }) => (
                         <g>
                           <line x1={x(pit + 0.5)} x2={x(pit + 0.5)} y1={0} y2={inner.height} style={{ stroke: "var(--ink-3)" }} strokeDasharray="2 4" />
                           <text x={x(pit + 0.5) + 8} y={14} style={{ fontSize: 11.5, fill: "var(--ink-2)" }}>pit stop · tyre age → 0, fuel unchanged</text>
                         </g>
                       )} />
            <div style={{ marginTop: 10 }}>
              <Legend items={[{ label: "truth", color: "var(--soft)" }, { label: "rival story, same first stint", color: "var(--s7)" }, { label: "observed laps", color: "var(--ink)", kind: "rect" }]} />
            </div>
          </div>
        </div>
      </Card>

      <div className="callout">
        <span aria-hidden>ⓘ</span>
        <span>
          <strong>The catch.</strong> A race lap also gets faster as the track rubbers in, and that moves with the
          same lap index as fuel burn. Tyre Trace measures track evolution separately, in <strong>qualifying</strong>,
          where every car runs minimum fuel.
        </span>
      </div>
    </div>
  );
}
