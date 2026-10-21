import { ModelRows, type ModelRow } from "../charts/ModelRows";
import { Card, GateBadge, Loading, Missing, PageHead, Stat } from "../components/ui";
import { useData, type Summary } from "../lib/data";
import { fx, int, pct, signed, weekend } from "../lib/format";

/** The Python summaries use ASCII "--"; typeset it. */
const dash = (t: string) => t.replace(/\s?--\s?/g, " — ");

const RUNG_NOTE: Record<string, string> = {
  L2: "fuel / track split", L3: "+ dirty air", L4: "+ stint offsets, heavy tails", L5: "+ σ(age) · rejected",
};

function Chain() {
  const steps = [
    { n: "1", title: "Qualifying", gives: "Track evolution", body: "Every car on minimum fuel, so pace against the clock is the track rubbering in." },
    { n: "2", title: "Races", gives: "Fuel vs degradation", body: "A pit stop resets tyre age while fuel carries on. That jump separates the two." },
    { n: "3", title: "Practice", gives: "Consumes both", body: "No longer has to identify anything; it inherits what qualifying and races measured." },
  ];
  return (
    <div className="grid g3" style={{ gap: 12 }}>
      {steps.map((s, i) => (
        <div key={s.n} className="card" style={{ position: "relative", padding: "18px 20px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
            <span style={{ width: 22, height: 22, borderRadius: 6, display: "grid", placeItems: "center",
                           background: "var(--accent-soft)", color: "var(--accent-ink)", fontSize: 12, fontWeight: 700 }}>{s.n}</span>
            <span style={{ fontWeight: 620 }}>{s.title}</span>
            <span style={{ marginLeft: "auto", fontSize: 12, color: "var(--ink-3)" }}>→ {s.gives}</span>
          </div>
          <p className="note" style={{ fontSize: 13 }}>{s.body}</p>
          {i < 2 && (
            <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden
                 style={{ position: "absolute", right: -13, top: "50%", transform: "translateY(-50%)", zIndex: 1 }}>
              <path d="M2 3l5 4-5 4" fill="none" style={{ stroke: "var(--ink-3)" }} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          )}
        </div>
      ))}
    </div>
  );
}

export default function Overview() {
  const s = useData<Summary>("summary.json");
  if (s.state === "loading") return <div className="page"><Loading /></div>;
  if (s.state === "error") return <div className="page"><Missing what="the overview" cmd="python -m tyretrace run all" /></div>;
  const d = s.data;
  const h = d.headline;

  const best = d.ladder.reduce((a, b) => (b.crps < a.crps ? b : a));
  const rows: ModelRow[] = [
    ...d.ladder.map((r) => ({
      id: r.rung, label: r.rung, sub: RUNG_NOTE[r.rung], crps: r.crps, cov90: r.cov90, mae: r.mae, rhat: r.rhat,
      color: r.rung === best.rung ? "var(--s1)" : r.rung === "L5" ? "var(--muted-series)" : "color-mix(in srgb, var(--s1) 55%, var(--surface))",
      group: "bayes",
    })),
    ...d.baselines.map((b) => ({
      id: b.key, label: b.name, sub: "baseline", crps: b.crps, cov90: b.cov90, mae: b.mae,
      color: "var(--muted-series)", group: "base",
    })),
  ];

  const bt = Object.fromEntries(d.backtest.methods.map((m) => [m.method, m]));
  const surprises = d.gates.filter((g) => g.gate && !g.as_expected);

  return (
    <div className="page">
      <PageHead eyebrow="TrackShift 2026 · Tyre degradation intelligence" title="Recovering true tyre wear from confounded lap times">
        Inside a stint, fuel burns off and tyres wear out <strong>in lock step</strong>, so a lap-time slope only
        ever shows their sum. Every published model splits it by importing a fuel coefficient nobody verifies.
        Tyre Trace treats it as an identification problem: find the data that actually separates the two, and say
        how sure it is.
      </PageHead>

      <div className="grid g4">
        <Stat label="Track evolution" value={signed(h.tau.median, 4)} unit="s / min"
              foot={`Negative in ${pct(h.tau.frac_negative)} of ${h.tau.n} qualifying sessions · z ≈ ${fx(h.tau.z, 1)}`} />
        <Stat label="Fuel effect, identified" value={fx(h.fuel_pace.mean, 4)} unit="s / lap"
              foot={<>90%: {fx(h.fuel_pace.q05, 3)}–{fx(h.fuel_pace.q95, 3)} · physics says {h.fuel_pace.physics[0]}–{h.fuel_pace.physics[1]} <span style={{ color: "var(--ink-2)" }}>(open question)</span></>} />
        <Stat label="Race vs qualifying evolution" value={fx(h.tau_scale.mean, 2)} unit="×"
              foot={`P(less than qualifying) = ${pct(h.tau_scale.p_lt_1, 1)}`} />
        <Stat label="Dirty air, full wake" value={fx(h.wake, 2)} unit="s / lap"
              foot="From gap-to-car-ahead in timing data alone" />
      </div>

      <section style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div>
          <h2 className="section-title">The identification chain</h2>
          <p className="section-sub">Each session type measures the one thing it can measure cleanly, and hands it on.</p>
        </div>
        <Chain />
      </section>

      <Card title="Every model, scored on the same held-out laps"
            sub={`Last 30% of each race, ${d.data.weekends} weekends · a model ships only if it beats the one below AND its 90% interval holds 85–95% of laps`}>
        <ModelRows rows={rows} band={[0.85, 0.95]} />
        <p className="note" style={{ marginTop: 12 }}>
          <strong>{best.rung} is the most accurate model and beats every baseline, but no model ships:</strong> all
          of them are overconfident, with too few laps inside their 90% intervals. Calibration is the open
          problem. Hover a row for the convergence check (r̂).
        </p>
      </Card>

      <div className="grid g2" style={{ alignItems: "start" }}>
        <Card title="Practice → race: the decisive test"
              sub={`Each race forecast from its own weekend · ${d.backtest.weekends} weekends · laps after a pit stop`}
              right={<GateBadge gate={d.backtest.gate} />}>
          <table className="table">
            <thead><tr><th>Method</th><th className="num">CRPS</th><th className="num">Bias</th></tr></thead>
            <tbody>
              {(["conventional", "identified", "no-track", "naive"] as const).map((m) => bt[m] && (
                <tr key={m}>
                  <td>{m === "identified" ? <strong>Identified (ours)</strong> : m === "no-track" ? "Ours, no track term" : m[0].toUpperCase() + m.slice(1)}</td>
                  <td className="num">{fx(bt[m].crps_after)}</td>
                  <td className="num ink2">{signed(bt[m].bias_after, 2)} s</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="note" style={{ marginTop: 12 }}>
            Ours beats naive by {pct(1 - bt.identified.crps_after / bt.naive.crps_after)} but trails a fixed
            0.030 s/kg by {pct(bt.identified.crps_after / bt.conventional.crps_after - 1, 1)}. The fuel effect
            learned from races is <strong>too small</strong>, which is the first out-of-sample evidence on the
            open fuel-pace gap.
          </p>
        </Card>

        <Card title="Gate board" sub={surprises.length ? `${surprises.length} unexpected result(s)` : "Every audit returned the verdict it was expected to"}>
          <table className="table">
            <tbody>
              {d.gates.filter((g) => g.gate).map((g) => (
                <tr key={g.task}>
                  <td style={{ width: 82 }}><GateBadge gate={g.gate} /></td>
                  <td>
                    <div style={{ fontSize: 13 }}>{dash(g.summary)}</div>
                    {g.expect !== "PASS" && <div className="note">Expected: {dash(g.why)}</div>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      </div>

      <p className="note">
        Data: {d.data.weekends} race weekends ({weekend(d.data.first)} to {weekend(d.data.last)}),{" "}
        {int(d.data.race_laps)} race, {int(d.data.quali_laps)} qualifying and {int(d.data.practice_laps)} practice
        laps from public FastF1 timing. No team-confidential inputs. Exported {new Date(d.generated_at).toLocaleString()}.
      </p>
    </div>
  );
}
