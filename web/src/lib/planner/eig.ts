/**
 * FP2 run planner -- TypeScript port of tyretrace/planner/eig.py.
 *
 * Same model, same candidates, same search (two greedy passes + beam + local
 * search), same labels. One implementation difference, for speed: each run's
 * free intercept is marginalised analytically (a Schur complement), so the
 * posterior precision of the 5 global parameters is
 *
 *   P0 + sum_r [ G_r'G_r - (G_r'1)(1'G_r) / (n_r + sigma^2 / v) ] / sigma^2
 *
 * with G_r the run's global design rows and v the run-intercept prior
 * variance. That is exactly the Python posterior's global block, computed on
 * 5x5 matrices instead of (5 + runs)^2. eig.test.ts checks every number
 * against fixtures exported from the Python planner, which is itself checked
 * against brute-force simulation.
 */

export const COMPOUNDS = ["SOFT", "MEDIUM", "HARD"] as const;
export type Compound = (typeof COMPOUNDS)[number];
export const PARAMS = ["beta_f", "deg_SOFT", "deg_MEDIUM", "deg_HARD", "tau"] as const;
const P = PARAMS.length;

export const OUT_IN_LAPS = 2;
export const SWAP_LAPS = 2;
export const TURNAROUND_MIN = 4.0;

export interface Prior {
  beta_f: [number, number];
  deg: [number, number];
  tau: [number, number];
  run_sd: number;
  sigma: number;
  kappa: number;
  lap_s: number;
}

export const DEFAULT_PRIOR: Prior = {
  beta_f: [0.033, 0.01], deg: [0.05, 0.04], tau: [-0.0165, 0.003],
  run_sd: 3.0, sigma: 0.3, kappa: 1.8, lap_s: 90.0,
};

export interface Run {
  compound: Compound;
  laps: number;
  start_fuel_kg: number;
  swap_to: Compound | null;
}

export const runMinutes = (r: Run, p: Prior) =>
  ((r.laps + OUT_IN_LAPS + (r.swap_to ? SWAP_LAPS : 0)) * p.lap_s) / 60;

/** Must match Run.label() in Python exactly -- labels key the beam dedupe. */
export const runLabel = (r: Run) =>
  `${r.laps} laps ${r.compound} (${r.start_fuel_kg.toFixed(0)} kg)` +
  (r.swap_to ? ` → ${r.swap_to} at half` : "");

export interface Programme { runs: Run[]; starts: number[] }

export const EMPTY: Programme = { runs: [], starts: [] };

export function progEnd(g: Programme, p: Prior): number {
  if (!g.runs.length) return 0;
  return g.starts[g.starts.length - 1] + runMinutes(g.runs[g.runs.length - 1], p);
}

export function addRun(g: Programme, r: Run, p: Prior, firstStart = 5.0): Programme {
  const start = g.runs.length ? progEnd(g, p) + TURNAROUND_MIN : firstStart;
  return { runs: [...g.runs, r], starts: [...g.starts, start] };
}

// ---------------------------------------------------------------------------
// small dense linear algebra
// ---------------------------------------------------------------------------

type M = number[][];
const zeros = (n: number, m = n): M => Array.from({ length: n }, () => new Array(m).fill(0));

/** Inverse and log|det| by Gauss-Jordan with partial pivoting. */
function invLogdet(a: M): { inv: M; logdet: number } {
  const n = a.length;
  const w = a.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  let logdet = 0;
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(w[r][c]) > Math.abs(w[piv][c])) piv = r;
    if (piv !== c) [w[piv], w[c]] = [w[c], w[piv]];
    const d = w[c][c];
    logdet += Math.log(Math.abs(d));
    for (let j = 0; j < 2 * n; j++) w[c][j] /= d;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = w[r][c];
      if (f !== 0) for (let j = 0; j < 2 * n; j++) w[r][j] -= f * w[c][j];
    }
  }
  return { inv: w.map((row) => row.slice(n)), logdet };
}

/** A S A' for A (k x P), S (P x P). */
function sandwich(A: M, S: M): M {
  const k = A.length;
  const AS = A.map((row) => S[0].map((_, j) => row.reduce((s, v, i) => s + v * S[i][j], 0)));
  const out = zeros(k);
  for (let i = 0; i < k; i++) for (let j = 0; j < k; j++)
    out[i][j] = AS[i].reduce((s, v, t) => s + v * A[j][t], 0);
  return out;
}

// ---------------------------------------------------------------------------
// design
// ---------------------------------------------------------------------------

/** Global-parameter rows for each timed lap -- mirrors run_rows() in Python. */
export function runRows(r: Run, startMin: number, p: Prior): number[][] {
  const rows: number[][] = [];
  const lapMin = p.lap_s / 60;
  const half = r.swap_to ? Math.floor(r.laps / 2) : r.laps;
  let driven = 1;
  let age = 1;
  let comp: Compound = r.compound;
  for (let k = 0; k < r.laps; k++) {
    if (r.swap_to && k === half) {
      driven += SWAP_LAPS;
      age = 1;
      comp = r.swap_to;
    }
    const row = new Array(P).fill(0);
    row[0] = r.start_fuel_kg - p.kappa * driven;
    row[1 + COMPOUNDS.indexOf(comp)] = age;
    row[4] = startMin + driven * lapMin;
    rows.push(row);
    driven += 1;
    age += 1;
  }
  return rows;
}

/** A run's contribution to the global precision, its intercept marginalised. */
function runInfo(r: Run, startMin: number, p: Prior): M {
  const G = runRows(r, startMin, p);
  const n = G.length;
  const s2 = p.sigma * p.sigma;
  const shrink = n + s2 / (p.run_sd * p.run_sd);
  const colsum = new Array(P).fill(0);
  for (const row of G) for (let j = 0; j < P; j++) colsum[j] += row[j];
  const out = zeros(P);
  for (let i = 0; i < P; i++) for (let j = 0; j < P; j++) {
    let gg = 0;
    for (const row of G) gg += row[i] * row[j];
    out[i][j] = (gg - (colsum[i] * colsum[j]) / shrink) / s2;
  }
  return out;
}

export function priorCov(p: Prior): M {
  const sd = [p.beta_f[1], p.deg[1], p.deg[1], p.deg[1], p.tau[1]];
  return sd.map((s, i) => sd.map((_, j) => (i === j ? s * s : 0)));
}

export function posteriorCov(g: Programme, p: Prior): M {
  const prec = priorCov(p).map((row, i) => row.map((v, j) => (i === j ? 1 / v : 0)));
  g.runs.forEach((r, i) => {
    const m = runInfo(r, g.starts[i], p);
    for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) prec[a][b] += m[a][b];
  });
  return invLogdet(prec).inv;
}

// ---------------------------------------------------------------------------
// targets and information gain
// ---------------------------------------------------------------------------

const e = (i: number) => PARAMS.map((_, j) => (i === j ? 1 : 0));

export const TARGETS: Record<string, M> = {
  "degradation per compound": [e(1), e(2), e(3)],
  "SOFT − HARD contrast": [e(1).map((v, j) => v - e(3)[j])],
  "degradation + fuel effect": [e(0), e(1), e(2), e(3)],
};

export function eig(g: Programme, p: Prior, A: M): number {
  const ld0 = invLogdet(sandwich(A, priorCov(p))).logdet;
  const ld = invLogdet(sandwich(A, posteriorCov(g, p))).logdet;
  return (0.5 * (ld0 - ld)) / Math.LN2;
}

export function posteriorSd(g: Programme, p: Prior): Record<string, number> {
  const S = posteriorCov(g, p);
  const out: Record<string, number> = {};
  PARAMS.forEach((name, i) => (out[name] = Math.sqrt(S[i][i])));
  const a = TARGETS["SOFT − HARD contrast"];
  out["SOFT−HARD"] = Math.sqrt(sandwich(a, S)[0][0]);
  return out;
}

// ---------------------------------------------------------------------------
// candidates and the planner
// ---------------------------------------------------------------------------

export function candidates(allowSwap = true, lengths = range(4, 21), fuels = [30, 100]): Run[] {
  const runs: Run[] = [];
  // Python: product(COMPOUNDS, lengths, fuels) -- same nesting order
  for (const c of COMPOUNDS) for (const n of lengths) for (const f of fuels)
    runs.push({ compound: c, laps: n, start_fuel_kg: f, swap_to: null });
  if (allowSwap)
    for (const c of COMPOUNDS) for (const n of lengths) for (const f of fuels) for (const s of COMPOUNDS)
      if (n >= 6) runs.push({ compound: c, laps: n, start_fuel_kg: f, swap_to: s });
  return runs;
}

function range(a: number, b: number) {
  return Array.from({ length: b - a }, (_, i) => a + i);
}

export const QUALI_SIM: Run = { compound: "SOFT", laps: 3, start_fuel_kg: 20, swap_to: null };

const schedule = (runs: Run[], p: Prior, start: Programme) =>
  runs.reduce((g, r) => addRun(g, r, p), start);

function greedy(p: Prior, A: M, session: number, cands: Run[], minGain: number,
                start: Programme, byRate: boolean): Run[] {
  let g = start;
  let total = g.runs.length ? eig(g, p, A) : 0;
  const added: Run[] = [];
  for (;;) {
    let best: [Run, Programme, number] | null = null;
    let bestScore = 0;
    for (const r of cands) {
      const nxt = addRun(g, r, p);
      if (progEnd(nxt, p) > session) continue;
      const gain = eig(nxt, p, A) - total;
      const score = byRate ? gain / runMinutes(r, p) : gain;
      if (gain > minGain && score > bestScore) {
        best = [r, nxt, gain];
        bestScore = score;
      }
    }
    if (!best) return added;
    g = best[1];
    total += best[2];
    added.push(best[0]);
  }
}

function beam(p: Prior, A: M, session: number, cands: Run[], start: Programme, width = 60): Run[] {
  const value = (rs: Run[]) => {
    const g = schedule(rs, p, start);
    return progEnd(g, p) <= session ? eig(g, p, A) : -Infinity;
  };
  let frontier: Run[][] = [[]];
  let best: Run[] = [];
  let bestV = start.runs.length ? value([]) : 0;
  while (frontier.length) {
    const grown = new Map<string, [number, Run[]]>();
    for (const rs of frontier) for (const c of cands) {
      const nxt = [...rs, c];
      const v = value(nxt);
      if (v === -Infinity) continue;
      const key = nxt.map(runLabel).sort().join("|");
      const cur = grown.get(key);
      if (!cur || v > cur[0]) grown.set(key, [v, nxt]);
    }
    const top = [...grown.values()].sort((a, b) => b[0] - a[0]).slice(0, width);
    for (const [v, rs] of top) if (v > bestV) { best = rs; bestV = v; }
    frontier = top.map(([, rs]) => rs);
  }
  return best;
}

function improve(runs: Run[], p: Prior, A: M, session: number, cands: Run[],
                 start: Programme, maxRounds = 25): Run[] {
  const value = (rs: Run[]) => {
    const g = schedule(rs, p, start);
    return progEnd(g, p) <= session ? eig(g, p, A) : -Infinity;
  };
  let best = value(runs);
  for (let round = 0; round < maxRounds; round++) {
    const moves: Run[][] = [];
    runs.forEach((_, i) => cands.forEach((c) => moves.push([...runs.slice(0, i), c, ...runs.slice(i + 1)])));
    runs.forEach((_, i) => moves.push([...runs.slice(0, i), ...runs.slice(i + 1)]));
    cands.forEach((c) => moves.push([...runs, c]));
    let top: Run[] | null = null;
    let topV = -Infinity;
    for (const m of moves) {
      const v = value(m);
      if (v > topV) { top = m; topV = v; }
    }
    if (!top || topV <= best + 1e-9) return runs;
    runs = top;
    best = topV;
  }
  return runs;
}

export interface Step { run: Run; start: number; gain: number; total: number }

export interface PlanOptions {
  session?: number;
  allowSwap?: boolean;
  reserveQualiSim?: boolean;
  minGain?: number;
}

/** Mirrors plan() in Python: greedy x2 + beam, then local search. */
export function plan(p: Prior, A: M, opts: PlanOptions = {}): { steps: Step[]; start: Programme } {
  const session = opts.session ?? 60;
  const cands = candidates(opts.allowSwap ?? true);
  const start = opts.reserveQualiSim ? addRun(EMPTY, QUALI_SIM, p) : EMPTY;
  const minGain = opts.minGain ?? 0.01;
  const value = (rs: Run[]) => eig(schedule(rs, p, start), p, A);
  const seeds = [
    greedy(p, A, session, cands, minGain, start, true),
    greedy(p, A, session, cands, minGain, start, false),
    beam(p, A, session, cands, start),
  ];
  let bestSeed = seeds[0];
  for (const s of seeds.slice(1)) if (value(s) > value(bestSeed)) bestSeed = s;
  const runs = improve(bestSeed, p, A, session, cands, start);

  let g = start;
  let total = g.runs.length ? eig(g, p, A) : 0;
  const steps: Step[] = [];
  for (const r of runs) {
    g = addRun(g, r, p);
    const v = eig(g, p, A);
    steps.push({ run: r, start: g.starts[g.starts.length - 1], gain: v - total, total: v });
    total = v;
  }
  return { steps, start };
}

export function programmeOf(steps: Step[], start: Programme): Programme {
  return { runs: [...start.runs, ...steps.map((s) => s.run)],
           starts: [...start.starts, ...steps.map((s) => s.start)] };
}

export function standardProgramme(p: Prior): Programme {
  return [QUALI_SIM,
          { compound: "MEDIUM", laps: 11, start_fuel_kg: 90, swap_to: null } as Run,
          { compound: "HARD", laps: 11, start_fuel_kg: 90, swap_to: null } as Run,
  ].reduce((g, r) => addRun(g, r, p), EMPTY);
}
