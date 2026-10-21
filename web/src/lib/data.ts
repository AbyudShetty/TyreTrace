import { useEffect, useState } from "react";

/** Shapes written by tyretrace/export.py. Kept in one place so a schema
 *  change on the Python side breaks the type-check, not the page. */

export type Gate = "PASS" | "FAIL" | "HOLD" | "SHIP" | null;

export interface GateRow {
  task: string;
  gate: Gate;
  expect: string;
  why: string;
  summary: string;
  as_expected: boolean | null;
  recorded_at: string | null;
}

export interface Summary {
  generated_at: string;
  data: { weekends: number; first: string; last: string; race_laps: number; quali_laps: number; practice_laps: number };
  gates: GateRow[];
  headline: {
    tau: { median: number; frac_negative: number; z: number; n: number; total_s: number };
    fuel_pace: { mean: number; q05: number; q95: number; physics: [number, number] };
    tau_scale: { mean: number; q05: number; q95: number; p_lt_1: number };
    wake: number;
  };
  ladder: { rung: string; crps: number; mae: number; cov90: number; rhat: number | null }[];
  twin: { rung: string; crps: number; cov90: number }[];
  baselines: { name: string; key: string; crps: number; mae: number; cov90: number }[];
  backtest: {
    gate: Gate;
    weekends: number;
    methods: { method: string; crps: number; crps_after: number; bias_after: number; cov90: number; stint_mae: number; n: number; n_after: number }[];
    slope_mae: Record<string, number>;
  };
}

const cache = new Map<string, Promise<unknown>>();

function load<T>(name: string): Promise<T> {
  if (!cache.has(name)) {
    cache.set(name, fetch(`${import.meta.env.BASE_URL}data/${name}`).then((r) => {
      if (!r.ok) throw new Error(`${name}: ${r.status}`);
      return r.json();
    }));
  }
  return cache.get(name) as Promise<T>;
}

export type Loaded<T> = { state: "loading" } | { state: "error"; error: string } | { state: "ready"; data: T };

export function useData<T>(name: string): Loaded<T> {
  const [s, set] = useState<Loaded<T>>({ state: "loading" });
  useEffect(() => {
    let live = true;
    load<T>(name).then(
      (data) => live && set({ state: "ready", data }),
      (e: Error) => live && set({ state: "error", error: e.message }),
    );
    return () => { live = false; };
  }, [name]);
  return s;
}
