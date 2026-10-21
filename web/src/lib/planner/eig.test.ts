/**
 * Parity with the Python planner. fixtures.json is written by
 * `python -m tyretrace export` from tyretrace/planner/eig.py, which is itself
 * checked against brute-force simulation. If these drift, the web page is
 * showing numbers the science does not back.
 */
import { describe, expect, it } from "vitest";
import fixtures from "./fixtures.json";
import {
  addRun, EMPTY, eig, plan, posteriorSd, programmeOf, standardProgramme,
  TARGETS, type Prior, type Run,
} from "./eig";

const priors = fixtures.priors as unknown as Record<string, Prior>;

describe("planner parity with Python", () => {
  for (const c of fixtures.cases) {
    it(`${c.prior} / ${c.programme}`, () => {
      const p = priors[c.prior];
      const g = (c.runs as Run[]).reduce((acc, r) => addRun(acc, r, p), EMPTY);
      g.starts.forEach((s, i) => expect(s).toBeCloseTo(c.starts[i], 9));
      for (const [t, v] of Object.entries(c.eig)) {
        expect(eig(g, p, TARGETS[t])).toBeCloseTo(v as number, 7);
      }
      const sd = posteriorSd(g, p);
      for (const [k, v] of Object.entries(c.posterior_sd)) {
        expect(sd[k]).toBeCloseTo(v as number, 9);
      }
    });
  }

  it("standard programme", () => {
    const p = priors["default"];
    expect(eig(standardProgramme(p), p, TARGETS["degradation per compound"]))
      .toBeCloseTo(fixtures.standard_bits, 7);
  });

  for (const pl of fixtures.plans) {
    it(`plan reserve=${pl.reserve} swaps=${pl.swaps} matches Python`, () => {
      const p = priors["default"];
      const A = TARGETS["degradation per compound"];
      const { steps, start } = plan(p, A, { reserveQualiSim: pl.reserve, allowSwap: pl.swaps });
      // the optimum's VALUE must match; with exact ties the chosen runs may differ
      expect(eig(programmeOf(steps, start), p, A)).toBeCloseTo(pl.bits, 6);
    });
  }
});
