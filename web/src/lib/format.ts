export const fx = (v: number | null | undefined, d = 3) =>
  v == null || Number.isNaN(v) ? "—" : v.toFixed(d);

export const signed = (v: number | null | undefined, d = 3) =>
  v == null || Number.isNaN(v) ? "—" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`;

export const pct = (v: number | null | undefined, d = 0) =>
  v == null || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(d)}%`;

export const int = (v: number | null | undefined) =>
  v == null ? "—" : Math.round(v).toLocaleString("en-US");

/** "2024_06" -> "2024 · R06" */
export const weekend = (k: string) => {
  const [y, r] = k.split("_");
  return `${y} · R${r}`;
};

/** Typographic minus everywhere a number can be negative. */
export const minus = (s: string) => s.replace(/^-/, "−");
