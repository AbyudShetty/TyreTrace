import { useEffect, useState } from "react";

export type Theme = "dark" | "light";
const KEY = "tyretrace-theme";

function initial(): Theme {
  // ?theme=light|dark in the URL wins -- for sharing a link in a set theme
  const q = new URLSearchParams(window.location.search).get("theme");
  if (q === "dark" || q === "light") return q;
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === "dark" || saved === "light") return saved;
  } catch {
    /* private mode / blocked storage: fall through */
  }
  return "dark";
}

export function useTheme(): [Theme, () => void] {
  const [theme, set] = useState<Theme>(initial);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem(KEY, theme); } catch { /* ignore */ }
  }, [theme]);
  return [theme, () => set((t) => (t === "dark" ? "light" : "dark"))];
}

/** Read a CSS custom property -- charts colour by role, not by raw hex, so
 *  a theme switch repaints them. Call inside render; App re-renders on toggle. */
export const cssVar = (name: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();
