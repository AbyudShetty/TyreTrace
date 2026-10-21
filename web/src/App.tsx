import { lazy, Suspense, type ComponentType } from "react";
import { Loading, PageHead } from "./components/ui";
import { useRoute } from "./lib/router";
import { useTheme } from "./lib/theme";

const Overview = lazy(() => import("./pages/Overview"));
const Problem = lazy(() => import("./pages/Problem"));
const Planner = lazy(() => import("./pages/Planner"));

interface Route { id: string; label: string; page?: ComponentType }

const NAV: { group: string; routes: Route[] }[] = [
  { group: "Tyre Trace", routes: [
    { id: "overview", label: "Overview", page: Overview },
    { id: "problem", label: "The problem", page: Problem },
  ] },
  { group: "Data", routes: [
    { id: "races", label: "Race explorer" },
    { id: "track", label: "Track evolution" },
  ] },
  { group: "Models", routes: [
    { id: "ladder", label: "Model ladder" },
    { id: "predictions", label: "Held-out forecasts" },
    { id: "backtest", label: "Practice → race" },
  ] },
  { group: "Decide", routes: [
    { id: "planner", label: "Run planner", page: Planner },
  ] },
  { group: "Ground truth", routes: [
    { id: "rig", label: "Game rig" },
  ] },
];

const ALL = NAV.flatMap((g) => g.routes);

function Logo() {
  return (
    <svg width="28" height="28" viewBox="0 0 32 32" aria-hidden>
      <circle cx="16" cy="16" r="14.5" style={{ fill: "var(--surface-2)" }} />
      <circle cx="16" cy="16" r="10.5" fill="none" style={{ stroke: "var(--soft)" }} strokeWidth="3"
              strokeDasharray="44 22" strokeLinecap="round" transform="rotate(-60 16 16)" />
      <circle cx="16" cy="16" r="4.5" style={{ fill: "var(--hard)" }} />
    </svg>
  );
}

function Soon({ label }: { label: string }) {
  return (
    <div className="page">
      <PageHead eyebrow="Coming next" title={label}>
        This page is being rebuilt for the new dashboard. Every number it will show already
        exists in the pipeline; it is the presentation that is in progress.
      </PageHead>
    </div>
  );
}

export default function App() {
  const route = useRoute("overview");
  const [theme, toggle] = useTheme();
  const current = ALL.find((r) => r.id === route) ?? ALL[0];
  const Page = current.page;

  return (
    <div className="shell">
      <nav className="sidebar" aria-label="Sections">
        <a className="brand" href="#/overview" style={{ textDecoration: "none", color: "inherit" }}>
          <Logo />
          <span>
            <div className="brand-name">Tyre Trace</div>
            <div className="brand-sub">Causal tyre-degradation intelligence</div>
          </span>
        </a>
        {NAV.map((g) => (
          <div className="nav-group" key={g.group}>
            <div className="nav-label">{g.group}</div>
            {g.routes.map((r) => (
              <a key={r.id} className="nav-item" href={`#/${r.id}`}
                 aria-current={r.id === current.id ? "page" : undefined}>
                <span className="nav-dot" />
                {r.label}
                {!r.page && <span className="nav-soon">soon</span>}
              </a>
            ))}
          </div>
        ))}
        <div className="sidebar-foot">
          <button className="icon-btn" onClick={toggle} aria-label="Toggle colour theme">
            <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden>
              <circle cx="7" cy="7" r="5.5" fill="none" stroke="currentColor" strokeWidth="1.4" />
              <path d="M7 1.5a5.5 5.5 0 0 1 0 11z" fill="currentColor" />
            </svg>
            {theme === "dark" ? "Light mode" : "Dark mode"}
          </button>
          <span>TrackShift Innovation Challenge 2026 · Team Oura Oura</span>
        </div>
      </nav>
      <main className="main">
        <Suspense fallback={<div className="page"><Loading /></div>}>
          {Page ? <Page /> : <Soon label={current.label} />}
        </Suspense>
      </main>
    </div>
  );
}
