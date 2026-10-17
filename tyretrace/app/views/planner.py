"""Run planner -- the FP2 programme that teaches the most, by information gain."""
from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from tyretrace.app import style
from tyretrace.planner import eig as E

st.title("🧭 Run planner")
st.markdown(
    "Every model so far *estimates* from whatever running happened. This page "
    "turns it around: **which practice runs would teach the most?** Each candidate "
    "run is scored by its exact expected information gain about the quantities a "
    "race strategy needs. The planner then fills the session greedily, best "
    "information per minute first.")

with st.sidebar:
    st.subheader("Planner settings")
    session = st.slider("Session length (min)", 30, 90, 60, 5)
    lap_s = st.slider("Lap time (s)", 70, 110, 90)
    sigma = st.slider("Lap-time noise σ (s)", 0.10, 0.60, 0.30, 0.05,
                      help="How noisy a single practice lap is. Higher noise favours "
                           "longer runs over clever ones.")
    bf_sd = st.slider("Fuel effect uncertainty (s/kg)", 0.002, 0.020, 0.010, 0.001,
                      help="Prior sd on β_f around the paddock convention 0.033 s/kg. "
                           "Our race data says the true value is unresolved.")
    deg_sd = st.slider("Degradation uncertainty (s/lap)", 0.01, 0.08, 0.04, 0.005)
    target = st.selectbox("Optimise for", list(E.TARGETS))
    reserve = st.toggle("Keep a qualifying simulation", value=True,
                        help="Teams need a low-fuel soft run regardless. Keeping it in "
                             "both programmes makes the comparison about long-run "
                             "choices only; turning it off flatters the planner.")
    swaps = st.toggle("Allow mid-run tyre changes", value=True,
                      help="A tyre change without refuelling resets tyre age while fuel "
                           "carries on: the practice-session version of a pit stop.")

prior = E.Prior(beta_f=(0.033, bf_sd), deg=(0.05, deg_sd), sigma=sigma, lap_s=float(lap_s))
A = E.TARGETS[target]
start = E.Programme().add(E.QUALI_SIM, prior) if reserve else None
steps = E.plan(prior, A, session_min=float(session),
               cands=E.candidates(allow_swap=swaps), start=start)
planned = E.programme(steps, start)
std = E.standard_programme(prior)
std_fits = std.end(prior) <= session

g_plan = E.eig(planned, prior, A)
g_std = E.eig(std, prior, A)

c = st.columns(3)
c[0].metric("Planned programme", f"{g_plan:.2f} bits")
c[1].metric("Typical FP2", f"{g_std:.2f} bits" if std_fits else "doesn't fit",
            help="Quali sim on softs, then long runs on medium and hard.")
if std_fits and g_std > 0:
    c[2].metric("Extra information", f"{100 * (g_plan / g_std - 1):+.0f}%")
st.caption("One bit = the uncertainty about the target halves (in volume). Exact for "
           "a linear-Gaussian lap model, so no simulation noise.")

# --------------------------------------------------------------------------
# timelines
# --------------------------------------------------------------------------
st.subheader("Session timeline")


def segments(prog: E.Programme):
    """(start, end, compound, label) for each tyre segment of each run."""
    lap_min = prior.lap_s / 60.0
    for r, t0 in zip(prog.runs, prog.starts):
        end = t0 + r.minutes(prior.lap_s)
        if r.swap_to:
            mid = t0 + (E.OUT_IN_LAPS / 2 + r.laps // 2 + E.SWAP_LAPS / 2) * lap_min
            yield t0, mid, r.compound, r.label()
            yield mid, end, r.swap_to, r.label()
        else:
            yield t0, end, r.compound, r.label()


fig = go.Figure()
seen = set()
for row, prog in (("Typical FP2", std), ("Planned", planned)):
    for t0, t1, comp, lab in segments(prog):
        fig.add_trace(go.Bar(
            y=[row], x=[t1 - t0], base=[t0], orientation="h",
            marker=dict(color=style.COMPOUND[comp], line=dict(color="black", width=1)),
            name=comp, legendgroup=comp, showlegend=comp not in seen,
            hovertemplate=f"{lab}<br>min %{{base:.0f}}–{t1:.0f}<extra></extra>"))
        seen.add(comp)
fig.add_vline(x=session, line_dash="dash", line_color="gray", annotation_text="chequered flag")
fig.update_xaxes(title="session minute", range=[0, max(session, std.end(prior)) + 2])
fig.update_layout(barmode="overlay", bargap=0.4)
st.plotly_chart(style.tidy(fig, height=230))

if reserve:
    st.caption("The 3-lap soft qualifying simulation at minute 5 is fixed in both "
               "programmes; the table lists what the planner added after it.")
if steps:
    st.dataframe(pl.DataFrame([{
        "start (min)": round(s.start, 1), "run": s.run.label(),
        "timed laps": s.run.laps, "info added (bits)": round(s.gain_bits, 3),
        "running total": round(s.total_bits, 3)} for s in steps]).to_pandas(),
        hide_index=True)
else:
    st.info("No run fits in this session length.")

# --------------------------------------------------------------------------
# what each programme leaves uncertain
# --------------------------------------------------------------------------
st.subheader("What's still uncertain after the session")
sd = {name: E.posterior_sd(p, prior) for name, p in
      (("before FP2", E.Programme()), ("typical FP2", std), ("planned", planned))}
keys = ["deg_SOFT", "deg_MEDIUM", "deg_HARD", "SOFT−HARD"]
fig = go.Figure()
for name, col in (("before FP2", "#B0B4BA"), ("typical FP2", "#5B8DEF"),
                  ("planned", "#E10600")):
    fig.add_trace(go.Bar(x=[k.replace("deg_", "") for k in keys],
                         y=[sd[name][k] for k in keys], name=name, marker_color=col))
fig.update_yaxes(title="posterior sd (s/lap)")
fig.update_layout(barmode="group", title="Lower is better: degradation per compound")
a, b = st.columns([2, 1])
a.plotly_chart(style.tidy(fig, height=360))
with b:
    st.metric("Fuel effect sd, typical FP2", f"{sd['typical FP2']['beta_f']:.4f} s/kg")
    st.metric("Fuel effect sd, planned", f"{sd['planned']['beta_f']:.4f} s/kg")

st.info(
    "**Why the planner likes tyre changes.** Within one run, fuel falls, tyre age "
    "rises and the track improves in lock step, so a run only reveals their "
    "combined effect. A tyre change *without refuelling* resets tyre age while fuel "
    "carries on: the one move that separates wear from fuel. It costs two "
    "untimed laps, and on short runs noise beats the benefit, so the planner only "
    "uses it on runs long enough to pay off.")
st.caption("v1 assumptions: linear degradation (no cliff), known fuel loads (the "
           "team's view), one car, no tyre-allocation limit, out/in laps untimed, "
           "4 min garage turnaround. Track evolution uses the qualifying anchor as its prior.")
