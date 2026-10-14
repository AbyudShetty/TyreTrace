"""Game rig -- recover a hidden state the estimator is never shown."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from audit.a5_rig_recovery import build_rig_laps, estimate
from tyretrace.app import data, style
from tyretrace.rig.fake import FakeSession

st.title("🎮 Game rig")
st.markdown(
    "The F1 game broadcasts **true fuel mass and true per-wheel tyre wear** for "
    "every car over UDP. Real F1 never shows either. The rig records a race, "
    "hides every `truth_*` column, gives the estimator only what real timing data "
    "contains, and checks whether it recovers the hidden truth.")
st.warning("**Status:** this runs on a *synthetic* packet stream that goes through "
           "the real decoder and recorder. The byte offsets haven't been checked "
           "against a live game yet. Run `python -m tyretrace.rig.record "
           "--self-check` during a session before trusting real recordings.")

a, b = st.columns([1, 3])
with a:
    deg = st.slider("True degradation (s/lap)", 0.02, 0.12, 0.055, 0.005)
    beta = st.slider("True fuel effect β (s/kg)", 0.015, 0.05, 0.033, 0.001)
    n_laps = st.slider("Race laps", 20, 60, 40)
    seed = st.number_input("Seed", 0, 999, 11)


@st.cache_data(show_spinner="Driving the packet stream…")
def run(deg: float, beta: float, n_laps: int, seed: int):
    fs = FakeSession(n_cars=20, n_laps=n_laps, seed=seed,
                     deg_per_lap=deg, beta_f=beta)
    laps = build_rig_laps(fs)
    observable = laps.drop([c for c in laps.columns if c.startswith("truth_")])
    return laps, estimate(observable), fs.kappa


laps, est, kappa = run(deg, beta, n_laps, int(seed))
d_m, d_se = est["deg"]
f_m, f_se = -est["lap_coef"][0], est["lap_coef"][1]
true_fuel = beta * kappa

with a:
    st.metric("Recovered degradation", f"{d_m:.4f} ± {3*d_se:.4f}",
              delta=f"{d_m-deg:+.4f} vs truth", delta_color="off")
    st.metric("Recovered fuel pace", f"{f_m:.4f} ± {3*f_se:.4f}",
              delta=f"{f_m-true_fuel:+.4f} vs truth", delta_color="off")
    ok = abs(d_m - deg) <= 3 * d_se and abs(f_m - true_fuel) <= 3 * f_se
    st.markdown(f"**Gate (within 3 SE):** {'🟢 PASS' if ok else '🔴 FAIL'}")

with b:
    fig = go.Figure()
    for name, m, se, truth in (("degradation", d_m, d_se, deg),
                               ("fuel pace", f_m, f_se, true_fuel)):
        fig.add_trace(go.Scatter(x=[m], y=[name], mode="markers",
                                 error_x=dict(type="data", array=[3 * se]),
                                 marker=dict(size=14, color="#1F6FD1"),
                                 name="recovered ± 3 SE", showlegend=name == "fuel pace"))
        fig.add_trace(go.Scatter(x=[truth], y=[name], mode="markers",
                                 marker=dict(size=18, symbol="x", color="#E10600"),
                                 name="hidden truth", showlegend=name == "fuel pace"))
    fig.update_xaxes(title="s/lap")
    fig.update_layout(title="Recovered vs hidden truth")
    st.plotly_chart(style.tidy(fig, height=260))

    cars = sorted(laps["driver"].unique().to_list())
    car = st.selectbox("Car", cars)
    g = laps.filter(pl.col("driver") == car).sort("lap_in_session")
    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["lap_in_session"].to_list(), y=g["truth_fuel_kg"].to_list(),
                             mode="lines", line=dict(color="#1F6FD1", width=3),
                             name="fuel (kg)"))
    fig.update_xaxes(title="lap")
    fig.update_layout(title="Hidden: fuel mass")
    c1.plotly_chart(style.tidy(fig, height=300, legend=False))
    fig = go.Figure()
    for w, col in (("fl", "#E10600"), ("fr", "#F5A623"), ("rl", "#2BA84A"), ("rr", "#5B8DEF")):
        fig.add_trace(go.Scatter(x=g["lap_in_session"].to_list(),
                                 y=g[f"truth_wear_{w}"].to_list(), mode="lines",
                                 name=w.upper(), line=dict(color=col)))
    fig.update_xaxes(title="lap")
    fig.update_yaxes(title="wear %")
    fig.update_layout(title="Hidden: tyre wear per wheel (resets at stops)")
    c2.plotly_chart(style.tidy(fig, height=300))

st.caption(f"{laps.height} laps from {laps['driver'].n_unique()} cars. In the "
           "synthetic stream fuel burns and tyres wear exactly linearly, so recovery "
           "is expected. The real test is the live game, where fuel burn depends on "
           "the fuel mix and safety cars and wear isn't linear.")
