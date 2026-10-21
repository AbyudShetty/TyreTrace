"""FP -> race backtest -- does the fuel / degradation split survive a real forecast?"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from tyretrace.app import data, style

st.title("🔁 FP → race backtest")
st.markdown(
    "Every other test forecast the end of a race from its start, which only needs "
    "the *combined* fuel-and-tyre slope. Here each race is forecast **from its own "
    "weekend's practice**. Only the information a strategist has before the "
    "race is used: practice, qualifying, races from *earlier* weekends, and each "
    "driver's first 5 laps as a pace anchor. The laps **after a pit stop** are "
    "where fuel and tyre wear have to be separated correctly.")

METHOD_COLOR = {"identified": "#E10600", "no-track": "#F5A623",
                "conventional": "#5B8DEF", "naive": style.BASELINE}
METHOD_NOTE = {
    "identified": "fuel effect learned from earlier races + track evolution from qualifying",
    "no-track": "identified fuel, track evolution ignored (ablation)",
    "conventional": "fuel assumed 0.030 s/kg, no track evolution",
    "naive": "no fuel correction: the practice slope *is* degradation",
}

wk = data.artifact("backtest", "weekends")
ft = data.artifact("backtest", "fuel_track")
pred = data.artifact("backtest", "predictive")
if not data.need(wk, "backtest", "the backtest"):
    st.stop()

methods = [m for m in METHOD_COLOR if m in wk["method"].unique().to_list()]
summ = {m: {k: data.metric("backtest", f"{k}_{m.replace('-', '_')}")
            for k in ("crps", "crps_after", "bias_after", "cov90", "stint_mae", "n_after")}
        for m in methods}
gate = data.result("backtest").get("gate", "-")

c = st.columns(4)
for col, m in zip(c, methods):
    s = summ[m]
    col.metric(m, f"{s['crps_after']:.3f}" if s["crps_after"] else "—",
               help=f"{METHOD_NOTE[m]}. Held-out CRPS on laps after a pit stop "
                    f"(lower is better).")
st.caption(f"CRPS after a pit stop, pooled over {data.metric('backtest', 'weekends')} "
           f"weekends and {summ['identified']['n_after']:,} laps. "
           f"Gate: {style.GATE.get(gate, '⚪')} **{gate}**: the identified method must "
           "beat both conventional and naive here.")

if summ["identified"]["crps_after"] and summ["conventional"]["crps_after"]:
    i, cv, nv = (summ[m]["crps_after"] for m in ("identified", "conventional", "naive"))
    st.markdown(
        f"**What it shows.** The identified method beats naive by "
        f"**{100 * (1 - i / nv):.0f}%**, and dropping track evolution makes it worse, "
        f"so both earn their place. But it **trails the conventional fixed fuel "
        f"assumption by {100 * (i / cv - 1):.1f}%**. The slope chart below says why: the "
        "fuel effect learned from earlier races is too small, so races speed up more "
        "than it predicts. That's out-of-sample evidence on the project's open "
        "fuel-pace gap. The race-based identification is biased low; physics isn't wrong.")

# --------------------------------------------------------------------------
# per weekend
# --------------------------------------------------------------------------
st.subheader("Weekend by weekend")
a, b = st.columns(2)
fig = go.Figure()
for m in methods:
    g = wk.filter(pl.col("method") == m).sort("weekend")
    fig.add_trace(go.Bar(x=[data.weekend_label(w) for w in g["weekend"].to_list()],
                         y=g["crps_after"].to_list(), name=m,
                         marker_color=METHOD_COLOR[m]))
fig.update_yaxes(title="CRPS after a stop (s)")
fig.update_layout(barmode="group", title="Forecast error after pit stops")
a.plotly_chart(style.tidy(fig, height=400))

if ft is not None and "slope_actual" in ft.columns:
    f = ft.sort("weekend")
    x = [data.weekend_label(w) for w in f["weekend"].to_list()]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=f["slope_actual"].to_list(), mode="markers",
                             marker=dict(size=14, symbol="x", color="black"),
                             name="what the race did"))
    for col, name, color in (("slope_identified", "identified", METHOD_COLOR["identified"]),
                             ("slope_conventional", "conventional", METHOD_COLOR["conventional"])):
        fig.add_trace(go.Scatter(x=x, y=f[col].to_list(), mode="markers+lines",
                                 marker=dict(size=9, color=color), line=dict(width=1),
                                 name=name))
    fig.update_yaxes(title="race lap slope, fuel + track (s/lap)")
    fig.update_layout(title="How fast each race got faster per lap")
    b.plotly_chart(style.tidy(fig, height=400))
    b.caption("More negative = the race sped up more per lap as fuel burned off. "
              "Black ✕ = measured afterwards; it was never used in the forecast.")

# --------------------------------------------------------------------------
# forecast explorer
# --------------------------------------------------------------------------
st.subheader("Forecast vs actual")
if data.need(pred, "backtest", "per-lap forecasts"):
    a, b, c = st.columns(3)
    w = a.selectbox("Race", sorted(pred["weekend"].unique().to_list()),
                    format_func=data.weekend_label)
    p = pred.filter(pl.col("weekend") == w)
    drv = b.selectbox("Driver", sorted(p["driver"].unique().to_list()))
    shown = c.multiselect("Methods", methods, default=["identified", "conventional"])
    g = p.filter(pl.col("driver") == drv).sort("lap_in_session")

    fig = go.Figure()
    stop_laps = g.filter(pl.col("after_stop"))["lap_in_session"]
    if stop_laps.len():
        fig.add_vrect(x0=stop_laps.min() - 0.5, x1=g["lap_in_session"].max() + 0.5,
                      fillcolor="rgba(0,0,0,0.04)", line_width=0,
                      annotation_text="after a stop", annotation_position="top left")
    for m in shown:
        gm = g.filter(pl.col("method") == m)
        x = gm["lap_in_session"].to_list()
        col = METHOD_COLOR[m]
        rgb = tuple(int(col.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        fig.add_trace(go.Scatter(x=x + x[::-1], y=gm["q95"].to_list() + gm["q05"].to_list()[::-1],
                                 fill="toself", fillcolor=f"rgba{rgb + (0.12,)}",
                                 line=dict(width=0), name=f"{m} 90%", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=gm["q50"].to_list(), mode="lines",
                                 line=dict(color=col, width=2.5), name=f"{m} median"))
    base = g.filter(pl.col("method") == (shown[0] if shown else methods[0]))
    fig.add_trace(go.Scatter(x=base["lap_in_session"].to_list(), y=base["y"].to_list(),
                             mode="markers", marker=dict(size=7, color="black"),
                             name="actual"))
    fig.update_xaxes(title="race lap")
    fig.update_yaxes(title="lap time (s)")
    fig.update_layout(title=f"{drv} · {data.weekend_label(w)}")
    st.plotly_chart(style.tidy(fig, height=440))

if ft is not None:
    with st.expander("What each weekend's forecast was built from"):
        cols = [c for c in ("weekend", "races_used", "fuel_pace", "fuel_pace_sd", "s",
                            "deg_prior", "deg_prior_sd", "prior_only_compounds")
                if c in ft.columns]
        st.dataframe(ft.select(cols).to_pandas(), hide_index=True)
        st.caption("fuel_pace: s/lap learned from earlier races. s: how much of "
                   "qualifying's track evolution carries into the race. deg_prior: "
                   "degradation prior from earlier races' pit stops; practice updates "
                   "it as far as it's informative. prior_only_compounds: raced but "
                   "never run in practice.")

st.caption("Assumptions: practice track evolution runs at the qualifying rate; "
           "κ = 110 kg / race laps; first 5 green laps per driver are the pace "
           "anchor; lap noise from earlier races. Coverage is ~0.75 for every method: "
           "intervals are too narrow when extrapolating 30+ laps.")
