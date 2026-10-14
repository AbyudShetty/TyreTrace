"""Track evolution -- why practice fails and qualifying works."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from tyretrace.app import data, style
from tyretrace.data import fastf1_pull as fp
from tyretrace.models import track_evolution as te

st.title("🛣️ Track evolution")
st.markdown(
    "As rubber goes down, the track gets faster through a session. That gain "
    "moves with the session clock, and so does fuel burn. To separate them you "
    "need a session where fuel *doesn't* change.")

quali = data.laps("quali")
prac = data.laps("practice")

# --------------------------------------------------------------------------
# qualifying: the anchor
# --------------------------------------------------------------------------
st.subheader("Qualifying: every car on minimum fuel")
if data.need(quali, "pull", "qualifying laps"):
    fits = te.fit(quali)
    summ = te.summary(fits)
    c = st.columns(4)
    c[0].metric("Median τ", f"{summ['median_tau']:+.4f} s/min")
    c[1].metric("Sessions with τ < 0", f"{100*summ['frac_negative']:.0f}%")
    c[2].metric("Median z", f"{summ['median_z']:.1f}")
    c[3].metric("Track gain per session", f"{abs(summ['median_total_s']):.2f} s faster",
                help="Median lap-time improvement from first to last push lap, "
                     "from track evolution alone.")

    a, b = st.columns([3, 2])
    uids = [f.session_uid for f in fits]
    uid = a.selectbox("Qualifying session", uids, format_func=data.label)
    f = next(x for x in fits if x.session_uid == uid)

    push = te.push_laps(quali.filter(pl.col("session_uid") == uid))
    push = push.with_columns(
        (pl.col("lap_time_s") - pl.col("lap_time_s").mean().over("driver")).alias("rel"),
        (pl.col("session_clock_s") / 60.0).alias("minute"))
    t0 = push["minute"].mean()
    xs = np.linspace(push["minute"].min(), push["minute"].max(), 2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=push["minute"].to_list(), y=push["rel"].to_list(),
                             mode="markers", text=push["driver"].to_list(),
                             marker=dict(size=7, color="#5B8DEF", opacity=0.7),
                             name="push lap",
                             hovertemplate="%{text}<br>minute %{x:.1f}<br>%{y:+.3f} s"
                                           "<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=f.tau_per_min * (xs - t0), mode="lines",
                             line=dict(color=style.COMPOUND["SOFT"], width=3),
                             name=f"τ = {f.tau_per_min:+.4f} s/min (z {f.z:.1f})"))
    fig.update_xaxes(title="session minute")
    fig.update_yaxes(title="lap time vs driver's own mean (s)")
    fig.update_layout(title="Each driver compared with themselves: the track gets faster")
    a.plotly_chart(style.tidy(fig, height=400))

    fs = sorted(fits, key=lambda x: x.session_uid)
    fig = go.Figure(go.Scatter(
        x=[x.tau_per_min for x in fs], y=[data.label(x.session_uid) for x in fs],
        error_x=dict(type="data", array=[2 * x.se for x in fs]),
        mode="markers", marker=dict(size=10, color=style.COMPOUND["SOFT"])))
    fig.add_vline(x=0, line_color="gray")
    fig.update_xaxes(title="τ (s/min) ± 2 SE")
    fig.update_layout(title="All sessions agree on the sign")
    b.plotly_chart(style.tidy(fig, height=400, legend=False))
    st.caption("Fitted with driver fixed effects plus tyre age, compound and "
               "fresh-tyre controls. Controls move τ by under 5%, so this is the "
               "track, not Q3 running fresher tyres.")

# --------------------------------------------------------------------------
# practice: the confound
# --------------------------------------------------------------------------
st.subheader("Why practice can't do this")
st.markdown(
    "The original plan used practice: different cars carry different fuel at the "
    "same minute. Measurement says otherwise. Teams run **low-fuel qualifying "
    "simulations early and heavy race simulations late**, so fuel load "
    "*rises* with the clock, against the track's improvement.")

if data.need(prac, "pull", "practice laps"):
    lr = fp.long_runs(prac, min_laps=4)
    runs = (lr.group_by("session_uid", "driver", "stint_id")
              .agg((pl.col("session_clock_s").min() / 60).alias("start_min"),
                   pl.col("lap_time_s").median().alias("pace"),
                   pl.len().alias("laps"))
              .with_columns((pl.col("pace") - pl.col("pace").median().over("session_uid"))
                            .alias("rel_pace")))
    per = (runs.group_by("session_uid")
               .agg(pl.corr("start_min", "rel_pace").alias("r"), pl.len().alias("n"))
               .filter(pl.col("n") >= 8).sort("session_uid"))
    a, b = st.columns([3, 2])
    uid = a.selectbox("Practice session", per["session_uid"].to_list(),
                      format_func=data.label)
    g = runs.filter(pl.col("session_uid") == uid)
    x, y = g["start_min"].to_numpy(), g["rel_pace"].to_numpy()
    slope, icpt = np.polyfit(x, y, 1) if len(x) > 2 else (0.0, 0.0)
    xs = np.linspace(x.min(), x.max(), 2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", text=g["driver"].to_list(),
                             marker=dict(size=(4 + 1.2 * g["laps"].to_numpy()).tolist(),
                                         color="#1F6FD1", opacity=0.6),
                             name="run (size = laps)",
                             hovertemplate="%{text}<br>start minute %{x:.0f}"
                                           "<br>%{y:+.2f} s<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=icpt + slope * xs, mode="lines",
                             line=dict(color="#9B59B6", width=3),
                             name=f"trend {slope:+.3f} s/min"))
    fig.update_xaxes(title="run start (session minute)")
    fig.update_yaxes(title="run pace vs session median (s)")
    fig.update_layout(title="Later runs are slower: heavier fuel beats a faster track")
    a.plotly_chart(style.tidy(fig, height=400))

    fig = go.Figure(go.Bar(x=per["r"].to_list(),
                           y=[data.label(u) for u in per["session_uid"].to_list()],
                           orientation="h", marker_color="#9B59B6"))
    fig.add_vline(x=0, line_color="gray")
    fig.update_xaxes(title="corr(run start, run pace)", range=[-1, 1])
    fig.update_layout(title=f"Median r = {per['r'].median():+.2f}: positive almost everywhere")
    b.plotly_chart(style.tidy(fig, height=400, legend=False))

    st.warning("A practice-only fit gets τ with the **wrong sign**: it says the "
               "track gets *slower*. The fix is the chain **qualifying → τ, "
               "race + τ → fuel and degradation**, with practice consuming both "
               "instead of identifying anything.")
