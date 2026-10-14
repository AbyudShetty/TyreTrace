"""Race explorer -- the raw material, before any model touches it."""
from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from tyretrace.app import data, style
from tyretrace.features import traffic

st.title("🏎️ Race explorer")
st.caption("Canonical laps from FastF1 timing data: the same 37-column table the "
           "simulator and the game rig emit.")

kind = st.radio("Session type", ["race", "quali", "practice"], horizontal=True)
df = data.laps(kind)
if not data.need(df, "pull", f"{kind} laps"):
    st.stop()

uids = sorted(df["session_uid"].unique().to_list())
uid = st.selectbox("Session", uids, format_func=data.label)
s = df.filter(pl.col("session_uid") == uid)

c = st.columns(4)
c[0].metric("Laps", f"{s.height:,}")
c[1].metric("Drivers", s["driver"].n_unique())
c[2].metric("Stints", s.select("driver", "stint_id").unique().height)
c[3].metric("Green laps", f"{100*s.filter(pl.col('track_status') == '1').height / max(s.height, 1):.0f}%")

# --------------------------------------------------------------------------
# tyre strategy
# --------------------------------------------------------------------------
st.subheader("Tyre strategy")
st.caption("Each bar is one stint on one set of tyres. In a race, the gaps between "
           "bars are pit stops: tyre age resets while fuel carries on. That reset "
           "is what lets degradation be separated from fuel.")
stints = (s.group_by("driver", "stint_id")
           .agg(pl.col("lap_in_session").min().alias("start"),
                pl.col("lap_in_session").max().alias("end"),
                pl.col("compound_rel").first().cast(pl.Utf8).alias("compound"),
                pl.len().alias("laps"))
           .sort("driver", "start"))
order = (stints.group_by("driver").agg(pl.col("end").max().alias("e"))
               .sort("e", "driver", descending=[True, False])["driver"].to_list())

fig = go.Figure()
seen = set()
for r in stints.iter_rows(named=True):
    comp = r["compound"] or "UNKNOWN"
    fig.add_trace(go.Bar(
        y=[r["driver"]], x=[r["end"] - r["start"] + 1], base=[r["start"] - 0.5],
        orientation="h", marker=dict(color=style.COMPOUND.get(comp, "#777"),
                                     line=dict(color="rgba(0,0,0,0.4)", width=1)),
        name=comp, legendgroup=comp, showlegend=comp not in seen,
        hovertemplate=f"{r['driver']} · {comp}<br>laps {r['start']}–{r['end']}"
                      f" ({r['laps']} timed)<extra></extra>"))
    seen.add(comp)
fig.update_yaxes(categoryorder="array", categoryarray=order[::-1])
fig.update_xaxes(title="lap")
fig.update_layout(barmode="overlay", bargap=0.25)
st.plotly_chart(style.tidy(fig, height=max(320, 22 * len(order))))

# --------------------------------------------------------------------------
# lap-time traces
# --------------------------------------------------------------------------
st.subheader("Lap times")
drivers = sorted(s["driver"].unique().to_list())
a, b, c2 = st.columns([3, 1, 1])
pick = a.multiselect("Drivers", drivers, default=order[:3])
green = b.toggle("Green-flag only", value=True)
clip = c2.toggle("Hide slow outliers", value=True,
                 help="Hides laps more than 7% slower than the session median "
                      "(in/out laps, safety car). Display only; models never drop "
                      "laps this way.")

t = s.filter(pl.col("driver").is_in(pick))
if green:
    t = t.filter(pl.col("track_status") == "1")
if clip:
    t = t.filter(pl.col("lap_time_s") <= 1.07 * s["lap_time_s"].median())

fig = go.Figure()
for drv in pick:
    g = t.filter(pl.col("driver") == drv).sort("lap_in_session")
    if g.height == 0:
        continue
    fig.add_trace(go.Scatter(
        x=g["lap_in_session"].to_list(), y=g["lap_time_s"].to_list(),
        mode="lines+markers", name=drv,
        line=dict(width=1.5),
        marker=dict(size=7, color=[style.COMPOUND.get(str(cc), "#777")
                                   for cc in g["compound_rel"].cast(pl.Utf8).to_list()],
                    line=dict(width=1, color="black")),
        customdata=g.select("tyre_age", "stint_id").to_numpy(),
        hovertemplate=f"{drv} lap %{{x}}<br>%{{y:.3f}} s<br>tyre age %{{customdata[0]}}"
                      "<extra></extra>"))
    for lp in g.filter(pl.col("pit_in"))["lap_in_session"].to_list():
        fig.add_vline(x=lp + 0.5, line_dash="dot", line_color="gray", opacity=0.5)
fig.update_xaxes(title="lap" if kind == "race" else "lap in session")
fig.update_yaxes(title="lap time (s)")
fig.update_layout(title="Marker colour = compound · dotted line = pit entry")
st.plotly_chart(style.tidy(fig, height=440))

# --------------------------------------------------------------------------
# traffic
# --------------------------------------------------------------------------
if kind == "race":
    st.subheader("Traffic")
    st.caption("Gap to the car ahead, from session-clock timing alone (no telemetry). "
               "L3 turns this into a 0–1 wake exposure, exp(−gap / 2 s).")
    g = traffic.add_gap_ahead(s).filter(pl.col("gap_ahead_s").is_not_null())
    if g.height:
        a, b = st.columns(2)
        fig = go.Figure(go.Histogram(x=g["gap_ahead_s"].to_list(), nbinsx=60,
                                     marker_color="#5B8DEF"))
        fig.add_vrect(x0=0, x1=2, fillcolor=style.WARN_FILL, line_width=0,
                      annotation_text="dirty air (< 2 s)")
        fig.update_xaxes(title="gap to car ahead (s)")
        fig.update_layout(title=f"{100*(g['gap_ahead_s'] < 2).mean():.0f}% of laps "
                                "within 2 s of a car")
        a.plotly_chart(style.tidy(fig, height=320, legend=False))

        m = (g.group_by("lap_in_session").agg(pl.col("gap_ahead_s").median().alias("gap"))
              .sort("lap_in_session"))
        fig = go.Figure(go.Scatter(x=m["lap_in_session"].to_list(), y=m["gap"].to_list(),
                                   mode="lines", line=dict(color="#F5A623", width=2)))
        fig.update_xaxes(title="lap")
        fig.update_yaxes(title="median gap (s)")
        fig.update_layout(title="The field spreads out: early laps are dirtier")
        b.plotly_chart(style.tidy(fig, height=320, legend=False))
