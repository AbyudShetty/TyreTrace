"""Held-out predictions -- forecasts against what actually happened."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from tyretrace.app import data, style

st.title("🎯 Held-out predictions")
st.markdown(
    "Every model was fitted on the **first 70% of each race** and asked to forecast "
    "the rest. Shaded bands are its 50% and 90% predictive intervals. Red points "
    "fell outside the 90% band. Too many red points means the model is overconfident.")

pred = data.artifact("l4", "predictive")
if not data.need(pred, "l4", "held-out predictions"):
    st.stop()
base = data.artifact("baselines", "predictive")
if base is not None:              # same laps, same columns -- protocol baselines
    pred = pl.concat([pred, base.select(pred.columns)])

pred = pred.with_columns(
    ((pl.col("y") >= pl.col("q05")) & (pl.col("y") <= pl.col("q95"))).alias("in90"),
    (pl.col("y") - pl.col("q50")).alias("resid"))

rungs = sorted(pred["rung"].unique().to_list())
a, b, c = st.columns(3)
rung = a.selectbox("Model", rungs, index=rungs.index("L4") if "L4" in rungs else 0)
wks = sorted(pred["weekend"].unique().to_list())
wk = b.selectbox("Race", wks, format_func=data.weekend_label)
p = pred.filter((pl.col("rung") == rung) & (pl.col("weekend") == wk))
drivers = sorted(p["driver"].unique().to_list())
drv = c.selectbox("Driver", drivers)
g = p.filter(pl.col("driver") == drv).sort("lap_in_session")

m = st.columns(4)
m[0].metric(f"{rung} coverage, this race", f"{p['in90'].mean():.1%}")
m[1].metric(f"{rung} coverage, all races", f"{pred.filter(pl.col('rung') == rung)['in90'].mean():.1%}")
m[2].metric("CRPS, this driver", f"{g['crps'].mean():.3f} s")
m[3].metric("Laps outside 90%", f"{(~g['in90']).sum()} / {g.height}")

# --------------------------------------------------------------------------
# forecast vs actual
# --------------------------------------------------------------------------
fig = go.Figure()
for stint, gs in g.group_by("stint", maintain_order=True):
    x = gs["lap_in_session"].to_list()
    show = stint == g["stint"][0]
    fig.add_trace(go.Scatter(x=x + x[::-1], y=gs["q95"].to_list() + gs["q05"].to_list()[::-1],
                             fill="toself", fillcolor="rgba(91,141,239,0.18)",
                             line=dict(width=0), name="90% interval",
                             legendgroup="90", showlegend=show, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x + x[::-1], y=gs["q75"].to_list() + gs["q25"].to_list()[::-1],
                             fill="toself", fillcolor="rgba(91,141,239,0.38)",
                             line=dict(width=0), name="50% interval",
                             legendgroup="50", showlegend=show, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=gs["q50"].to_list(), mode="lines",
                             line=dict(color="#1F4FB0", width=2), name="median forecast",
                             legendgroup="med", showlegend=show))
fig.add_trace(go.Scatter(
    x=g["lap_in_session"].to_list(), y=g["y"].to_list(), mode="markers", name="actual",
    marker=dict(size=8, color=np.where(g["in90"].to_numpy(), "black", "#E10600").tolist()),
    customdata=g.select("tyre_age", "compound").to_numpy(),
    hovertemplate="lap %{x}<br>%{y:.3f} s<br>tyre age %{customdata[0]} · "
                  "%{customdata[1]}<extra></extra>"))
fig.update_xaxes(title="race lap (held-out block)")
fig.update_yaxes(title="lap time (s)")
fig.update_layout(title=f"{drv} · {data.weekend_label(wk)} · {rung}")
st.plotly_chart(style.tidy(fig, height=440))

# --------------------------------------------------------------------------
# where the misses are
# --------------------------------------------------------------------------
st.subheader("Where the intervals miss")
bins = [0, 5, 10, 15, 20, 25, 30, 999]
names = ["1–5", "6–10", "11–15", "16–20", "21–25", "26–30", "31+"]
pa = pred.with_columns(pl.col("tyre_age").cut(bins[1:-1], labels=names,
                                              left_closed=False).alias("age_bin"))

a, b = st.columns(2)
fig = go.Figure()
fig.add_hrect(y0=style.BAND[0], y1=style.BAND[1], fillcolor=style.BAND_FILL,
              line_width=0, annotation_text="ship band")
for r in rungs:
    t = (pa.filter(pl.col("rung") == r).group_by("age_bin")
           .agg(pl.col("in90").mean().alias("cov"), pl.len().alias("n")))
    t = t.with_columns(pl.col("age_bin").cast(pl.Utf8))
    lookup = dict(zip(t["age_bin"].to_list(), t["cov"].to_list()))
    fig.add_trace(go.Scatter(x=names, y=[lookup.get(n) for n in names],
                             mode="lines+markers", name=r,
                             line=dict(color=style.RUNG.get(r, style.BASELINE), width=2.5,
                                       dash="dot" if r.startswith("B") else "solid")))
fig.update_xaxes(title="tyre age (laps)")
fig.update_yaxes(title="90% coverage", range=[0.6, 1.0])
fig.update_layout(title="Coverage by tyre age")
a.plotly_chart(style.tidy(fig, height=380))

t = (pa.filter(pl.col("rung") == rung).group_by("age_bin")
       .agg((pl.col("y") > pl.col("q95")).mean().alias("slow"),
            (pl.col("y") < pl.col("q05")).mean().alias("fast")))
t = t.with_columns(pl.col("age_bin").cast(pl.Utf8))
slow = dict(zip(t["age_bin"].to_list(), t["slow"].to_list()))
fast = dict(zip(t["age_bin"].to_list(), t["fast"].to_list()))
fig = go.Figure()
fig.add_trace(go.Bar(x=names, y=[slow.get(n) for n in names], name="slower than forecast",
                     marker_color="#E10600"))
fig.add_trace(go.Bar(x=names, y=[fast.get(n) for n in names], name="faster than forecast",
                     marker_color="#1F6FD1"))
fig.add_hline(y=0.05, line_dash="dash", line_color="gray",
              annotation_text="5% each side if calibrated")
fig.update_xaxes(title="tyre age (laps)")
fig.update_yaxes(title="share of laps outside the 90% band", tickformat=".0%")
fig.update_layout(title=f"{rung}: which side the misses fall on", barmode="group")
b.plotly_chart(style.tidy(fig, height=380))
st.caption("A calibrated model misses 5% of laps on each side. The misses here are "
           "lopsided: far more laps come in *slower* than forecast, increasingly so "
           "on older tyres, while the median forecast stays within ~0.1 s. The noise "
           "model is symmetric, but real lap times have a long slow tail (traffic, "
           "mistakes, tyres going off). That points at a right-skewed likelihood, or "
           "a regime model with a cliff state, as the next calibration fix.")

a, b = st.columns(2)
r_ = pred.filter(pl.col("rung") == rung)["resid"].to_numpy()
sd = float(np.std(r_))
xs = np.linspace(-4 * sd, 4 * sd, 200)
fig = go.Figure()
fig.add_trace(go.Histogram(x=r_, histnorm="probability density", nbinsx=120,
                           marker_color=style.RUNG.get(rung, style.BASELINE), name="actual − median"))
fig.add_trace(go.Scatter(x=xs, y=np.exp(-0.5 * (xs / sd) ** 2) / (sd * np.sqrt(2 * np.pi)),
                         mode="lines", line=dict(color="black", dash="dash"),
                         name="Gaussian, same SD"))
z = np.abs(r_ - r_.mean()) / sd
fig.update_xaxes(title="forecast error (s)", range=[-4 * sd, 4 * sd])
fig.update_layout(title=f"Heavy tails: {np.mean(z > 3):.2%} beyond 3 SD (Gaussian: 0.27%)")
a.plotly_chart(style.tidy(fig, height=360))

per = (pred.filter(pl.col("rung") == rung).group_by("weekend")
           .agg(pl.col("in90").mean().alias("cov")).sort("weekend"))
fig = go.Figure(go.Bar(y=[data.weekend_label(w) for w in per["weekend"].to_list()],
                       x=per["cov"].to_list(), orientation="h",
                       marker_color=[("#2BA84A" if style.BAND[0] <= v <= style.BAND[1]
                                      else "#E10600") for v in per["cov"].to_list()]))
fig.add_vrect(x0=style.BAND[0], x1=style.BAND[1], fillcolor=style.BAND_FILL, line_width=0)
fig.update_xaxes(title="90% coverage", range=[0.5, 1.0])
fig.update_layout(title=f"{rung} coverage by race")
b.plotly_chart(style.tidy(fig, height=360, legend=False))
