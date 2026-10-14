"""Compound contrasts -- estimand E1, within-weekend degradation differences."""
from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from tyretrace.app import data, style

st.title("⚖️ Compound contrasts")
st.markdown(
    "How much faster does a SOFT degrade than a HARD **at the same race**? Within "
    "one weekend the labels are unambiguous, so this contrast (estimand **E1**) "
    "needs no Pirelli compound map. Across weekends 'SOFT' is a different rubber "
    "each time, so these are compared, not pooled.")

deg = data.artifact("l2", "degradation")
if not data.need(deg, "l2", "degradation estimates"):
    st.stop()

con = deg.filter(pl.col("term") == "SOFT-HARD").sort("weekend")
if con.height:
    sig = (con["q05"] > 0) | (con["q95"] < 0)
    st.metric("Races where SOFT−HARD is resolved", f"{int(sig.sum())} of {con.height}",
              help="90% interval excludes zero.")
    fig = go.Figure(go.Scatter(
        x=con["mean"].to_list(), y=[data.weekend_label(w) for w in con["weekend"].to_list()],
        error_x=dict(type="data", symmetric=False,
                     array=(con["q95"] - con["mean"]).to_list(),
                     arrayminus=(con["mean"] - con["q05"]).to_list()),
        mode="markers",
        marker=dict(size=11, color=["#E10600" if s else "#9AA0A6" for s in sig.to_list()])))
    fig.add_vline(x=0, line_color="gray")
    fig.update_xaxes(title="SOFT − HARD degradation (s/lap), 90% interval")
    fig.update_layout(title="Red = interval excludes zero")
    st.plotly_chart(style.tidy(fig, height=420, legend=False))

st.subheader("Degradation by compound, per race")
per = deg.filter(pl.col("term") != "SOFT-HARD")
fig = go.Figure()
for comp in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
    g = per.filter(pl.col("term") == comp).sort("weekend")
    if g.height == 0:
        continue
    fig.add_trace(go.Scatter(
        x=[data.weekend_label(w) for w in g["weekend"].to_list()], y=g["mean"].to_list(),
        error_y=dict(type="data", symmetric=False,
                     array=(g["q95"] - g["mean"]).to_list(),
                     arrayminus=(g["mean"] - g["q05"]).to_list()),
        mode="markers", name=comp,
        marker=dict(size=10, color=style.COMPOUND[comp], line=dict(width=1, color="black"))))
fig.update_yaxes(title="degradation (s/lap of tyre age)")
fig.update_layout(scattermode="group")
st.plotly_chart(style.tidy(fig, height=420))
st.caption("A compound with no long green-flag stints at a race gets a wide "
           "interval that stays near the prior. That's the model saying it "
           "doesn't know, not a measurement.")
