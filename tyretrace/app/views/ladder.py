"""Model ladder -- every rung, its gate, its calibration and its posteriors."""
from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from tyretrace.app import data, style
from tyretrace.eval import metrics

st.title("🪜 Model ladder")
st.markdown(
    "Each rung adds one idea and has to earn its place: **better held-out CRPS** "
    "*and* **90% intervals that cover 85–95% of laps**. A sharper model with "
    "dishonest intervals is a regression, not progress.")

ADDS = {
    "L0": "Pooled: driver pace, one fused fuel+track slope, one degradation slope",
    "L1": "+ hierarchy: drivers partially pooled, degradation per compound",
    "L2": "Splits fuel from track evolution using variation across circuits",
    "L3": "+ dirty air (wake exposure from gap to car ahead)",
    "L4": "+ per-stint offsets, heavy-tailed (Student-t) noise",
    "L5": "+ noise growing with tyre age (rejected: the real shape is U)",
}
REAL = ["L2", "L3", "L4", "L5"]

# --------------------------------------------------------------------------
# gate table
# --------------------------------------------------------------------------
st.subheader("Ship gate")
rows, base = [], None
for r in REAL:
    crps = data.metric("l4", f"crps_{r.lower()}")
    cov = data.metric("l4", f"cov90_{r.lower()}")
    if crps is None or cov is None:
        continue
    sc = {"crps": crps, "coverage_90": cov}
    ok, why = metrics.passes_ship_gate(sc, base)
    rows.append({"rung": r, "adds": ADDS[r], "CRPS": round(crps, 4),
                 "MAE": round(data.metric("l4", f"mae_{r.lower()}", float("nan")), 4),
                 "cov90": round(cov, 3),
                 "gate": ("🟢 SHIP" if ok else "🟡 HOLD") if r != "L5" else "🔴 REJECTED",
                 "why": why})
    base = sc
if rows:
    st.dataframe(pl.DataFrame(rows).to_pandas(), hide_index=True)
    st.caption("Real races, same held-out laps for every rung: the last 30% of each "
               "race, so the models extrapolate forward through fuel burn and tyre wear.")
else:
    data.need(None, "l4", "the ladder")

twin = [(r, data.metric("ladder", f"twin_crps_{r.lower()}"),
         data.metric("ladder", f"twin_cov90_{r.lower()}")) for r in ("L0", "L1")]
if all(c is not None for _, c, _ in twin):
    with st.expander("L0 / L1: the identification core, on the synthetic twin"):
        st.dataframe(pl.DataFrame([{"rung": r, "adds": ADDS[r], "CRPS": round(c, 4),
                                    "cov90": round(v, 3)} for r, c, v in twin]).to_pandas(),
                     hide_index=True)
        st.caption("Scored on simulated races with known truth, so not comparable "
                   "with the real-race numbers above.")

# --------------------------------------------------------------------------
# calibration curves
# --------------------------------------------------------------------------
st.subheader("Calibration: do the intervals mean what they say?")
st.caption("A model whose X% intervals catch X% of laps sits on the diagonal. "
           "**Below the diagonal = overconfident**: intervals too narrow.")

cal = data.artifact("l4", "calibration")
twin_cal = data.artifact("ladder", "calibration_twin")
a, b = st.columns(2)


def cal_fig(df: pl.DataFrame, rungs, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                             line=dict(color="gray", dash="dash")))
    fig.add_shape(type="rect", x0=0.89, x1=0.91, y0=style.BAND[0], y1=style.BAND[1],
                  fillcolor=style.BAND_FILL, line_width=0)
    for r in rungs:
        g = df.filter(pl.col("rung") == r).sort("level")
        if g.height:
            fig.add_trace(go.Scatter(x=g["level"].to_list(), y=g["coverage"].to_list(),
                                     mode="lines+markers", name=r,
                                     line=dict(color=style.RUNG[r], width=2.5)))
    fig.update_xaxes(title="nominal interval level", range=[0, 1])
    fig.update_yaxes(title="share of held-out laps inside", range=[0, 1])
    fig.update_layout(title=title)
    return style.tidy(fig, height=420)


if data.need(cal, "l4", "real-race calibration curves"):
    a.plotly_chart(cal_fig(cal, REAL, "Real races (L2–L5)"))
if twin_cal is not None:
    b.plotly_chart(cal_fig(twin_cal, ["L0", "L1"], "Synthetic twin (L0–L1)"))
else:
    with b:
        data.need(None, "ladder", "twin calibration")

# --------------------------------------------------------------------------
# posteriors
# --------------------------------------------------------------------------
st.subheader("What each rung believes")
post = data.artifact("l4", "posterior")
if data.need(post, "l4", "posterior draws"):
    PARAMS = {
        "fuel_pace": ("Fuel pace (s/lap)", style.PHYSICS_FUEL,
                      "Shaded: what fuel physics implies. The gap is the project's "
                      "biggest open question."),
        "tau_scale": ("Race / quali track-evolution scale s", None,
                      "s < 1: the race track evolves less than in qualifying."),
        "wake_s": ("Dirty-air cost in full wake (s/lap)", None,
                   "Only rungs L3+ model wake."),
        "nu": ("Student-t degrees of freedom ν", None,
               "Low ν = heavy tails. Gaussian is ν → ∞."),
        "sd_stint": ("Per-stint offset SD (s)", None,
                     "How much one set of tyres / fuel load differs from the next."),
    }
    names = [p for p in PARAMS if p in post.columns]
    cols = st.columns(2)
    for i, p in enumerate(names):
        title, band, note = PARAMS[p]
        fig = go.Figure()
        if band:
            fig.add_vrect(x0=band[0], x1=band[1], fillcolor=style.WARN_FILL,
                          line_width=0, annotation_text="physics")
        for r in REAL:
            v = post.filter(pl.col("rung") == r)[p].drop_nulls()
            if v.len():
                fig.add_trace(go.Histogram(x=v.to_list(), name=r, opacity=0.55,
                                           histnorm="probability density", nbinsx=50,
                                           marker_color=style.RUNG[r]))
        fig.update_layout(barmode="overlay", title=title)
        fig.update_yaxes(showticklabels=False)
        with cols[i % 2]:
            st.plotly_chart(style.tidy(fig, height=320))
            st.caption(note)
