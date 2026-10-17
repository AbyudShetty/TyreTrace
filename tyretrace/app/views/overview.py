"""Overview -- headline numbers, the gate board, and where the ladder stands."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from tyretrace import pipeline
from tyretrace.app import data, style
from tyretrace.models import track_evolution as te

st.title("🛞 Tyre Trace")
st.markdown(
    "**Recovering true tyre degradation from confounded lap times.** Inside a "
    "stint, fuel burn and tyre wear both move with lap number, so a lap-time "
    "slope only shows their *sum*. Tyre Trace separates them, using qualifying "
    "for track evolution and pit stops for degradation, and states how sure it is.")

# --------------------------------------------------------------------------
# headline numbers
# --------------------------------------------------------------------------
st.subheader("What the data says")
c1, c2, c3, c4 = st.columns(4)

quali = data.laps("quali")
if quali is not None:
    s = te.summary(te.fit(quali))
    c1.metric("Track evolution (quali)", f"{s['median_tau']:+.4f} s/min",
              help=f"Median over {s['n']} qualifying sessions; negative = track "
                   f"getting faster. Negative in {100*s['frac_negative']:.0f}% of "
                   f"sessions, median z = {s['median_z']:.1f}.")
else:
    c1.metric("Track evolution (quali)", "—")

fp = data.metric("l2", "fuel_pace")
if fp is not None:
    lo, hi = data.metric("l2", "fuel_pace_q05"), data.metric("l2", "fuel_pace_q95")
    c2.metric("Fuel pace", f"{fp:.4f} s/lap",
              help=f"Time lost per lap of fuel carried, identified by L2. "
                   f"90% interval [{lo:.4f}, {hi:.4f}]. Physics implies "
                   f"{style.PHYSICS_FUEL[0]}–{style.PHYSICS_FUEL[1]}: still unexplained.")
else:
    c2.metric("Fuel pace", "—")

ts = data.metric("l2", "tau_scale")
c3.metric("Race / quali evolution", f"{ts:.2f}×" if ts is not None else "—",
          help="How much of qualifying's track evolution carries into the race "
               "(s in L2). Below 1: the race track is already rubbered in.")

wake = data.metric("l3", "wake_s_per_lap")
c4.metric("Dirty-air cost", f"{wake:.2f} s/lap" if wake is not None else "—",
          help="Lap time lost in full wake (L3), from gap-to-car-ahead in timing data.")

# --------------------------------------------------------------------------
# ladder at a glance
# --------------------------------------------------------------------------
st.subheader("Model ladder on real races")
st.caption("Each rung ships only if it beats the one below on held-out CRPS *and* "
           "keeps 90% coverage inside the green band (protocol §5). "
           "Lower CRPS is better.")

rungs = ["L2", "L3", "L4", "L5"]
crps = [data.metric("l4", f"crps_{r.lower()}") for r in rungs]
cov = [data.metric("l4", f"cov90_{r.lower()}") for r in rungs]

# protocol section 4 baselines, scored on the same held-out laps
BASE = {"B1": "naive", "B2": "conventional", "B3": "boosting"}
base = [(f"{k} {v}", data.metric("baselines", f"crps_{k.lower()}"),
         data.metric("baselines", f"cov90_{k.lower()}")) for k, v in BASE.items()]
base = [b for b in base if b[1] is not None]

if all(v is not None for v in crps + cov):
    a, b = st.columns(2)
    names = rungs + [n for n, _, _ in base]
    allc = crps + [c for _, c, _ in base]
    allv = cov + [v for _, _, v in base]
    colors = [style.RUNG[r] for r in rungs] + [style.BASELINE] * len(base)

    fig = go.Figure(go.Bar(x=names, y=allc, marker_color=colors,
                           text=[f"{v:.3f}" for v in allc], textposition="outside"))
    fig.update_yaxes(title="held-out CRPS (s)", range=[0, max(allc) * 1.2])
    fig.update_layout(title="Accuracy of the whole forecast (grey = baselines)")
    a.plotly_chart(style.tidy(fig, height=340, legend=False))

    fig = go.Figure()
    fig.add_hrect(y0=style.BAND[0], y1=style.BAND[1], fillcolor=style.BAND_FILL,
                  line_width=0, annotation_text="ship band",
                  annotation_position="top left")
    fig.add_trace(go.Scatter(x=names, y=allv, mode="markers+text",
                             marker=dict(size=16, color=colors),
                             text=[f"{v:.3f}" for v in allv], textposition="middle right"))
    fig.update_yaxes(title="90% interval coverage", range=[0.3, 1.0])
    fig.update_layout(title="Are the 90% intervals honest?")
    b.plotly_chart(style.tidy(fig, height=340, legend=False))

    best = rungs[int(np.argmin(crps))]
    st.markdown(
        f"**Verdict:** {best} has the best CRPS, but no rung's intervals are "
        "wide enough yet, so nothing ships. Too-narrow intervals are the open "
        "problem. See *Held-out predictions* for where they miss.")
    if base:
        st.caption(
            "Baselines: forecasting the rest of the *same* race only needs the fused "
            "slope, so the conventional fuel-correction (an assumed 0.030 s/kg) "
            "forecasts nearly as well. It reports the wrong *degradation* when its fuel "
            "assumption is wrong, and this test can't see that. The FP → race backtest can.")
else:
    data.need(None, "l4", "the real-race ladder")

twin = {r: (data.metric("ladder", f"twin_crps_{r.lower()}"),
            data.metric("ladder", f"twin_cov90_{r.lower()}")) for r in ("L0", "L1")}
if all(v[0] is not None for v in twin.values()):
    st.caption(
        "On the synthetic twin, where truth is known: "
        + " · ".join(f"{r} CRPS {c:.3f}, coverage {v:.3f}" for r, (c, v) in twin.items())
        + ". Not comparable with the real-race numbers above: different data.")

# --------------------------------------------------------------------------
# gate board
# --------------------------------------------------------------------------
st.subheader("Gate board")
st.caption("Every audit prints a gate. Some are *expected* to fail or hold. "
           "Those are recorded findings, not breakage.")

rows = []
for name, t in pipeline.TASKS.items():
    e = data.result(name)
    gate = e.get("gate", "-")
    if gate == "-":
        verdict = "not run"
    elif gate == t.expect:
        verdict = "as expected" + (f": {t.expect_why}" if t.expect != "PASS" else "")
    else:
        verdict = f"UNEXPECTED (wanted {t.expect})"
    rows.append({"": style.GATE.get(gate, "⚪"), "task": name, "gate": gate,
                 "what it checks": t.summary, "verdict": verdict,
                 "last run": e.get("recorded_at", "")})
st.dataframe(pl.DataFrame(rows).to_pandas(), hide_index=True)
st.caption("Refresh with `python -m tyretrace run all`.")
