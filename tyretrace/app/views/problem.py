"""The problem, interactively: why a stint cannot split fuel from wear, and
why a pit stop can. Pure numpy, no data -- this page is the argument."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from tyretrace.app import style

st.title("❓ The problem")
st.markdown(
    "Over a stint, two things change lap time at once. The car gets **lighter** "
    "as fuel burns, which makes it faster, and the tyres **wear**, which makes it "
    "slower. Both move by the same amount every lap, so a stint's lap times only "
    "show the *difference* between the two effects.")

# --------------------------------------------------------------------------
# 1. one stint, many truths
# --------------------------------------------------------------------------
st.subheader("1 · One stint, infinitely many explanations")
a, b = st.columns([1, 2])
with a:
    net = st.slider("Observed slope in the stint (s/lap)", -0.05, 0.08, 0.01, 0.005,
                    help="What a regression of lap time on lap-in-stint returns.")
    fuel = st.slider("Your assumed fuel effect (s/lap)", 0.00, 0.10, 0.055, 0.005,
                     help="The coefficient everyone imports and nobody verifies.")
    deg = net + fuel
    st.metric("⇒ implied tyre degradation", f"{deg:+.3f} s/lap")
    st.caption("Move the fuel slider. The data doesn't change, but the "
               "'degradation' you'd report moves one-for-one with your assumption.")

k = np.arange(0, 21)
fig = go.Figure()
for f, alpha in ((0.03, 0.35), (0.055, 0.6), (0.08, 0.35)):
    d = net + f
    fig.add_trace(go.Scatter(x=k, y=d * k, mode="lines",
                             line=dict(color=f"rgba(225,6,0,{alpha})", width=2),
                             name=f"tyre wear {d:+.3f}", legendgroup="t"))
    fig.add_trace(go.Scatter(x=k, y=-f * k, mode="lines",
                             line=dict(color=f"rgba(31,111,209,{alpha})", width=2),
                             name=f"fuel burn {-f:+.3f}", legendgroup="f"))
fig.add_trace(go.Scatter(x=k, y=net * k, mode="lines",
                         line=dict(color="black", width=4), name="what you observe"))
fig.update_xaxes(title="lap in stint")
fig.update_yaxes(title="Δ lap time (s)")
fig.update_layout(title="Three different truths, identical data")
b.plotly_chart(style.tidy(fig, height=380))

# --------------------------------------------------------------------------
# 2. the pit stop
# --------------------------------------------------------------------------
st.subheader("2 · The pit stop breaks the tie")
st.markdown(
    "At a pit stop the **tyres reset to new but the fuel does not refill**. "
    "Two explanations that matched perfectly before the stop now predict "
    "different jumps, and the real jump shows which one is right.")

a, b = st.columns([1, 2])
with a:
    true_deg = st.slider("True degradation (s/lap)", 0.02, 0.12, 0.06, 0.005)
    pit = st.slider("Pit stop on lap", 10, 40, 22)
    noise = st.slider("Lap-time noise (s)", 0.0, 0.4, 0.15, 0.05)
    seed = st.number_input("Random seed", 0, 999, 7)

n_laps, true_fuel = 50, 0.058
net_obs = true_deg - true_fuel                 # what stint 1 alone shows
alt_deg = true_deg + 0.04                       # a rival story, same stint slope
alt_fuel = alt_deg - net_obs

lap = np.arange(1, n_laps + 1)
age = np.where(lap <= pit, lap, lap - pit)


def pace(d, f):
    return d * age - f * lap


rng = np.random.default_rng(int(seed))
obs = pace(true_deg, true_fuel) + rng.normal(0, noise, n_laps)

# the race design: regress on lap AND age; the age reset identifies both
X = np.column_stack([np.ones(n_laps), lap, age])
coef, *_ = np.linalg.lstsq(X, obs, rcond=None)
resid = obs - X @ coef
s2 = resid @ resid / max(n_laps - 3, 1)
se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * s2)

fig = go.Figure()
fig.add_vline(x=pit + 0.5, line_dash="dot", line_color="gray",
              annotation_text="pit stop: tyre age → 0, fuel unchanged")
fig.add_trace(go.Scatter(x=lap, y=pace(alt_deg, alt_fuel), mode="lines",
                         line=dict(color="#9B59B6", width=3, dash="dash"),
                         name=f"rival story: deg {alt_deg:.3f}, fuel {alt_fuel:.3f}"))
fig.add_trace(go.Scatter(x=lap, y=pace(true_deg, true_fuel), mode="lines",
                         line=dict(color=style.COMPOUND["SOFT"], width=3),
                         name=f"truth: deg {true_deg:.3f}, fuel {true_fuel:.3f}"))
fig.add_trace(go.Scatter(x=lap, y=obs, mode="markers",
                         marker=dict(color="black", size=6), name="observed laps"))
fig.update_xaxes(title="race lap")
fig.update_yaxes(title="Δ lap time (s)")
fig.update_layout(title="Identical until the stop. Then they split.")
b.plotly_chart(style.tidy(fig, height=420))

with a:
    st.metric("Recovered degradation", f"{coef[2]:.3f} ± {2*se[2]:.3f}",
              delta=f"{coef[2]-true_deg:+.3f} vs truth", delta_color="off")
    st.metric("Recovered fuel effect", f"{-coef[1]:.3f} ± {2*se[1]:.3f}",
              delta=f"{-coef[1]-true_fuel:+.3f} vs truth", delta_color="off")

st.info(
    "**The catch.** A race lap also gets faster as the track rubbers in, and that "
    "moves with the same lap index as fuel burn. Tyre Trace measures track "
    "evolution separately in **qualifying**, where every car runs minimum fuel. "
    "See *Track evolution*.")
