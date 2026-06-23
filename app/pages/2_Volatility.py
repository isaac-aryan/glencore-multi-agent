import sys; sys.path.insert(0, "src")
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from arch import arch_model
from glencore_multi_agent.data import load_glencore

st.set_page_config(page_title="Volatility", layout="wide")
st.title("📈 Volatility Modelling")
st.caption("GJR-GARCH(1,1) conditional volatility — the Stage 2 finding")

@st.cache_data(ttl=3600)
def fit_gjr_garch():
    glen = load_glencore()
    r    = glen["log_return"].dropna() * 100
    model = arch_model(r, vol="Garch", p=1, o=1, q=1, dist="normal")
    res   = model.fit(disp="off")
    forecast = res.forecast(horizon=5, reindex=False)
    fcast_vol = np.sqrt(forecast.variance.iloc[-1].values) * np.sqrt(252)
    return glen, r, res, fcast_vol

with st.spinner("Fitting GJR-GARCH model..."):
    glen, r, res, fcast_vol = fit_gjr_garch()

cond_vol = res.conditional_volatility * np.sqrt(252)
params   = res.params
alpha    = float(params["alpha[1]"])
beta     = float(params["beta[1]"])
gamma    = float(params["gamma[1]"])
persist  = alpha + beta + 0.5 * gamma
curr_vol = float(cond_vol.iloc[-1])
hist_75  = float(cond_vol.quantile(0.75))
hist_25  = float(cond_vol.quantile(0.25))
regime   = "🔴 High" if curr_vol > hist_75 else "🟢 Low" if curr_vol < hist_25 else "🟡 Medium"

# Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Vol (ann.)", f"{curr_vol:.1f}%")
c2.metric("Vol Regime", regime)
c3.metric("Persistence (α+β+γ/2)", f"{persist:.4f}")
c4.metric("Leverage Effect (γ)", f"{gamma:.4f}")
st.divider()

# Conditional vol chart
st.subheader("Conditional Volatility — GJR-GARCH(1,1)")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=cond_vol.index, y=cond_vol.values,
    fill="tozeroy", fillcolor="rgba(249,115,22,0.15)",
    line=dict(color="#f97316", width=1.5), name="Cond. Vol",
    hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}%<extra></extra>",
))
lr_vol = float(params["omega"]) / (1 - persist) if persist < 1 else None
if lr_vol:
    lr_ann = np.sqrt(lr_vol * 252)
    fig.add_hline(y=lr_ann, line_dash="dash", line_color="white",
                  opacity=0.4, annotation_text=f"Long-run avg {lr_ann:.1f}%")
fig.update_layout(
    height=340, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Annualised Vol (%)",
    margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig, use_container_width=True)

# 5-day forecast
st.subheader("5-Day Volatility Forecast")
fc_fig = go.Figure(go.Bar(
    x=[f"t+{i+1}" for i in range(5)],
    y=fcast_vol, marker_color="#f97316", opacity=0.8,
    text=[f"{v:.1f}%" for v in fcast_vol], textposition="outside",
))
fc_fig.update_layout(
    height=260, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Ann. Vol (%)",
    margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fc_fig, use_container_width=True)

# Model parameters table
with st.expander("GJR-GARCH Model Parameters"):
    import pandas as pd
    st.dataframe(pd.DataFrame({
        "Parameter": ["ω (omega)", "α (alpha)", "β (beta)", "γ (gamma)"],
        "Value":     [float(params["omega"]), alpha, beta, gamma],
        "p-value":   [
            float(res.pvalues["omega"]),
            float(res.pvalues["alpha[1]"]),
            float(res.pvalues["beta[1]"]),
            float(res.pvalues["gamma[1]"]),
        ],
    }))