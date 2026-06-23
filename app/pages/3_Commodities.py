import sys; sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from glencore_multi_agent.data import load_glencore, load_commodities

st.set_page_config(page_title="Commodities", layout="wide")
st.title("🔗 Commodity Relationships")
st.caption("Correlation, Granger causality, and cointegration — Stage 3 findings")

@st.cache_data(ttl=3600)
def get_data():
    return load_glencore(), load_commodities()

glen, comms = get_data()

# Build return panel
glen_r = glen["log_return"].rename("glen")
comm_cols = [c for c in ["copper_log_ret", "brent_log_ret", "natgas_log_ret"]
             if c in comms.columns]
ret_panel = pd.concat(
    [glen_r] + [comms[c].rename(c.replace("_log_ret", "")) for c in comm_cols],
    axis=1
).dropna()

# Findings callout
st.info("**Stage 3 Findings:** Copper Granger-causes Glencore (p=0.006) but not vice versa. No cointegration detected over the full sample (EG p=0.27). Glencore is commodity-sensitive but not a commodity proxy.")
st.divider()

# Rebased price chart
st.subheader("Cumulative Returns — Rebased to 100 at IPO")
fig = go.Figure()
colors = {"glen": "#4ade9e", "copper": "#f97316", "brent": "#6c8cff", "natgas": "#e879f9"}
for col in ret_panel.columns:
    cumret = (1 + ret_panel[col]).cumprod() * 100
    fig.add_trace(go.Scatter(
        x=cumret.index, y=cumret.values,
        name=col.upper(), line=dict(color=colors.get(col, "#aaa"), width=1.5),
        hovertemplate=f"{col.upper()} %{{x|%Y-%m-%d}}: %{{y:.1f}}<extra></extra>",
    ))
fig.update_layout(
    height=360, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Rebased (100 = start)",
    legend=dict(orientation="h", y=1.02),
    margin=dict(l=0, r=0, t=30, b=0),
)
st.plotly_chart(fig, use_container_width=True)

# Rolling correlation
st.subheader("Rolling 252-day Correlation with Glencore")
fig2 = go.Figure()
for col, color in [("copper", "#f97316"), ("brent", "#6c8cff")]:
    if col in ret_panel.columns:
        rc = ret_panel["glen"].rolling(252).corr(ret_panel[col])
        fig2.add_trace(go.Scatter(
            x=rc.index, y=rc.values, name=f"vs {col.upper()}",
            line=dict(color=color, width=1.5),
        ))
fig2.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
fig2.update_layout(
    height=280, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Correlation",
    legend=dict(orientation="h", y=1.02),
    margin=dict(l=0, r=0, t=30, b=0),
)
st.plotly_chart(fig2, use_container_width=True)

# Correlation matrix
with st.expander("Full Correlation Matrix"):
    st.dataframe(ret_panel.corr().round(3))