import sys
from pathlib import Path

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from glencore_multi_agent.data import load_glencore, load_dividends

st.set_page_config(page_title="Overview", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@600;700&family=Inter:wght@400;500&display=swap');
html, body, [class*="st-"] { font-family: serif !important; }
h1, h2, h3 { font-family: serif !important; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Overview")
st.caption("Live price data, return history, and distribution — GLEN.L (LSE, priced in pence)")

@st.cache_data(ttl=3600)
def get_data():
    return load_glencore(), load_dividends()

glen, divs = get_data()
r = glen["log_return"].dropna()

current_p = float(glen["adj_close"].iloc[-1])
prev_p    = float(glen["adj_close"].iloc[-2])
daily_chg = (current_p - prev_p) / prev_p * 100
ann_vol   = r.std() * np.sqrt(252) * 100
ytd_ret   = float(
    glen[glen.index.year == glen.index[-1].year]["log_return"].sum() * 100
)
as_of = str(glen.index[-1].date())

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Price (GBX)",  f"{current_p:.1f}p",       f"{daily_chg:+.2f}%")
col2.metric("Price (GBP)",  f"£{current_p/100:.3f}")
col3.metric("YTD Return",   f"{ytd_ret:+.1f}%")
col4.metric("Ann. Vol",     f"{ann_vol:.1f}%")
col5.metric("As of",        as_of)

st.divider()

st.subheader("Adjusted Close Price (GBX)")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=glen.index, y=glen["adj_close"],
    name="Adj Close",
    line=dict(color="#4ade9e", width=1.5),
    hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}p<extra></extra>",
))
div_dates  = [d for d in divs.index if d in glen.index]
div_prices = [float(glen.loc[d, "adj_close"]) for d in div_dates]
fig.add_trace(go.Scatter(
    x=div_dates, y=div_prices, mode="markers",
    name="Ex-dividend",
    marker=dict(color="#f97316", size=7, symbol="diamond"),
    hovertemplate="Ex-div: %{x|%Y-%m-%d}<extra></extra>",
))
fig.update_layout(
    height=380, template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", y=1.02),
    margin=dict(l=0, r=0, t=30, b=0),
)
st.plotly_chart(fig, use_container_width=True)
st.caption("Orange diamonds = ex-dividend dates. The visual dips in the raw price chart that originally prompted this project — they're dividends, not a seasonal cycle.")

st.divider()

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Daily Log Returns")
    fig2 = go.Figure(go.Scatter(
        x=r.index, y=r.values * 100,
        line=dict(color="#6c8cff", width=0.6),
        hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}%<extra></extra>",
    ))
    fig2.add_hline(y=0, line_color="white", opacity=0.2)
    fig2.update_layout(
        height=260, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Return (%)",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.subheader("Return Distribution")
    fig3 = go.Figure(go.Histogram(
        x=r.values * 100, nbinsx=100,
        marker_color="#6c8cff", opacity=0.8,
        hovertemplate="%{x:.2f}%: %{y} days<extra></extra>",
    ))
    fig3.update_layout(
        height=260, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Return (%)", yaxis_title="Days",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig3, use_container_width=True)

st.caption(f"Excess kurtosis: {r.kurtosis():.2f} (normal = 0). Fat tails confirm GARCH is the right volatility model.")