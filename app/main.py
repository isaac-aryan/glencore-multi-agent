import sys
import streamlit as st
sys.path.insert(0, "src")  # run from project root

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from glencore_multi_agent.data import load_glencore, load_dividends

st.set_page_config(
    page_title="Glencore Quant Research",
    page_icon="⛏",
    layout="wide",
)

st.title("⛏ Glencore Quant Research")
st.caption("A time series, volatility and commodity analysis of GLEN.L")

# ── Load data ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)  # cache for 1 hour — avoid re-downloading on every rerun
def get_data():
    return load_glencore(), load_dividends()

glen, divs = get_data()
r = glen["log_return"].dropna()

# ── Key metrics row ──────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
current_p = float(glen["adj_close"].iloc[-1])
prev_p    = float(glen["adj_close"].iloc[-2])
daily_chg = (current_p - prev_p) / prev_p * 100
ann_ret   = r.mean() * 252 * 100
ann_vol   = r.std() * np.sqrt(252) * 100
ytd_ret   = float(
    glen[glen.index.year == glen.index[-1].year]["log_return"].sum() * 100
)

col1.metric("Price (GBX)",    f"{current_p:.1f}p", f"{daily_chg:+.2f}%")
col2.metric("Price (GBP)",    f"£{current_p/100:.3f}")
col3.metric("YTD Return",     f"{ytd_ret:+.1f}%")
col4.metric("Ann. Vol",       f"{ann_vol:.1f}%")
col5.metric("Data since",     f"{glen.index[0].date()}")

st.divider()

# ── Price chart ──────────────────────────────────────────────────────────────
st.subheader("Adjusted Close Price (GBX)")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=glen.index, y=glen["adj_close"],
    name="Adj Close", line=dict(color="#4ade9e", width=1.5),
    hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}p<extra></extra>",
))

# Mark ex-dividend dates
div_dates = [d for d in divs.index if d in glen.index]
div_prices = [float(glen.loc[d, "adj_close"]) for d in div_dates]
fig.add_trace(go.Scatter(
    x=div_dates, y=div_prices, mode="markers",
    name="Ex-dividend", marker=dict(color="#f97316", size=6, symbol="diamond"),
    hovertemplate="Ex-div: %{x|%Y-%m-%d}<extra></extra>",
))

fig.update_layout(
    height=380, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.02),
    margin=dict(l=0, r=0, t=30, b=0),
)
st.plotly_chart(fig, use_container_width=True)
st.caption("Orange diamonds = ex-dividend dates. Adjusted prices remove dividend drops.")

# ── Return distribution ──────────────────────────────────────────────────────
st.subheader("Daily Log Return Distribution")
fig2 = go.Figure()
fig2.add_trace(go.Histogram(
    x=r * 100, nbinsx=100, name="Returns",
    marker_color="#6c8cff", opacity=0.75,
    hovertemplate="%{x:.2f}%: %{y} days<extra></extra>",
))
fig2.update_layout(
    height=280, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Return (%)",
    margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig2, use_container_width=True)
st.caption(
    f"Kurtosis: {r.kurtosis():.2f} (normal=0). Fat tails justify GARCH volatility modelling."
)