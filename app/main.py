import sys
from pathlib import Path

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

import streamlit as st

st.set_page_config(
    page_title="Glencore Quantitative Research Dashboard",
    page_icon="🪨",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;500;600;700&display=swap');

.stMarkdown p,
.stMarkdown li,
.stCaption,
[data-testid="stMarkdownContainer"] {
    font-family: 'Lora', serif !important;
}


h1, h2, h3, h4 {
    font-family: 'Lora', serif !important;
    font-weight: 700 !important;
}

.story-block {
    background: #171b26;
    border: 1px solid #2a2f45;
    border-left: 4px solid #4ade9e;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 14px;
    color: #c8d0dc;
    line-height: 1.8;
    font-family: 'Lora', serif;
}
.story-block strong { color: #e2e8f0; }

.finding-chip {
    display: inline-block;
    background: rgba(74,222,158,0.08);
    border: 1px solid rgba(74,222,158,0.25);
    border-radius: 6px;
    padding: 5px 13px;
    font-size: 13px;
    color: #4ade9e;
    margin: 4px 4px 4px 0;
    font-family: 'Lora', serif;
}

.stage-card {
    background: #171b26;
    border: 1px solid #2a2f45;
    border-radius: 8px;
    padding: 16px 18px;
    margin-bottom: 10px;
    height: 100%;
}
.stage-num {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4ade9e;
    margin-bottom: 5px;
    font-family: 'Lora', serif;
}
.stage-title {
    color: #e2e8f0;
    font-weight: 600;
    font-size: 15px;
    margin-bottom: 5px;
    font-family: 'Lora', serif;
}
.stage-desc {
    color: #8892a4;
    font-size: 13px;
    line-height: 1.55;
    font-family: 'Lora', serif;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("Glencore Quantitative Research Dashboard")
st.markdown("**GLEN.L** | Time series, volatility, and commodity analysis: self-directed learning project")

col_gh, _ = st.columns([1, 6])
col_gh.link_button("GitHub", "https://github.com/isaac-aryan/glencore-multi-agent/")

st.divider()

# Motivation
st.subheader("Background")

st.markdown("""
<div class="story-block">
I hold Glencore in my portfolio and noticed what appeared to be a recurring annual cycle
in the price chart on Trading212. The pattern looked consistent enough that it seemed worth
investigating rigorously rather than treating it as obvious. My initial assumption was that
commodity demand seasonality was bleeding through into the equity price, but I wanted to
test that properly before drawing any conclusions.
</div>

<div class="story-block">
The project grew from that single question into a full quantitative research workflow:
stationarity testing, spectral analysis, GARCH volatility modelling, multivariate cointegration
with the underlying commodity basket, and a walk-forward validated ML forecasting pipeline.
The subject matter reflects a stock I genuinely follow, which kept the analysis grounded.
The goal throughout was methodological rigour rather than finding a trading strategy.
</div>

<div class="story-block">
As an extension, the analysis pipeline was wrapped into a multi-agent system using the
OpenAI Agents SDK and a custom MCP server, so the statistical tools can be queried
conversationally via the Research Agent page. This dashboard is the final output.
</div>
""", unsafe_allow_html=True)

st.divider()

# Findings
st.subheader("Key findings")

st.markdown("""
<span class="finding-chip">Stage 1 — No significant annual seasonality (F-test p=0.71)</span>
<span class="finding-chip">Stage 2 — Leverage effect confirmed in GJR-GARCH (gamma p=0.008)</span>
<span class="finding-chip">Stage 3 — Copper Granger-causes Glencore (p=0.006)</span>
<span class="finding-chip">Stage 3 — No cointegration with copper (EG p=0.27)</span>
""", unsafe_allow_html=True)

st.caption("The apparent annual cycle was largely explained by semi-annual dividend drops on the raw price chart, which disappear in adjusted returns. A negative result, honestly arrived at.")

st.divider()

# Stages
st.subheader("Project structure")

stages = [
    ("Stage 0", "Data Foundation",
     "Download-once caching, adjusted price handling, commodity data sourcing, calendar alignment."),
    ("Stage 1", "Seasonality",
     "ADF/KPSS tests, STL decomposition, periodogram, SARIMA, calendar F-test."),
    ("Stage 2", "Volatility",
     "ARCH-LM test, GARCH(1,1), GJR-GARCH, Student-t innovations, VaR."),
    ("Stage 3", "Commodities",
     "VAR, impulse response, Granger causality, Engle-Granger and Johansen cointegration."),
    ("Stage 4", "ML Forecasting",
     "Walk-forward CV, XGBoost, baseline comparison, cost-adjusted backtest."),
    ("Stage 5", "Agentic System",
     "Multi-agent pipeline via OpenAI Agents SDK and custom FastMCP server."),
]

col1, col2, col3 = st.columns(3)
cols = [col1, col2, col3]
for i, (num, title, desc) in enumerate(stages):
    with cols[i % 3]:
        st.markdown(f"""
<div class="stage-card">
  <div class="stage-num">{num}</div>
  <div class="stage-title">{title}</div>
  <div class="stage-desc">{desc}</div>
</div>
""", unsafe_allow_html=True)

st.divider()
st.caption("Python · yfinance · statsmodels · arch · scikit-learn · XGBoost · OpenAI Agents SDK · FastMCP · Streamlit · Plotly")