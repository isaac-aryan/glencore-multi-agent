# Glencore Quant Research

A self-directed quantitative research project analysing Glencore (GLEN.L) using classical time series econometrics, volatility modelling, multivariate commodity analysis, and machine learning — with a multi-agent AI research system built on top.

**Live app:** [glencore-multi-agent.streamlit.app](https://glencore-multi-agent.streamlit.app)

---

## Background

I hold Glencore in my personal portfolio and noticed what appeared to be a recurring annual cycle in the price chart. Rather than assuming seasonality, I decided to test it properly using the statistical tools that would give an honest answer. The project grew from that single question into a five-stage research workflow, extended with a multi-agent AI system that makes the analysis pipeline queryable in natural language.

The goal throughout was methodological rigour rather than finding a trading strategy. A clean negative result, honestly arrived at, is treated as a success here.

---

## Findings

| Stage | Question | Finding |
|-------|----------|---------|
| 1 — Seasonality | Is the annual cycle real? | No significant seasonality. ACF at 252-day lag: 0.017 (within 95% CI). Calendar F-test p=0.71. The visual pattern is explained by semi-annual dividend drops on the raw price chart. |
| 2 — Volatility | Is volatility predictable? | GJR-GARCH(1,1) confirms the leverage effect (γ=0.062, p=0.008) and high persistence (α+β+γ/2=0.976). Volatility clustering is strong — shocks decay over ~28 trading days. |
| 3 — Commodities | Is Glencore cointegrated with copper? | Copper Granger-causes Glencore (p=0.006) but not vice versa. No long-run cointegration (EG p=0.27). Glencore is commodity-sensitive but not a commodity proxy. |
| 4 — ML | Can ML beat naive baselines? | Walk-forward validated XGBoost with honest baseline comparison and cost-adjusted backtesting. |

---

## Project structure

```
glencore-multi-agent/
├── src/glencore_multi_agent/
│   ├── data.py              # Download, cache, align — the data foundation
│   ├── mcp_server.py        # FastMCP server exposing analysis as tools
│   ├── features.py
│   ├── models.py
│   └── backtest.py
├── agent/
│   ├── agents.py            # Orchestrator + specialist agent definitions
│   └── runner.py            # CLI entry point
├── notebooks/
│   ├── 00_data_setup.ipynb
│   ├── 01_seasonality.ipynb
│   ├── 02_volatility.ipynb
│   ├── 03_commodities.ipynb
│   └── 04_ml_forecasting.ipynb
├── app/
│   ├── main.py              # Landing page
│   └── pages/
│       ├── 1_Overview.py
│       ├── 2_Volatility.py
│       ├── 3_Commodities.py
│       ├── 4_Research_Agent.py
│       └── 5_Findings.py
├── reports/
│   └── findings.md          # Running research log
├── data/                    # Cached downloads (gitignored)
└── config.yaml              # Tickers, paths — no hardcoding
```

---

## Stack

| Layer | Tools |
|-------|-------|
| Data | yfinance, pandas, pandas-datareader |
| Time series | statsmodels (ARIMA, VAR, cointegration) |
| Volatility | arch (GARCH, GJR-GARCH, EGARCH) |
| ML | scikit-learn, XGBoost, LightGBM |
| Agents | OpenAI Agents SDK, FastMCP |
| Dashboard | Streamlit, Plotly |

---

## Running locally

```bash
git clone https://github.com/isaac-aryan/glencore-multi-agent
cd glencore-multi-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add a `.env` file in the project root:

```
OPENAI_API_KEY=sk-proj-...
```

Run the dashboard:

```bash
streamlit run app/main.py
```

Run the research agent from the terminal:

```bash
python -m agent.runner "What is the current volatility regime?"
```

---

## The agent system

The Research Agent page runs a multi-agent pipeline: an Orchestrator delegates to a DataAgent (live market data), an AnalysisAgent (statistical tests via MCP tools), and an InterpretationAgent (plain-English synthesis). The MCP server wraps the quantitative analysis functions — stationarity tests, seasonality analysis, GARCH fitting, Granger causality, cointegration — as tools that the agents call on demand.

```
User question
    → OrchestratorAgent
        → DataAgent         (get_glencore_data, get_dividend_history, ...)
        → AnalysisAgent     (run_stationarity_tests, fit_garch, run_granger_causality, ...)
        → InterpretationAgent
    → Final research note
```

---

## Methodology notes

- **No random k-fold** — walk-forward expanding-window cross-validation only
- **Baselines first** — ML models compared against always-up, momentum, and mean-reversion baselines before any model is claimed to add value
- **Transaction costs included** in all backtests
- **Negative results documented** — the seasonality and cointegration findings are null results treated as valid conclusions, not failures
- All data cached to `data/raw/` on first download and reloaded from disk on subsequent runs for reproducibility

---

## Disclaimer

This is a research and learning project. Nothing here constitutes financial advice or a trading signal.