"""
mcp_server.py — FastMCP server exposing glencore_quant analysis tools.

Run directly:
    python -m glencore_quant.mcp_server

The OpenAI Agents SDK will spawn this as a subprocess (stdio transport)
and discover available tools automatically via list_tools().
"""

from __future__ import annotations
import json
from typing import Literal

import numpy as np
import pandas as pd
from fastmcp import FastMCP

# Import your existing Stage 0 data module
from glencore_multi_agent.data import load_glencore, load_aligned_panel, load_dividends

# ── Create the server ──────────────────────────────────────────────────────
mcp = FastMCP(
    "glencore-research",
    instructions="""Tools for Glencore (GLEN.L) quantitative research.
    Prices are in GBX (pence) unless specified. Use adjusted prices for
    return analysis. All statistical tests return structured dicts.""",
)


# ════════════════════════════════════════════════════════════════════════════
# DATA TOOLS
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool
def get_glencore_data(period: Literal["1y", "2y", "5y", "full"] = "1y") -> dict:
    """
    Load GLEN.L price data (adjusted close) with basic summary statistics.

    Args:
        period: How much history to return.
                '1y' = last 12 months, '2y' = 2 years, '5y' = 5 years,
                'full' = since IPO (May 2011).

    Returns a dict with: date_range, num_rows, price_summary (min/max/current),
    annualised_return_pct, annualised_vol_pct, and the last 5 rows of data.
    """
    df = load_glencore()

    if period != "full":
        years = {"1y": 1, "2y": 2, "5y": 5}[period]
        cutoff = df.index[-1] - pd.DateOffset(years=years)
        df = df[df.index >= cutoff]

    r = df["log_return"].dropna()
    return {
        "period": period,
        "date_range": {
            "start": str(df.index[0].date()),
            "end": str(df.index[-1].date()),
        },
        "num_rows": len(df),
        "price_summary_gbx": {
            "current": round(float(df["adj_close"].iloc[-1]), 2),
            "min": round(float(df["adj_close"].min()), 2),
            "max": round(float(df["adj_close"].max()), 2),
        },
        "annualised_return_pct": round(float(r.mean() * 252 * 100), 2),
        "annualised_vol_pct": round(float(r.std() * np.sqrt(252) * 100), 2),
        "skewness": round(float(r.skew()), 3),
        "excess_kurtosis": round(float(r.kurtosis()), 3),
    }


@mcp.tool
def get_dividend_history() -> dict:
    """
    Return Glencore's full dividend history with ex-dates and amounts (in GBX).
    Useful for understanding whether price drops align with ex-dividend dates,
    which is the primary alternative explanation for any apparent annual cycle.
    """
    divs = load_dividends()
    return {
        "num_dividends": len(divs),
        "dividends": [
            {"ex_date": str(d.date()), "amount_gbx": round(float(v), 4)}
            for d, v in divs.items()
        ],
    }


@mcp.tool
def get_rolling_statistics(window_days: int = 30, period: str = "1y") -> dict:
    """
    Compute rolling mean return and volatility for Glencore.

    Args:
        window_days: Rolling window size in trading days (e.g. 30, 60, 252).
        period: How much history to include ('1y', '2y', '5y', 'full').

    Returns the last 20 observations of rolling stats, plus the current
    vol regime (low/medium/high) relative to full history.
    """
    df = load_glencore()
    if period != "full":
        years = {"1y": 1, "2y": 2, "5y": 5}.get(period, 1)
        df = df[df.index >= df.index[-1] - pd.DateOffset(years=years)]

    r = df["log_return"].dropna()
    roll_vol = r.rolling(window_days).std() * np.sqrt(252) * 100
    current_vol = float(roll_vol.iloc[-1])
    hist_75 = float(roll_vol.quantile(0.75))
    hist_25 = float(roll_vol.quantile(0.25))

    return {
        "window_days": window_days,
        "current_annualised_vol_pct": round(current_vol, 2),
        "vol_regime": (
            "high" if current_vol > hist_75
            else "low" if current_vol < hist_25
            else "medium"
        ),
        "recent_observations": [
            {"date": str(d.date()), "ann_vol_pct": round(float(v), 2)}
            for d, v in roll_vol.dropna().tail(10).items()
        ],
    }


# ════════════════════════════════════════════════════════════════════════════
# STATISTICAL ANALYSIS TOOLS
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool
def run_stationarity_tests(period: str = "full") -> dict:
    """
    Run ADF and KPSS stationarity tests on Glencore log returns.

    ADF null hypothesis: series HAS a unit root (non-stationary).
        p < 0.05 → reject null → series IS stationary.
    KPSS null hypothesis: series IS stationary.
        p < 0.05 → reject null → series is NOT stationary.

    Both tests together give a clearer picture than either alone.
    Returns test statistics, p-values, and a plain-English interpretation.
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    df = load_glencore()
    r = df["log_return"].dropna()

    adf_stat, adf_p, _, _, adf_crit, _ = adfuller(r, autolag="AIC")
    kpss_stat, kpss_p, _, kpss_crit = kpss(r, regression="c")

    adf_stationary = adf_p < 0.05
    kpss_stationary = kpss_p >= 0.05

    if adf_stationary and kpss_stationary:
        verdict = "STATIONARY — both tests agree the series is stationary. Good for modelling."
    elif not adf_stationary and not kpss_stationary:
        verdict = "NON-STATIONARY — both tests agree. Do not model the raw series."
    else:
        verdict = "INCONCLUSIVE — tests disagree. May indicate structural breaks."

    return {
        "series": "GLEN.L log returns",
        "n_observations": len(r),
        "adf_test": {
            "statistic": round(adf_stat, 4),
            "p_value": round(adf_p, 6),
            "critical_values": {k: round(v, 4) for k, v in adf_crit.items()},
            "conclusion": "stationary" if adf_stationary else "non-stationary",
        },
        "kpss_test": {
            "statistic": round(kpss_stat, 4),
            "p_value": round(kpss_p, 6),
            "conclusion": "stationary" if kpss_stationary else "non-stationary",
        },
        "verdict": verdict,
    }


@mcp.tool
def run_seasonality_analysis() -> dict:
    """
    Test for annual seasonality in Glencore returns.

    Runs three complementary approaches:
    1. Month-of-year average returns (are any months consistently better/worse?)
    2. Autocorrelation at seasonal lags (lag 21 = monthly, lag 63 = quarterly, lag 252 = annual)
    3. A simple calendar-effect F-test (do monthly dummies jointly explain returns?)

    Returns structured results for each test plus a combined interpretation.
    This directly tests the 'annual cycle' hypothesis from chart observation.
    """
    from statsmodels.tsa.stattools import acf
    from scipy import stats

    df = load_glencore()
    r = df["log_return"].dropna().copy()
    r.index = pd.DatetimeIndex(r.index)

    # 1. Monthly average returns
    month_returns = r.groupby(r.index.month).mean() * 100
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    # 2. ACF at key seasonal lags
    acf_values = acf(r, nlags=260, fft=True)
    seasonal_lags = {
        "monthly_lag21": round(float(acf_values[21]), 4),
        "quarterly_lag63": round(float(acf_values[63]), 4),
        "semiannual_lag126": round(float(acf_values[126]), 4),
        "annual_lag252": round(float(acf_values[252]), 4),
    }
    ci_95 = 1.96 / np.sqrt(len(r))  # approx 95% CI bound

    # 3. F-test for joint significance of month dummies
    months = r.index.month
    groups = [r[months == m].values for m in range(1, 13)]
    f_stat, f_p = stats.f_oneway(*groups)

    return {
        "monthly_avg_daily_returns_pct": {  # was: monthly_avg_returns_pct
            month_names[i-1]: round(float(month_returns.get(i, 0)), 4)
            for i in range(1, 13)
        },
        "acf_at_seasonal_lags": seasonal_lags,
        "acf_95pct_ci_bound": round(float(ci_95), 4),
        "calendar_f_test": {
            "f_statistic": round(float(f_stat), 4),
            "p_value": round(float(f_p), 6),
            "significant_at_5pct": bool(f_p < 0.05),
            "interpretation": (
                "Month-of-year effects are jointly statistically significant."
                if f_p < 0.05
                else "No significant calendar effect detected across months."
            ),
        },
        "seasonal_lags_significant": {
            k: bool(abs(v) > ci_95) for k, v in seasonal_lags.items()
        },
        "units_note": "Values are average daily log returns in percent (e.g. 0.10 means 0.10% per day, not 10%)",
    }

@mcp.tool
def run_arch_test(period: str = "full") -> dict:
    """
    Run Engle's ARCH-LM test to check whether GARCH modelling is warranted.

    Tests the null hypothesis of constant variance (no ARCH effects).
    A significant result (p < 0.05) confirms volatility clustering is present
    and GARCH modelling is statistically justified.

    Args:
        period: Data period to test ('1y', '2y', '5y', 'full').

    Returns test statistic, p-value, and a plain-English verdict.
    """
    from statsmodels.stats.diagnostic import het_arch

    df = load_glencore()
    if period != "full":
        years = {"1y": 1, "2y": 2, "5y": 5}.get(period, 5)
        df = df[df.index >= df.index[-1] - pd.DateOffset(years=years)]

    r = df["log_return"].dropna() * 100
    lm_stat, lm_p, f_stat, f_p = het_arch(r, nlags=10)

    return {
        "test": "Engle ARCH-LM",
        "null_hypothesis": "No ARCH effects (constant variance)",
        "period": period,
        "n_observations": len(r),
        "lm_statistic": round(float(lm_stat), 4),
        "lm_p_value": round(float(lm_p), 8),
        "f_statistic": round(float(f_stat), 4),
        "f_p_value": round(float(f_p), 8),
        "arch_effects_present": bool(lm_p < 0.05),
        "verdict": (
            "ARCH effects confirmed — GARCH modelling is statistically justified."
            if lm_p < 0.05
            else
            "No significant ARCH effects — constant variance model may suffice."
        ),
    }

@mcp.tool
def fit_garch(
    p: int = 1,
    q: int = 1,
    asymmetric: bool = False,
    distribution: Literal["normal", "t"] = "normal",
    period: str = "full",
) -> dict:
    """
    Fit a GARCH volatility model to Glencore log returns.

    Args:
        p: ARCH order. Default 1.
        q: GARCH order. Default 1.
        asymmetric: If True, fits GJR-GARCH which captures the leverage effect
                    (negative shocks increasing vol more than positive ones).
        distribution: Innovation distribution. 'normal' or 't' (Student-t,
                      better for fat tails). Default 'normal'.
        period: Data period ('1y', '2y', '5y', 'full').

    Returns parameters, p-values, persistence, AIC, current conditional vol,
    5-day forecast, and leverage effect results if asymmetric=True.
    """
    from arch import arch_model

    df = load_glencore()
    if period != "full":
        years = {"1y": 1, "2y": 2, "5y": 5}.get(period, 5)
        df = df[df.index >= df.index[-1] - pd.DateOffset(years=years)]

    r = df["log_return"].dropna() * 100
    o = 1 if asymmetric else 0
    model = arch_model(r, vol="Garch", p=p, o=o, q=q, dist=distribution)
    fit = model.fit(disp="off")
    params = fit.params
    pvals  = fit.pvalues

    alpha = float(params.get("alpha[1]", 0))
    beta  = float(params.get("beta[1]", 0))
    gamma = float(params.get("gamma[1]", 0))
    persistence = alpha + beta + 0.5 * gamma if asymmetric else alpha + beta

    forecast = fit.forecast(horizon=5, reindex=False)
    fcast_vol = np.sqrt(forecast.variance.iloc[-1].values) * np.sqrt(252)

    result = {
        "model": f"{'GJR-' if asymmetric else ''}GARCH({p},{q})-{distribution}",
        "n_observations": len(r),
        "parameters": {
            "omega": round(float(params["omega"]), 6),
            "alpha": round(alpha, 4),
            "beta":  round(beta, 4),
        },
        "p_values": {
            "omega": round(float(pvals["omega"]), 4),
            "alpha": round(float(pvals.get("alpha[1]", 1)), 4),
            "beta":  round(float(pvals.get("beta[1]", 1)), 4),
        },
        "persistence": round(persistence, 4),
        "persistence_interpretation": (
            "High — volatility shocks decay slowly (weeks)"
            if persistence > 0.95
            else "Moderate — shocks decay within days"
        ),
        "aic": round(float(fit.aic), 2),
        "bic": round(float(fit.bic), 2),
        "current_conditional_vol_ann_pct": round(
            float(fit.conditional_volatility.iloc[-1]) * np.sqrt(252), 2),
        "vol_forecast_5day_ann_pct": [round(float(v), 2) for v in fcast_vol],
    }

    if asymmetric:
        result["leverage_effect"] = {
            "gamma": round(gamma, 4),
            "gamma_p_value": round(float(pvals.get("gamma[1]", 1)), 4),
            "significant": bool(float(pvals.get("gamma[1]", 1)) < 0.05),
            "interpretation": (
                "Leverage effect confirmed: negative shocks increase vol more than positive"
                if gamma > 0 and float(pvals.get("gamma[1]", 1)) < 0.05
                else "Leverage effect not statistically significant"
            ),
        }

    if distribution == "t":
        result["student_t_df"] = round(float(params.get("nu", 30)), 2)

    return result

if __name__ == "__main__":
    mcp.run(transport="stdio")