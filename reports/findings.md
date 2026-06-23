## Stage 1 Finding — Seasonality (June 2026)

**Verdict: No significant annual seasonal pattern detected.**

- ACF at annual lag (252 days): 0.017, within 95% CI bound of ±0.032 — not significant
- ACF at all seasonal lags (monthly, quarterly, semi-annual, annual): all insignificant
- Calendar F-test p-value: 0.71 — month-of-year has no explanatory power over returns
- Both ADF and KPSS confirm log returns are stationary — suitable for further modelling

**Interpretation:** The visual cycle observed on the Trading212 chart is most likely
explained by semi-annual dividend drops on the raw price chart, which disappear in
adjusted returns. With only ~15 annual cycles since the 2011 IPO, statistical power
to detect a true annual pattern is limited regardless.

## Stage 2 Finding — Volatility Modelling

**Date:** June 2026

### Stylized Facts Confirmed
- ARCH-LM test: LM stat=424.24, p≈0 — ARCH effects confirmed, GARCH justified
- Excess kurtosis confirmed in Stage 0 (fat tails)
- Volatility clustering visible in return series

### Model Comparison
| Model          |    AIC  |
|----------------|---------|
| GARCH(1,1)     | 16905.61|
| GJR-GARCH(1,1) | 16877.85| ← selected
| EGARCH(1,1)    | 16922.54|

**Selected model:** GJR-GARCH(1,1)-normal (AIC lower by 27.76 points)

### Key Parameters
- alpha: 0.0434, beta: 0.9013, gamma: 0.0619
- Persistence: 0.9756 — shock half-life ~28 trading days
- Long-run vol: ~35% annualised

### Leverage Effect
- gamma=0.0619, p=0.0076 — confirmed
- Negative shocks increase variance 2.4x more than equivalent positive shocks
- Consistent with Glencore's behaviour during commodity downturns

### Current Regime (June 2026)
- Conditional vol: 34.95% annualised — moderate regime
- 5-day forecast: 34.70% → 34.99% (slowly mean-reverting upward toward long-run)

## Stage 3 Finding — Commodities & Cointegration

**Date:** June 2026

### Granger Causality
- Copper → Glencore: p=0.0064 ✓ significant
- Glencore → Copper: p=0.1792 ✗ not significant
- **Verdict:** Copper leads Glencore in the short run. Glencore is a
  price-taker — no evidence the trading division gives it information
  advantage over the copper market.

### Cointegration (Engle-Granger, Glencore vs Copper)
- EG stat: -2.53, p=0.267 — not cointegrated at 5%
- Critical value (5%): -3.34 — stat is not sufficiently negative
- **Verdict:** No stable long-run equilibrium between log prices.
  Deviations between Glencore and copper can persist indefinitely.

### Combined interpretation
Copper has short-run predictive power over Glencore returns (Granger),
but the two prices are not tethered over the long run (no cointegration).
Glencore is commodity-sensitive but not a commodity proxy.

### Caveats
- Nickel (NI=F) delisted — removed from analysis
- Zinc coverage poor — not tested
- Relationship may hold over sub-periods — rolling stability not yet run
- Full basket (copper + brent jointly) not tested via Johansen yet

### Next
Stage 4 — ML forecasting with walk-forward validation