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

**Next:** Stage 2 — Volatility modelling (GARCH).