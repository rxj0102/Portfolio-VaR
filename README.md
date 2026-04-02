# Portfolio Value-at-Risk — Multi-Asset ETF (99% Confidence)

A rigorous, end-to-end study of **13 VaR models** across a diversified multi-asset ETF portfolio, evaluated with the Christoffersen (1998) conditional coverage framework and the Hansen, Lunde & Nason (2011) Model Confidence Set.

---

## Motivation

Regulatory capital requirements (Basel III/IV) mandate that banks and asset managers maintain daily VaR estimates at 99% confidence. Choosing the wrong model can mean either holding excessive capital (parametric models with fat-tail misspecification) or under-reserving during crises (normal-distribution models that break down precisely when risk is highest).

This project benchmarks the full spectrum of industry-standard VaR methodologies — from simple Historical Simulation to APARCH-filtered Extreme Value Theory — on a realistic 5-asset ETF portfolio over 20 years of live market data, including three major stress periods: the Global Financial Crisis, COVID-19, and the 2022 rate shock.

---

## Portfolio

| ETF | Exposure | Weight |
|-----|----------|--------|
| SPY | US Large-Cap Equities | 35% |
| QQQ | Technology / Growth | 30% |
| EFA | International Developed | 15% |
| GLD | Gold | 10% |
| TLT | Long-Duration Treasuries | 10% |

**Study period:** November 2004 – December 2024 (~4,800 aligned trading days)

---

## Methodology

### Model Suite

| Family | Models | Key Feature |
|--------|--------|-------------|
| **Historical Simulation** | HS | Non-parametric rolling quantile |
| **Parametric GARCH** | GARCH-N, GARCH-t | Symmetric volatility clustering |
| **Asymmetric GARCH** | GJR-GARCH-N/t, EGARCH-t | Leverage effect (bad news → more vol) |
| **Long-memory GARCH** | FIGARCH-N/t | Fractional integration (d ∈ (0,1)) |
| **Filtered HS** | FHS, FHS-EWMA, FHS-APARCH | GARCH volatility + empirical residual tail |
| **EVT/GPD Hybrid** | FHS-EVT, APARCH-EVT | Peaks-Over-Threshold tail fitting |

### Backtesting Framework — Christoffersen (1998)

Let $I_t = \mathbf{1}\{r_t < -\widehat{\text{VaR}}_t\}$ be the daily violation indicator.

**POF test (Kupiec 1995)** — unconditional coverage, $\chi^2(1)$:

$$\text{LR}_\text{POF} = -2\left[n_1 \ln\frac{\alpha}{\hat{\pi}} + n_0 \ln\frac{1-\alpha}{1-\hat{\pi}}\right]$$

**IND test** — serial independence of violations, $\chi^2(1)$:

$$\text{LR}_\text{IND} = -2\left[\ln\mathcal{L}(H_0) - \ln\mathcal{L}(H_1)\right]$$

where $H_1$ models a first-order Markov switching process on $\{I_t\}$.

**CC test** — conditional coverage (joint), $\chi^2(2)$:

$$\text{LR}_\text{CC} = \text{LR}_\text{POF} + \text{LR}_\text{IND}$$

### EVT — Peaks-Over-Threshold

Standardised GARCH residuals are passed through the Generalised Pareto Distribution (GPD) at threshold $u$ (90th percentile):

$$\widehat{\text{VaR}}_\alpha = \hat{\sigma}_{t+1|t} \cdot \left[u + \frac{\hat{\sigma}_\text{GPD}}{\hat{\xi}}\left(\left(\frac{n_u/n}{\alpha}\right)^{\hat{\xi}} - 1\right)\right]$$

### Model Confidence Set — Hansen et al. (2011)

Surviving models are ranked by **tick (quantile) loss** — a proper scoring rule — and iteratively refined via a bootstrap t-bar statistic until the null of equal predictive ability cannot be rejected at 10%.

$$L_t = (I_t - \alpha)(r_t + \widehat{\text{VaR}}_t)$$

---

## Key Results

| Model | Violations | Rate | p_POF | p_IND | p_CC | MCS |
|-------|-----------|------|-------|-------|------|-----|
| **GJR-GARCH-t** | 41 | 0.85% | 0.291 | 0.000 | 0.001 | ★ |
| **GARCH-t** | 38 | 0.79% | 0.128 | 0.000 | 0.000 | ★ |
| **FIGARCH-t** | 38 | 0.79% | 0.128 | 0.003 | 0.004 | ★ |
| APARCH-EVT | 48 | 1.00% | 0.987 | 0.001 | 0.005 | |
| FHS-EVT | 47 | 0.98% | 0.872 | 0.001 | 0.005 | |
| GARCH-N | 83 | 1.73% | 0.000 | 0.072 | 0.000 | |
| GJR-GARCH-N | 85 | 1.77% | 0.000 | 0.084 | 0.000 | |

> Expected violations at 99% VaR over the backtest window: **48.1**

**Headline findings:**

- **GJR-GARCH-t dominates** — captures both the leverage effect (GJR) and fat tails (Student-t), passes the POF test, and survives the Model Confidence Set.
- **Normal-distribution GARCH models fail** — systematic under-estimation of tail risk (83–85 violations vs. ~48 expected), driven by leptokurtic return distribution (excess kurtosis ≈ 9.9).
- **EVT improves crisis-period calibration** — FHS-EVT and APARCH-EVT show lower squared-exceedance loss during the GFC but are slightly conservative in normal regimes.
- **All models fail the IND test** — violation clustering persists even in the best models, suggesting regime-switching dynamics not captured by single-regime GARCH.

### Component VaR — Risk Attribution

Portfolio VaR: **2.00%** (99%), using APARCH-EVT critical value.

| Asset | Weight | CompVaR | Risk Share |
|-------|--------|---------|------------|
| SPY | 35% | 0.99% | 49.9% |
| QQQ | 30% | 0.40% | 20.2% |
| EFA | 15% | 0.38% | 19.1% |
| GLD | 10% | 0.24% | 12.0% |
| TLT | 10% | −0.02% | −1.2% ✦ |

✦ TLT acts as a diversifier under normal conditions; this benefit collapses to −2.10% during GFC stress.

**Stress VaR (GFC 2008–2009 covariance):** **3.78%** — an 89% widening vs. normal regime.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| `arch` | GARCH/GJR/FIGARCH/EGARCH/APARCH fitting |
| `scipy.stats` | Generalised Pareto Distribution, statistical tests |
| `statsmodels` | ADF stationarity test, ACF |
| `yfinance` | Market data (split- and dividend-adjusted) |
| `numpy / pandas` | Numerical computation and time-series handling |
| `matplotlib` | Visualisation |
| `pytest` | Unit tests |

---

## Project Structure

```
Portfolio-VaR/
├── src/
│   ├── __init__.py
│   └── metrics.py              # Backtesting core: Christoffersen, ES test, MCS
├── notebooks/
│   └── portfolio_var_analysis.ipynb   # Full analysis notebook
├── tests/
│   ├── __init__.py
│   └── test_metrics.py         # 26 unit tests for src/metrics.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## How to Run

### 1. Clone and install dependencies

```bash
git clone https://github.com/rxj0102/portfolio-var.git
cd portfolio-var
pip install -r requirements.txt
```

### 2. Run the analysis notebook

```bash
jupyter notebook notebooks/portfolio_var_analysis.ipynb
```

The notebook downloads data from Yahoo Finance automatically — no local data files required.

### 3. Run tests

```bash
pytest tests/ -v
```

All 26 tests should pass in under 5 seconds.

---

## Methodology Notes

### Why Student-t over Normal?

The portfolio's excess kurtosis of **9.94** implies tail events occur ~10× more often than a Normal distribution predicts. At 99% VaR, the critical value shifts from $z_{0.01}^{\mathcal{N}} = -2.33$ to $z_{0.01}^{t_5} \approx -3.37$ — a 45% larger VaR estimate that prevents systematic under-reserving.

### Why GARCH over HS?

Historical Simulation responds to volatility with a lag of ~half the rolling window (126 days for a 252-day window). During the COVID crash in March 2020, HS VaR was still anchored to the calm 2019 regime while realised volatility tripled overnight. GARCH updates daily.

### Why EVT for the Tail?

The GPD — with estimated shape parameter $\hat{\xi} \approx 0.18$ — places mass beyond the empirical support of the data. This matters at 99%: the EVT quantile exceeds the empirical FHS quantile by ~15% in stress regimes, providing a systematic tail buffer.

---

## Future Improvements

- **DCC-GARCH**: Dynamic Conditional Correlation for time-varying asset correlations in Component VaR
- **ES backtest**: Full ES regression backtest (Acerbi-Szekely 2014) beyond McNeil-Frey
- **Regime-switching GARCH**: Two-state Markov model to address persistent IND test failures
- **Monte Carlo VaR**: Full simulation cross-check on analytical parametric methods
- **Real-time monitoring**: Streamlit dashboard for live VaR with automated backtesting alerts
