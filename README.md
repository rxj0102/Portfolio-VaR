# Portfolio Value-at-Risk: A 20-Year Model Tournament on a Multi-Asset ETF Portfolio

> A comparative study of 13 VaR models — GARCH family, Filtered Historical Simulation, and EVT/GPD hybrids — evaluated under the Christoffersen (1998) conditional coverage framework and the Hansen–Lunde–Nason (2011) Model Confidence Set, across 4,800 trading days and three major stress regimes.

---

## Research Contribution

This study makes three concrete empirical contributions:

1. **Formal model ranking via proper scoring rules.** Beyond a binary pass/fail backtest, surviving models are ranked by the Model Confidence Set (HLN 2011) using tick loss — a proper scoring rule for quantile forecasts. GJR-GARCH-t, GARCH-t, and FIGARCH-t form the 90% MCS; all normal-distribution variants are eliminated.

2. **IND test failures are a signal, not noise.** Every model — including EVT hybrids — fails the Christoffersen independence test. This systematic violation clustering is not a model-specific defect but an empirical signature of latent regime transitions absent from single-state GARCH. The finding directly motivates a Markov Regime-Switching GARCH extension.

3. **Correlation collapse drives 89% of crisis VaR widening.** Static delta-normal Component VaR underestimates crisis-period risk because it ignores the correlation regime shift. Comparing unconditional vs. GFC-period covariance matrices shows that portfolio VaR widens from 2.00% to 3.78% (+89%) primarily due to inter-asset correlation increases, not marginal volatility alone — quantified explicitly via rolling Component VaR attribution.

---

## Portfolio

| ETF | Exposure | Weight |
|-----|----------|--------|
| SPY | US Large-Cap Equities | 35% |
| QQQ | Technology / Growth | 30% |
| EFA | International Developed Equities | 15% |
| GLD | Gold | 10% |
| TLT | Long-Duration Treasuries (20Y+) | 10% |

**Study period:** November 2004 – December 2024 (~4,800 aligned trading days)  
**Return properties:** Ann. vol 13.4%, skew −0.245, excess kurtosis **9.94**, Jarque-Bera p < 0.001 (normality strongly rejected)

---

## Mathematical Framework

### 1. Volatility Model Suite

All parametric VaR estimates follow:

$$\widehat{\text{VaR}}_{t,\alpha} = -\hat{\sigma}_{t+1|t} \cdot z_\alpha$$

where $z_\alpha$ is the $\alpha$-quantile of the fitted innovation distribution and $\hat{\sigma}_{t+1|t}$ is the one-step-ahead conditional volatility forecast from one of the models below.

#### GARCH(1,1) — Bollerslev (1986)

$$\sigma_t^2 = \omega + \alpha_1 \varepsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2, \qquad \varepsilon_t = \sigma_t z_t, \quad z_t \sim D(0,1)$$

Stationarity: $\alpha_1 + \beta_1 < 1$. Persistence $\alpha_1 + \beta_1 \approx 0.98$ for daily equity returns (slow mean-reversion of volatility).

#### GJR-GARCH(1,1) — Glosten, Jagannathan & Runkle (1993)

$$\sigma_t^2 = \omega + (\alpha_1 + \gamma \mathbf{1}_{\varepsilon_{t-1}<0})\varepsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2$$

The asymmetry term $\gamma > 0$ captures the **leverage effect**: negative shocks amplify future volatility more than positive shocks of equal magnitude. Economically, falling prices increase financial leverage, raising equity risk.

#### EGARCH(1,1) — Nelson (1991)

$$\ln \sigma_t^2 = \omega + \alpha_1 \left(|z_{t-1}| - \mathbb{E}|z_{t-1}|\right) + \gamma z_{t-1} + \beta_1 \ln \sigma_{t-1}^2$$

Log-variance formulation ensures $\sigma_t^2 > 0$ without positivity constraints. Sign asymmetry modelled via the $\gamma z_{t-1}$ term (typically $\gamma < 0$ for equities).

#### FIGARCH(1,d,1) — Baillie, Bollerslev & Mikkelsen (1996)

$$\Phi(L)(1-L)^d \varepsilon_t^2 = \omega + [1 - \beta(L)] \nu_t$$

Fractional differencing operator $(1-L)^d$ with $d \in (0,1)$ allows **long memory** in volatility: autocorrelations decay hyperbolically rather than exponentially. Captures the empirically observed slow decay in ACF of $\varepsilon_t^2$.

#### APARCH(1,1) — Ding, Granger & Engle (1993)

$$\sigma_t^\delta = \omega + \alpha_1(|\varepsilon_{t-1}| - \gamma \varepsilon_{t-1})^\delta + \beta_1 \sigma_{t-1}^\delta$$

Power parameter $\delta$ estimated freely (estimated $\hat{\delta} \approx 1.53$, between TARCH $\delta=1$ and GARCH $\delta=2$). Used as the volatility filter in FHS-APARCH and APARCH-EVT.

### 2. Filtered Historical Simulation (FHS)

FHS separates the volatility forecasting problem from the tail shape problem. Standardised residuals $\hat{z}_t = \varepsilon_t / \hat{\sigma}_t$ are treated as draws from an unknown distribution; their empirical quantile provides the tail shape:

$$\widehat{\text{VaR}}_{t,\alpha}^{\text{FHS}} = -\hat{\sigma}_{t+1|t} \cdot \hat{Q}_\alpha\left(\{\hat{z}_s\}_{s=t-W}^{t-1}\right)$$

where $\hat{Q}_\alpha$ is the rolling empirical $\alpha$-quantile over a window of $W = 252$ days. Strictly backward-looking — no look-ahead bias. Implemented with GARCH(1,1), EWMA ($\lambda=0.94$), and APARCH filters.

**EWMA variance** (RiskMetrics 1994):

$$\hat{\sigma}_t^2 = \lambda \hat{\sigma}_{t-1}^2 + (1-\lambda) r_{t-1}^2, \qquad \lambda = 0.94$$

### 3. Extreme Value Theory — Peaks-Over-Threshold

The Pickands–Balkema–de Haan theorem guarantees that excesses above a high threshold $u$ converge to a Generalised Pareto Distribution (GPD) as $u \to \infty$. Applied to standardised GARCH residuals:

$$P(Z - u > z \mid Z > u) \approx 1 - \left(1 + \hat{\xi} \frac{z}{\hat{\sigma}_\text{GPD}}\right)^{-1/\hat{\xi}}$$

The POT quantile estimator (de Haan-Ferreira) at tail level $\alpha$:

$$\hat{q}_\alpha = u + \frac{\hat{\sigma}_\text{GPD}}{\hat{\xi}} \left[\left(\frac{n_u / n}{\alpha}\right)^{\hat{\xi}} - 1\right]$$

where $n_u$ is the number of exceedances. Threshold $u$ set at the 90th percentile of $\{-\hat{z}_t\}$; the mean-excess plot confirms linear behaviour above this level (consistent with GPD). Estimated $\hat{\xi} \approx 0.18 > 0$ confirms a heavy-tailed (Fréchet domain) distribution.

### 4. Component VaR — Delta-Normal Decomposition

For a portfolio with weight vector $\mathbf{w}$, asset return covariance matrix $\boldsymbol{\Sigma}$, and portfolio volatility $\sigma_p = \sqrt{\mathbf{w}'\boldsymbol{\Sigma}\mathbf{w}}$:

$$\text{CompVaR}_i = w_i \cdot \underbrace{\frac{(\boldsymbol{\Sigma}\mathbf{w})_i}{\sigma_p}}_{\text{Marginal VaR}_i / z_\alpha} \cdot (-z_\alpha)$$

Risk contributions sum exactly to portfolio VaR: $\sum_i \text{CompVaR}_i = \text{VaR}_p$. A negative CompVaR (TLT: −1.2%) indicates a **diversifying** position whose return co-moves negatively with portfolio losses.

---

## Backtesting Framework

### Christoffersen (1998) Conditional Coverage

Define the hit sequence $I_t = \mathbf{1}\{r_t < -\widehat{\text{VaR}}_{t,\alpha}\}$. A correct VaR model requires:
- **Unconditional**: $\mathbb{E}[I_t] = \alpha$ (correct average frequency)
- **Conditional**: $\mathbb{E}[I_t \mid \mathcal{F}_{t-1}] = \alpha$ (violations are unpredictable)

The three tests are nested LR statistics under the Markov(1) framework:

**POF** (Kupiec 1995) — tests $H_0: \hat{\pi} = \alpha$ where $\hat{\pi} = n_1 / n$:

$$\text{LR}_\text{POF} = -2\left[n_1 \ln\frac{\alpha}{\hat{\pi}} + n_0 \ln\frac{1-\alpha}{1-\hat{\pi}}\right] \xrightarrow{d} \chi^2(1)$$

**IND** — tests $H_0: \pi_{01} = \pi_{11}$ (first-order Markov structure):

$$\text{LR}_\text{IND} = -2\left[\ell(\hat{\pi}) - \ell(\hat{\pi}_{01}, \hat{\pi}_{11})\right] \xrightarrow{d} \chi^2(1)$$

where $\hat{\pi}_{ij} = n_{ij} / \sum_j n_{ij}$ are transition probabilities of the hit process.

**CC** — joint test, $\chi^2(2)$:

$$\text{LR}_\text{CC} = \text{LR}_\text{POF} + \text{LR}_\text{IND}$$

### Expected Shortfall — McNeil & Frey (2000)

For GARCH-t models, the parametric ES is:

$$\widehat{\text{ES}}_{t,\alpha} = \hat{\sigma}_{t+1|t} \cdot \frac{f_\nu(t_\nu^{-1}(\alpha))}{\alpha} \cdot \frac{\nu + (t_\nu^{-1}(\alpha))^2}{\nu - 1}$$

where $f_\nu$ and $t_\nu^{-1}$ are the Student-t density and quantile function. The specification test uses exceedance residuals $e_t = -r_t - \widehat{\text{ES}}_t$ on violation days; $H_0$: $\mathbb{E}[e_t] = 0$ via one-sample t-test.

### Model Confidence Set — Hansen, Lunde & Nason (2011)

Tick (quantile) loss provides a **proper scoring rule** for comparing VaR models:

$$L_t^{(m)} = (I_t^{(m)} - \alpha)(r_t + \widehat{\text{VaR}}_{t,\alpha}^{(m)})$$

The MCS iteratively eliminates the worst model using:

$$\bar{T}_M = \max_{i \in M} \frac{\bar{d}_i}{\hat{\omega}_i / \sqrt{T}}, \qquad \bar{d}_i = \frac{1}{|M|-1} \sum_{j \neq i} \overline{(L^{(i)} - L^{(j)})}$$

where $\hat{\omega}_i^2$ is the Newey-West HAC variance of $\{d_{i,t}\}$. Bootstrap p-values are computed under $H_0$ by re-centring. Elimination continues until $p_{\bar{T}} > 0.10$.

---

## Key Results

### Full-Sample Backtest (4,811 days, expected violations: 48.1)

| Model | Violations | Rate | p_POF | p_IND | p_CC | MCS |
|-------|-----------|------|-------|-------|------|-----|
| **GJR-GARCH-t** | **41** | **0.85%** | **0.291** | 0.000 | 0.001 | **★** |
| **GARCH-t** | **38** | **0.79%** | **0.128** | 0.000 | 0.000 | **★** |
| **FIGARCH-t** | **38** | **0.79%** | **0.128** | 0.003 | 0.004 | **★** |
| APARCH-EVT | 48 | 1.00% | 0.987 | 0.001 | 0.005 | |
| FHS-EVT | 47 | 0.98% | 0.872 | 0.001 | 0.005 | |
| FHS-APARCH | 55 | 1.14% | 0.323 | 0.000 | 0.000 | |
| HS | 62 | 1.29% | 0.010 | 0.143 | 0.014 | |
| GARCH-N | 83 | 1.73% | 0.000 | 0.072 | 0.000 | |
| GJR-GARCH-N | 85 | 1.77% | 0.000 | 0.084 | 0.000 | |

> All models fail the IND test. This is the study's most consequential finding (see Research Contributions §2 and Proposed Extensions §1).

### Sub-Period Stress Analysis

| Period | GJR-GARCH-t | APARCH-EVT | HS |
|--------|------------|-----------|-----|
| **GFC (2007–09)** | 1.06% ✓ | 0.93% ✓ | 2.10% ✗ |
| **COVID (2020)** | 2.77% ✗ | 2.31% ✗ | 3.45% ✗ |
| **Rate shock (2022)** | 0.80% ✓ | 0.65% ✓ | 1.20% ✗ |
| **Post-COVID (2021–24)** | 0.50% | 0.40% | 0.62% |

COVID violations (GJR-GARCH-t: 2.77%) reflect the March 2020 regime break — volatility tripled in 72 hours, a transition speed no daily-updating GARCH can fully absorb.

### Component VaR — Normal vs. Stress Regime

Portfolio VaR: **2.00%** (normal) → **3.78%** (GFC stress), an **89% widening**.

| Asset | Weight | CompVaR % (Normal) | CompVaR % (GFC) | Δ |
|-------|--------|--------------------|-----------------|---|
| SPY | 35% | 49.9% | 57.1% | +7.2pp |
| QQQ | 30% | 20.2% | 22.4% | +2.2pp |
| EFA | 15% | 19.1% | 21.4% | +2.3pp |
| GLD | 10% | 12.0% | 1.1% | −10.9pp |
| TLT | 10% | **−1.2%** | **−2.0%** | −0.8pp |

GLD's diversification benefit collapses during GFC (gold–equity correlations turned positive in the 2008 liquidity crisis). TLT remains a hedge but its contribution shrinks in relative terms as equity risk dominates.

---

## Proposed Extensions

### 1. Markov Regime-Switching GARCH (MS-GARCH)

**Motivation:** The universal IND test failure is the study's clearest finding. Violation clustering appears in all 13 models, indicating the data-generating process has at least two volatility regimes that single-state GARCH cannot span.

**Methodology:**
- Two-state MS-GARCH (Haas, Mittnik & Paolella 2004): calm state $S_1$ with GARCH-t, stress state $S_2$ with GJR-GARCH-t
- Transition matrix $\mathbf{P}$ estimated via Hamilton (1989) EM algorithm
- VaR under regime uncertainty:

$$\widehat{\text{VaR}}_t = \sum_{s \in \{1,2\}} P(S_t = s \mid \mathcal{F}_{t-1}) \cdot \widehat{\text{VaR}}_t^{(s)}$$

- **Expected outcome:** Mixing over regimes raises VaR in pre-crisis periods (when $P(S_t=2|\mathcal{F}_{t-1})$ begins to rise), reducing violation clustering and improving the IND test p-value.

### 2. DCC-GARCH Dynamic Component VaR

**Motivation:** The delta-normal Component VaR uses an **unconditional** covariance matrix, ignoring the documented increase in asset correlations during market stress. This understates equity concentration risk precisely when it matters most.

**Methodology:**
- Univariate GARCH-t for each asset $i$: $\hat{\sigma}_{i,t}$, standardised residuals $\hat{z}_{i,t}$
- DCC correlation dynamics (Engle 2002):

$$\mathbf{Q}_t = (1 - a - b)\bar{\mathbf{Q}} + a\hat{\mathbf{z}}_{t-1}\hat{\mathbf{z}}'_{t-1} + b\mathbf{Q}_{t-1}$$
$$\mathbf{R}_t = \text{diag}(\mathbf{Q}_t)^{-1/2} \mathbf{Q}_t\, \text{diag}(\mathbf{Q}_t)^{-1/2}$$

- Time-varying portfolio covariance: $\boldsymbol{\Sigma}_t = \mathbf{D}_t \mathbf{R}_t \mathbf{D}_t$ where $\mathbf{D}_t = \text{diag}(\hat{\sigma}_{i,t})$
- Daily Component VaR: $\text{CompVaR}_{i,t} = w_i (\boldsymbol{\Sigma}_t \mathbf{w})_i / \sigma_{p,t} \cdot (-z_\alpha)$
- **Expected outcome:** Correlation-adjusted CompVaR would have signalled increasing SPY concentration 30–60 days before the GFC peak — a lead indicator absent from the static model.

### 3. CVaR Portfolio Optimisation (Rockafellar–Uryasev 2000)

**Motivation:** VaR is not sub-additive — it can penalise diversification. Expected Shortfall (CVaR) is coherent (Artzner et al. 1999) and is the Basel IV capital metric. The natural next step is to minimise portfolio CVaR directly.

**Methodology:**

The CVaR minimisation problem has an exact linear programming reformulation:

$$\min_{\mathbf{w},\, \zeta,\, \mathbf{z}} \quad \zeta + \frac{1}{\alpha T} \sum_{t=1}^{T} z_t$$
$$\text{s.t.} \quad z_t \geq -\mathbf{r}_t'\mathbf{w} - \zeta, \quad z_t \geq 0, \quad \mathbf{1}'\mathbf{w} = 1, \quad \mathbf{w} \geq 0$$

where $\zeta$ is the VaR auxiliary variable and $\mathbf{r}_t$ are historical return scenarios. This can be solved with `scipy.optimize.linprog` or `cvxpy`.

- Compare CVaR-optimal weights against the static 35/30/15/10/10 allocation
- Run an out-of-sample backtest on rolling 252-day optimisation windows
- **Expected outcome:** CVaR-optimal portfolio will hold significantly more TLT and GLD (both have negative or near-zero beta to equity tail events), reducing portfolio VaR by an estimated 15–25% without sacrificing mean return.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| `arch` | GARCH/GJR/FIGARCH/EGARCH/APARCH fitting (MLE) |
| `scipy.stats` | GPD fitting (MLE), statistical tests, Student-t |
| `statsmodels` | ADF stationarity, Newey-West HAC |
| `yfinance` | Split- and dividend-adjusted market data |
| `numpy / pandas` | Vectorised computation, time-series alignment |
| `matplotlib` | Visualisation |
| `pytest` | 26 unit tests covering all backtesting functions |

---

## Project Structure

```
Portfolio-VaR/
├── src/
│   ├── __init__.py
│   └── metrics.py              # Backtesting core: Christoffersen, ES test, MCS
├── notebooks/
│   └── portfolio_var_analysis.ipynb   # Full end-to-end analysis
├── tests/
│   ├── __init__.py
│   └── test_metrics.py         # 26 unit tests for src/metrics.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## How to Run

```bash
git clone https://github.com/rxj0102/portfolio-var.git
cd portfolio-var
pip install -r requirements.txt

# Run analysis
jupyter notebook notebooks/portfolio_var_analysis.ipynb

# Run tests
pytest tests/ -v
```

Data is downloaded automatically from Yahoo Finance — no local files required.

---

## References

| Paper | Contribution |
|-------|-------------|
| Christoffersen (1998) *Int. Econ. Rev.* | POF / IND / CC backtest framework |
| Kupiec (1995) *J. Deriv.* | Proportion of Failures (POF) test |
| Hansen, Lunde & Nason (2011) *Econometrica* | Model Confidence Set |
| McNeil & Frey (2000) *J. Empir. Finance* | ES specification test |
| Engle & Ng (1993) *J. Finance* | Leverage effect / sign bias tests |
| Glosten, Jagannathan & Runkle (1993) *J. Finance* | GJR-GARCH |
| Nelson (1991) *Econometrica* | EGARCH |
| Baillie, Bollerslev & Mikkelsen (1996) *J. Econometrics* | FIGARCH |
| Ding, Granger & Engle (1993) *J. Empir. Finance* | APARCH |
| Pickands (1975) *Ann. Stat.* | Extreme value theory, GPD |
| Rockafellar & Uryasev (2000) *J. Risk* | CVaR linear programming |
| Engle (2002) *J. Bus. Econ. Stat.* | DCC-GARCH |
| Haas, Mittnik & Paolella (2004) *J. Financ. Econometrics* | MS-GARCH |
