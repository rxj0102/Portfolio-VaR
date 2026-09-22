# Writing-Sample Source Material — Portfolio-VaR

Everything below comes from files already in this repository at commit `3e58762`, or from its git history. The one exception is §5: the test suite was run for this extraction. No new analysis was run and no numbers were estimated. Where a value in one file conflicts with another file, both values are shown and the conflict is flagged. **Treat the flags in §3.4 and §4.2 as critical before quoting any number.**

Sources and abbreviations used below:
- `README` = `README.md` (current version, last changed in commit `8ff88a2`)
- `NB` = `notebooks/portfolio_var_analysis.ipynb` (cell numbers are 0-indexed; "saved output" means the text actually stored in the cell's output)
- `metrics.py` = `src/metrics.py`
- `tests` = `tests/test_metrics.py`

---

## 1. PROJECT OVERVIEW

### 1.1 Research question / motivation (verbatim)

**NB, cell 0 (markdown):**
> **Research question:** Which VaR model best balances accuracy and tail-risk capture for a diversified ETF portfolio?

**README, title and subtitle:**
> # Portfolio Value-at-Risk: A 20-Year Model Tournament on a Multi-Asset ETF Portfolio
>
> A comparative study of 13 VaR models — GARCH family, Filtered Historical Simulation, and EVT/GPD hybrids — evaluated under the Christoffersen (1998) conditional coverage framework and the Hansen–Lunde–Nason (2011) Model Confidence Set, across 4,800 trading days and three major stress regimes.

**README, "Research Contribution"** (the three claimed contributions; quoted in full in §3.3 because some conflict with the saved output):
1. "Formal model ranking via proper scoring rules." (MCS on tick loss)
2. "IND test failures are a signal, not noise."
3. "Correlation collapse drives 89% of crisis VaR widening."

**Earlier README motivation paragraph** (commit `e568cf7`, `git show e568cf7:README.md`, line 9; removed in the current README):
> Regulatory capital requirements (Basel III/IV) mandate that banks and asset managers maintain daily VaR estimates at 99% confidence. Choosing the wrong model can mean either holding excessive capital (parametric models with fat-tail misspecification) or under-reserving during crises (normal-distribution models that break down precisely when risk is highest).

### 1.2 Dataset (exact values)

| Item | Value | Source |
|---|---|---|
| Assets / weights | SPY 0.35, QQQ 0.30, EFA 0.15, GLD 0.10, TLT 0.10 (static weights) | NB cell 2 (`TICKERS`, `WEIGHTS`) |
| Data source | Yahoo Finance via `yf.download(..., auto_adjust=True)` using the `Close` column (split- and dividend-adjusted) | NB cells 3–4 |
| Requested download window | `START, END = "2000-01-01", "2025-01-01"` | NB cell 2 |
| Returns | daily log returns; `dropna()` across all 5 ETFs; portfolio return = `log_returns.dot(WEIGHTS)` | NB cell 4 |
| **Full return sample (saved output)** | **2004-11-19 to 2024-12-31, 5062 trading days** | NB cell 4 output |
| **Aligned backtest sample (saved output)** | **4811 days, 2005-11-17 to 2024-12-31** | NB cell 10 output |
| Expected violations at 99% | 48.1 | NB cell 16 output |
| VaR level | `ALPHA = 0.01` (99% VaR) | NB cell 2 |
| Rolling window (HS / FHS) | `HS_WINDOW = 252` | NB cell 2 |
| GFC stress-covariance window (saved output) | 2008-01-02 to 2009-06-30, 377 days | NB cell 24 output |

**Descriptive statistics (NB cell 4 saved output, verbatim):**
```
Sample: 2004-11-19 to 2024-12-31  (5062 trading days)

Return statistics
  Mean: 7.56% ann.  |  Vol: 13.43% ann.
  Skew: -0.245  |  Ex. Kurt: 9.939
  Jarque-Bera p = 0.00e+00  -> REJECT normality
  ADF p = 0.00e+00  -> stationary
```

**How README describes the sample:** "November 2004 – December 2024 (~4,800 aligned trading days)", "Ann. vol 13.4%, skew −0.245, excess kurtosis **9.94**, Jarque-Bera p < 0.001". Title and subtitle: "20-Year", "4,800 trading days".

**Data-provenance notes (factual observations):**
- NB cell 3 says: "`dropna()` on the full return matrix truncates to the latest common ETF inception (TLT: 2002, EFA: 2001)." The saved output, however, starts on **2004-11-19**, which is later than both dates the note gives. The repo never states which ticker sets the 2004 start date.
- No price or return data snapshot is saved in the repo. README: "Data is downloaded automatically from Yahoo Finance — no local files required." Re-running the notebook downloads current Yahoo data, and adjusted prices can be revised later, so a re-run may not reproduce the saved figures exactly.
- The notebook was last run with kernel "base", Python 3.13.5 (from the notebook metadata).

---

## 2. METHODOLOGY (as implemented)

### 2.1 VaR models: 13 models, all in `NB`

The count "13" is confirmed by the NB cell 16 saved output: "Backtesting 13 models | 4811 days". General convention: every GARCH-type model is fit **once, on the full return sample** (`arch_model(port_ret * 100, ...).fit(...)`), with `mean="Zero"`. The VaR for day t uses `conditional_volatility.shift(1)`. The comment in the code reads "shift(1) = use today's estimate for tomorrow".

| # | Model (column name) | How it is actually coded | Cell |
|---|---|---|---|
| 1 | `HS VaR` | `-port_ret.rolling(252).quantile(0.01)`. Comment: "empirical rolling quantile — no distributional assumption, look-ahead safe". | 8 |
| 2 | `GARCH-N VaR` | `arch_model(vol="Garch", p=1, o=0, q=1, dist="normal")`. VaR = −σ̂_fwd × `norm.ppf(0.01)`. | 10 |
| 3 | `GARCH-t VaR` | Same as #2 with `dist="t"`. Critical value = `student_t.ppf(0.01, df=ν̂)` using the fitted ν. | 10 |
| 4 | `GJR-GARCH-N VaR` | `vol="Garch", p=1, o=1, q=1, dist="normal"` (the `o=1` term is the asymmetry/leverage term). | 10 |
| 5 | `GJR-GARCH-t VaR` | Same as #4 with `dist="t"`. | 10 |
| 6 | `FIGARCH-N VaR` | `vol="FIGARCH", p=1, o=0, q=1, dist="normal"`. | 10 |
| 7 | `FIGARCH-t VaR` | Same as #6 with `dist="t"`. | 10 |
| 8 | `EGARCH-t VaR` | `vol="EGarch", p=1, o=1, q=1, dist="t"`. | 10 |
| 9 | `FHS VaR` | Function `fhs_var`: VaR_t = −σ̂_fwd,t × empirical 1% quantile of standardized residuals over the preceding 252 obs (`z.iloc[max(0,i-window):i]`). Returns NaN until i ≥ max(20, 63). The filter is GARCH(1,1)-Normal fit on the full sample. Docstring: "Strictly backward-looking window -- no look-ahead bias." | 12 |
| 10 | `FHS-EWMA VaR` | Same `fhs_var`. The filter is a hand-coded EWMA, λ = 0.94, σ²₀ = sample variance of all returns, σ²_t = λσ²_{t−1} + (1−λ)r²_{t−1}. | 12 |
| 11 | `FHS-APARCH VaR` | Same `fhs_var`. The filter is `arch_model(vol="APARCH", p=1, o=1, q=1, dist="normal")`. Saved output: "APARCH delta = 1.527". | 12 |
| 12 | `FHS-EVT VaR` | Function `fit_evt_var`: GPD fit (`genpareto.fit(exceed, floc=0)`) to exceedances of the **negated GARCH(1,1) standardized residuals over the aligned backtest index** above the **90th percentile**. Quantile q_α = u + `genpareto.isf(α/(n_u/n), ξ, 0, scale)`. VaR_t = σ̂_fwd,t × q_α, so it uses **a single static tail quantile scaled by the volatility forecast**. | 14 |
| 13 | `APARCH-EVT VaR` | Same `fit_evt_var` on APARCH standardized residuals, but with `u_pct=0.92` (92nd percentile), not 0.90. | 14 |

**Saved fit output (NB cell 10, verbatim):**
```
  Model                        AIC       BIC     nu        d
  ----------------------------------------------------------
  GARCH-N VaR              10878.2   10897.8    nan      nan
  GARCH-t VaR              10672.7   10698.9   7.22      nan
  GJR-GARCH-N VaR          10811.3   10837.5    nan      nan
  GJR-GARCH-t VaR          10617.6   10650.2   7.55      nan
  FIGARCH-N VaR            10856.0   10882.1    nan   0.4227
  FIGARCH-t VaR            10661.7   10694.3   7.29   0.4346
  EGARCH-t VaR             10605.8   10638.5   7.46      nan

Aligned sample: 4811 days (2005-11-17 to 2024-12-31)
```

**Saved EVT output (NB cell 14, verbatim):**
```
FHS-EVT    | xi=0.0570, scale=0.6446, n_u=481
APARCH-EVT | xi=0.0979, scale=0.6034, n_u=385
```
(The cell also saves a mean-excess plot image. Its title is "Mean Excess Plot -- linear above u confirms GPD". No numbers are printed for it.)

**Implementation facts to be aware of.** These are my readings of the code, not limitations the repo states itself:
- All GARCH/APARCH parameters, and the GPD tail parameters of the EVT models, are estimated **once on the full sample** and then used to produce VaR over that same sample. The code does not re-estimate on a rolling or expanding window. README describes FHS as "Strictly backward-looking — no look-ahead bias". That is true only of the 252-day residual-quantile window, not of the filter parameters.
- `HS VaR` uses `port_ret.rolling(252).quantile(0.01)` without a `.shift(1)`. The window for day t therefore includes r_t itself, even though the comment says "look-ahead safe".
- The fitted GARCH/EGARCH/FIGARCH α, β, γ and ω values are **not printed** anywhere. Only AIC, BIC, ν and d are.

### 2.2 Component VaR (NB cells 23–25)

Delta-normal: `CompVaR = (-(crit_val / sig_p) * (Sigma @ WEIGHTS)) * WEIGHTS`. Σ is the sample covariance of the 5 ETF log returns over the aligned backtest index (the "normal" case) or over 2008-01-01 to 2009-06-30 (the "GFC" case). `crit_val = _crits.get(best_model, norm.ppf(ALPHA))`. A rolling 252-day version is computed in cell 25 and saved **only as a plot**. No numbers are printed for it.

### 2.3 Backtesting framework: `src/metrics.py`

Module docstring (`metrics.py:1-35`) lists what it implements:
> christoffersen_test  : Christoffersen (1998) LR framework (POF, IND, CC tests)
> mcneil_frey_es_test  : McNeil & Frey (2000) Expected Shortfall specification test
> model_confidence_set : Hansen, Lunde & Nason (2011) MCS procedure
> run_backtest         : orchestrates the full backtest pipeline on a model suite

- **`christoffersen_test`** (`metrics.py:76-182`). Hit sequence `hit = (r < -v)`.
  - POF (Kupiec) LR ~ χ²(1). It returns NaN if π̂ ∈ {0, 1}.
  - IND: first-order Markov transition counts n00/n01/n10/n11, with a 1e-10 epsilon inside the logs, LR ~ χ²(1).
  - CC = LR_POF + LR_IND ~ χ²(2).
  - Returns `n_violations, violation_rate, p_POF, p_IND, p_CC`.
- **`mcneil_frey_es_test`** (`metrics.py:185-242`). On exceedance days it computes e_t = −r_t − ES_t, then runs `scipy.stats.ttest_1samp(e, 0)`. It returns NaN if there are fewer than 2 exceedances. Docstring: "t > 0 : actual losses exceed ES forecasts → model underestimates tail risk".
- **`model_confidence_set`** (`metrics.py:245-328`). Defaults: `alpha=0.10`, `B=500`, `seed=42`.
  - Relative loss d_{i,t} = L_{i,t} minus the cross-sectional mean.
  - t̄_i uses the Newey-West (Bartlett) variance with lag = ⌊T^(1/3)⌋ (the helper is `_newey_west_var`, `metrics.py:45-59`, hand-coded rather than from statsmodels).
  - T_max = max t̄_i. The bootstrap is **IID row resampling** (docstring: "IID bootstrap replications"), re-centered.
  - It eliminates argmax t̄_i while p ≤ 0.10.
  - Output: `in_MCS` = "★" or "".
- **`_tick_loss`** (`metrics.py:62-71`), current code: `L_t = (alpha − I_t)(r_t + VaR_t)`.
- **`run_backtest`** (`metrics.py:331-396`). It runs `christoffersen_test` on each model column.
  - Status labels: "FAIL" if p_POF ≤ 0.05; "CC ✓" if p_POF > 0.05 and p_CC > 0.05; otherwise "IND ✗".
  - It then builds tick losses and runs the MCS.

**How README describes the MCS** ("Model Confidence Set — Hansen, Lunde & Nason (2011)"): tick loss is written as $L_t = (I_t - \alpha)(r_t + \widehat{VaR}_t)$. This is the **pre-fix sign**; the code now uses (α − I_t) (see §3.4). README: "Bootstrap p-values are computed under $H_0$ by re-centring. Elimination continues until $p_{\bar{T}} > 0.10$."

**How the NB describes the MCS loss** (cell 15 markdown), which contradicts both README and code:
> **MCS loss:** squared exceedance $(r_t + \text{VaR}_t)^2 \cdot \mathbf{1}[\text{violation}]$ -- penalises severity not just count (HLN 2011).

Other tests present in the NB: Jarque-Bera and ADF (`adfuller(maxlag=10, autolag="AIC")`) in cell 4, and an ACF of squared returns (plot only) in cell 6. README's reference table lists "Engle & Ng (1993) … Leverage effect / sign bias tests", but **no sign-bias test is implemented anywhere in the repo**.

---

## 3. RESULTS: ONLY SAVED NUMBERS

**No standalone output files exist.** There is no `results/` or `output/` directory and no CSV, JSON, log, .tex or .pdf file. The only saved results are (a) the text outputs stored in the notebook and (b) numbers typed into README.md. The notebook also holds 7 embedded PNG figures: EDA 2×2, mean-excess plot, p-value heatmap, GFC VaR comparison, COVID VaR comparison, rolling 252-day violation rate, and rolling/stress Component VaR. These are images with no printed numbers.

### 3.1 Notebook saved outputs, verbatim

**Full-sample backtest, NB cell 16:**
```
Backtesting 13 models | 4811 days | Expected violations: 48.1

  Model                    Violations  ViolRate    p_POF    p_IND     p_CC  Status   MCS
  ------------------------------------------------------------------------------------------
  GJR-GARCH-N VaR                  85     1.77%   0.0000   0.0837   0.0000  FAIL     POF✗ IND✓ CC✗ 
  GARCH-N VaR                      83     1.73%   0.0000   0.0716   0.0000  FAIL     POF✗ IND✓ CC✗ 
  FIGARCH-N VaR                    79     1.64%   0.0000   0.0514   0.0000  FAIL     POF✗ IND✓ CC✗ 
  HS VaR                           69     1.43%   0.0045   0.0004   0.0000  FAIL     POF✗ IND✗ CC✗ 
  FHS-EWMA VaR                     65     1.35%   0.0202   0.0128   0.0030  FAIL     POF✗ IND✗ CC✗ 
  FHS-APARCH VaR                   63     1.31%   0.0394   0.0101   0.0044  FAIL     POF✗ IND✗ CC✗ 
  FHS VaR                          62     1.29%   0.0540   0.0090   0.0051  FAIL     POF✓ IND✗ CC✗ 
  APARCH-EVT VaR                   48     1.00%   0.9873   0.0012   0.0053  FAIL     POF✓ IND✗ CC✗ 
  FHS-EVT VaR                      47     0.98%   0.8717   0.0010   0.0045  FAIL     POF✓ IND✗ CC✗ 
  GJR-GARCH-t VaR                  41     0.85%   0.2905   0.0003   0.0009  FAIL     POF✓ IND✗ CC✗ ★
  GARCH-t VaR                      38     0.79%   0.1283   0.0002   0.0003  FAIL     POF✓ IND✗ CC✗ ★
  FIGARCH-t VaR                    38     0.79%   0.1283   0.0030   0.0038  FAIL     POF✓ IND✗ CC✗ ★
  EGARCH-t VaR                     35     0.73%   0.0459   0.0001   0.0001  FAIL     POF✗ IND✗ CC✗ ★

  p > 0.05 = accepted model  |  * = in Model Confidence Set
```
What this output directly shows, reading only the printed values:
- **No model passes the joint CC test** at 5%. The "pass/FAIL" column reads FAIL for all 13.
- **POF passes (p > 0.05): 6 models.** FHS, APARCH-EVT, FHS-EVT, GJR-GARCH-t, GARCH-t and FIGARCH-t. FHS VaR is borderline at 0.0540.
- **IND passes (p > 0.05): 3 models.** GJR-GARCH-N, GARCH-N and FIGARCH-N. The other 10 fail IND.
- **MCS membership (★): 4 models.** GJR-GARCH-t, GARCH-t, FIGARCH-t and **EGARCH-t**. EGARCH-t is in the MCS even though it fails POF (p = 0.0459). See §3.4 for why these MCS stars may be invalid.
- The closest violation counts to the 48.1 expected are APARCH-EVT (48) and FHS-EVT (47).
- The t-distribution GARCH models have **fewer** violations than expected (35–41). The normal-distribution models have more (79–85).
- MCS p-values and elimination order are **not saved**. Only the ★ flags are.

**Sub-period violation rates, NB cell 17.** The flag "X" means |rate − 1%| / 1% > 0.5:
```
  Model                          GFC (2007-09)        COVID (2020)   Rate shock (2022)  Post-COVID (21-24)
  ----------------------------------------------------------------------------------------------------
  HS VaR                                1.85% X              2.77% X              2.79% X              1.29%  
  GARCH-N VaR                           2.51% X              3.95% X              1.59% X              1.29%  
  GARCH-t VaR                           1.06%                2.37% X              0.80%                0.50% X
  GJR-GARCH-N VaR                       2.51% X              3.56% X              1.99% X              1.69% X
  GJR-GARCH-t VaR                       1.06%                2.77% X              0.80%                0.50% X
  FIGARCH-N VaR                         1.98% X              3.95% X              2.39% X              1.49%  
  FIGARCH-t VaR                         1.19%                2.37% X              0.80%                0.50% X
  EGARCH-t VaR                          0.66%                2.37% X              0.40% X              0.30% X
  FHS VaR                               1.19%                2.77% X              1.59% X              1.19%  
  FHS-EWMA VaR                          1.19%                2.37% X              1.20%                1.29%  
  FHS-APARCH VaR                        1.32%                2.77% X              1.20%                1.09%  
  FHS-EVT VaR                           1.32%                2.77% X              1.20%                0.60%  
  APARCH-EVT VaR                        1.46%                3.16% X              0.80%                0.50% X
```
Period definitions (cell 17): GFC 2007-01-01 to 2009-12-31; COVID 2020-01-01 to 2020-12-31; Rate shock 2022-01-01 to 2022-12-31; Post-COVID 2021-01-01 to 2024-12-31. Directly from the table: **every one of the 13 models is flagged X in COVID (2020).** No formal test is run per sub-period; the rates are descriptive only.

**McNeil-Frey ES test, NB cell 18 (verbatim):**
```
McNeil-Frey ES test  |  H0: ES correctly specified  |  p > 0.05 = accept

  Model                     n_exceed    t-stat     p-val  Status
  --------------------------------------------------------------
  GJR-GARCH-t VaR                 41     0.887    0.3805  accept
  GARCH-t VaR                     38     1.083    0.2859  accept
  FIGARCH-t VaR                   38     0.819    0.4180  accept

  t > 0 = model underestimates tail losses  |  t < 0 = conservative
```
EGARCH-t is in the current cell source but has **no saved result**; the output predates that edit (see §3.4). **No ES test result exists for any Normal model, HS, FHS or EVT model.**

**Component VaR, NB cell 24 (verbatim):**
```
Component VaR (model: APARCH-EVT VaR)

|     |   Weight (%) |   CompVaR |   CompVaR % |
|:----|-------------:|----------:|------------:|
| SPY |           35 |   0.00999 |       49.91 |
| QQQ |           30 |   0.00404 |       20.19 |
| EFA |           15 |   0.00383 |       19.13 |
| GLD |           10 |   0.00239 |       11.95 |
| TLT |           10 |  -0.00024 |       -1.18 |

Portfolio VaR: 2.00%
TLT CompVaR < 0 = negative equity beta provides diversification benefit

Stress vs Normal Component VaR  |  GFC window: 2008-01-02 to 2009-06-30 (377 days)

|     |   CompVaR % (normal) |   CompVaR % (GFC) |
|:----|---------------------:|------------------:|
| SPY |                49.91 |             57.09 |
| EFA |                19.13 |             18.18 |
| QQQ |                20.19 |             13.47 |
| GLD |                11.95 |             13.36 |
| TLT |                -1.18 |             -2.1  |

Stress portfolio VaR: 3.78%  (unconditional: 2.00%)
```
The "model: APARCH-EVT VaR" label matters. Commit `3e58762`'s message says the original `BEST_RANK` made Component VaR fall "back to norm.ppf(ALPHA)" instead of using a Student-t critical value. **These saved 2.00% / 3.78% figures therefore use the Normal 1% critical value** under an unconditional delta-normal covariance.

### 3.2 README "Key Results" tables, verbatim, checked against saved NB output

**README full-sample table** ("Full-Sample Backtest (4,811 days, expected violations: 48.1)"):

| Model | Violations | Rate | p_POF | p_IND | p_CC | MCS | Matches NB cell 16? |
|---|---|---|---|---|---|---|---|
| GJR-GARCH-t | 41 | 0.85% | 0.291 | 0.000 | 0.001 | ★ | Yes (rounded) |
| GARCH-t | 38 | 0.79% | 0.128 | 0.000 | 0.000 | ★ | Yes (rounded) |
| FIGARCH-t | 38 | 0.79% | 0.128 | 0.003 | 0.004 | ★ | Yes (rounded) |
| APARCH-EVT | 48 | 1.00% | 0.987 | 0.001 | 0.005 | | Yes (rounded) |
| FHS-EVT | 47 | 0.98% | 0.872 | 0.001 | 0.005 | | Yes (rounded) |
| FHS-APARCH | 55 | 1.14% | 0.323 | 0.000 | 0.000 | | **NO.** NB: 63, 1.31%, 0.0394, 0.0101, 0.0044 |
| HS | 62 | 1.29% | 0.010 | 0.143 | 0.014 | | **NO.** NB: 69, 1.43%, 0.0045, 0.0004, 0.0000 |
| GARCH-N | 83 | 1.73% | 0.000 | 0.072 | 0.000 | | Yes (rounded) |
| GJR-GARCH-N | 85 | 1.77% | 0.000 | 0.084 | 0.000 | | Yes (rounded) |

- The README table shows **9 of the 13 models**. It **omits EGARCH-t (which carries a ★ in the saved output), FIGARCH-N, FHS and FHS-EWMA**.
- The README HS and FHS-APARCH rows first appear in commit `8ff88a2`. They do not appear in the earlier README (`e568cf7`), and **no saved output in the repo supports them**.

**README sub-period table vs NB cell 17:**

| Period | README GJR-GARCH-t | NB | README APARCH-EVT | NB | README HS | NB |
|---|---|---|---|---|---|---|
| GFC (2007–09) | 1.06% ✓ | 1.06% | 0.93% ✓ | **1.46%** | 2.10% ✗ | **1.85% X** |
| COVID (2020) | 2.77% ✗ | 2.77% X | 2.31% ✗ | **3.16% X** | 3.45% ✗ | **2.77% X** |
| Rate shock (2022) | 0.80% ✓ | 0.80% | 0.65% ✓ | **0.80%** | 1.20% ✗ | **2.79% X** |
| Post-COVID (2021–24) | 0.50% | 0.50% **X** | 0.40% | **0.50% X** | 0.62% | **1.29%** |

Only the GJR-GARCH-t column matches. **None of the README's APARCH-EVT or HS sub-period numbers appear in any saved output.**

**README Component VaR table vs NB cell 24:**

| Asset | README Normal | NB Normal | README GFC | NB GFC |
|---|---|---|---|---|
| SPY | 49.9% | 49.91 | 57.1% | 57.09 |
| QQQ | 20.2% | 20.19 | **22.4%** | **13.47** |
| EFA | 19.1% | 19.13 | **21.4%** | **18.18** |
| GLD | 12.0% | 11.95 | **1.1%** | **13.36** |
| TLT | −1.2% | −1.18 | −2.0% | −2.1 |

Portfolio VaR 2.00% → 3.78% matches the saved output. README's narrative "GLD's diversification benefit collapses during GFC (gold–equity correlations turned positive in the 2008 liquidity crisis)" relies on GLD falling to 1.1%. **The saved output shows GLD's share rising from 11.95% to 13.36%**, the opposite direction.

### 3.3 README statements that no saved number supports

| README / NB claim (verbatim) | Status |
|---|---|
| "GJR-GARCH-t, GARCH-t, and FIGARCH-t form the 90% MCS" | Saved output shows **4** ★ (also EGARCH-t). The ★s themselves may be invalid (§3.4). |
| "all normal-distribution variants are eliminated" | Consistent with the saved ★ flags, subject to §3.4. |
| "Every model — including EVT hybrids — fails the Christoffersen independence test." / "All models fail the IND test." | **Contradicted by saved output**: GARCH-N (0.0716), GJR-GARCH-N (0.0837) and FIGARCH-N (0.0514) have p_IND > 0.05. README's own table shows 0.072 and 0.084 for two of them. |
| `metrics.py:104-106` docstring: "All models studied here fail the IND test (p < 0.01)" | **Contradicted**: the three Normal models have p > 0.05, and FHS-EWMA (0.0128) and FHS-APARCH (0.0101) are above 0.01. |
| "Correlation collapse drives 89% of crisis VaR widening… primarily due to inter-asset correlation increases, not marginal volatility alone — quantified explicitly via rolling Component VaR attribution." | The +89% is only the ratio of 3.78% to 2.00% (saved). **The repo contains no decomposition of the widening into correlation vs. volatility effects.** The rolling Component VaR is saved only as a plot. |
| "Persistence α₁ + β₁ ≈ 0.98" | **No fitted α or β is printed anywhere.** |
| "Estimated ξ̂ ≈ 0.18 > 0 confirms a heavy-tailed (Fréchet domain) distribution" | **Saved ξ values are 0.0570 (FHS-EVT) and 0.0979 (APARCH-EVT).** No 0.18 appears in any output. |
| "Threshold u set at the 90th percentile" | True for FHS-EVT only. APARCH-EVT uses the 92nd percentile (`u_pct=0.92`). |
| "estimated δ̂ ≈ 1.53" (APARCH) | Supported: saved "APARCH delta = 1.527". |
| EGARCH "(typically γ < 0 for equities)" | No γ is printed. |
| "COVID … volatility tripled in 72 hours" | **No saved output supports this.** |
| NB cell 0: "**Key result:** GJR-GARCH-t dominates on all backtests" and NB cell 26: "Passes all Christoffersen tests" | **Contradicted by saved output**: GJR-GARCH-t p_IND = 0.0003, p_CC = 0.0009 (FAIL). |
| NB cell 26: "FHS-EVT and APARCH-EVT show lower squared-exceedance MCS loss" | **No loss values are saved**, and neither EVT model has a ★. |
| NB cell 26: "TLT … benefit collapses in 2022 rate shock (positive equity correlation)" | **No 2022 Component VaR or correlation is printed.** Only GFC was computed numerically. |
| NB cell 26: "FIGARCH … AIC/BIC improvement is small" | Saved AIC: FIGARCH-t 10661.7, GARCH-t 10672.7, GJR-GARCH-t 10617.6, EGARCH-t 10605.8 (lowest AIC and BIC of the seven). |
| `metrics.py:206-207`: "This exceeds the normal ES by ~30% for ν=5, explaining why GARCH-t passes this test while GARCH-N does not." | **GARCH-N was never ES-tested**; no saved result exists. The fitted ν are 7.22–7.55, not 5. |
| Earlier README (`e568cf7`): "z^{t_5} ≈ −3.37 … a 45% larger VaR" and "EVT quantile exceeds the empirical FHS quantile by ~15% in stress regimes" | No saved output supports either. |
| README "Proposed Extensions" expected outcomes: CVaR-optimal weights "reducing portfolio VaR by an estimated 15–25%"; DCC CompVaR "would have signalled … 30–60 days before the GFC peak"; MS-GARCH "improving the IND test p-value" | **Not implemented.** These are projections only. |
| README tech stack: "`statsmodels` … Newey-West HAC" | Newey-West is hand-coded in `metrics.py:45-59`; statsmodels is used only for ADF and ACF. |
| README / NB: "26 unit tests" | The suite collects **27** (§5). |

### 3.4 Critical provenance flag: saved outputs predate the current code

Verified from git history:
1. **All saved notebook text outputs are byte-identical to those in the original upload** (commit `743a9ac`, 2026-04-01, file `03_portfolio_var.ipynb`). They were never regenerated in later commits.
2. The original notebook imported `metrics` from a **hard-coded absolute local path outside this repo** (`sys.path.insert(0, str(Path("/Users/.../github") / "src"))`). `src/metrics.py` was first added to this repo later, in commit `e568cf7`. **The implementation that produced the saved backtest and MCS numbers is therefore not in the repo.**
3. The saved status strings ("POF✗ IND✓ CC✗") match **no version** of `run_backtest` in the repo's history. Those versions produce "FAIL" / "CC ✓" / "POF ✓" (before `3e58762`) or "IND ✗" (current). This is further evidence that the outputs came from a different metrics implementation.
4. Commit `3e58762` ("Fix tick-loss sign inversion, MCS bootstrap variance, and status labels") states, verbatim:
   > _tick_loss: formula was (I_t - α)(r_t + VaR_t) — always non-positive, causing MCS to eliminate best models and retain worst.
   > model_confidence_set: bootstrap t-bars used simple sample variance while observed t-bars used Newey-West HAC variance — inconsistency corrupted p-value calibration.

   The notebook was **not re-run** after this fix; only two source lines changed. **The ★ MCS memberships in the saved output and in README cannot be attributed to the corrected code.** If the saved outputs came from an equivalent of the buggy version, they may be inverted. The repo contains no post-fix MCS result.
5. The same commit changed NB cell 18 to add EGARCH-t and NB cell 24 so that Component VaR uses a Student-t critical value. Neither cell was re-run, so the saved outputs reflect the old cell code: 3 ES rows, and Component VaR on the Normal critical value.
6. The README MCS formula still shows the pre-fix sign (I_t − α).

Christoffersen violation counts and POF/IND/CC p-values do not depend on the tick-loss bug. They are computed from the hit sequence, and the current `christoffersen_test` math matches the LR formulas in README. Even so, they were produced by the out-of-repo implementation, not by the repo's code.

---

## 4. HONEST FINDINGS / LIMITATIONS

### 4.1 Limitations and negative findings stated by the project itself (verbatim)

- README, Contribution 2: "**IND test failures are a signal, not noise.** Every model — including EVT hybrids — fails the Christoffersen independence test. This systematic violation clustering is not a model-specific defect but an empirical signature of latent regime transitions absent from single-state GARCH. The finding directly motivates a Markov Regime-Switching GARCH extension." (Note: the word "every" is contradicted by the saved output; see §3.3.)
- README, below the results table: "All models fail the IND test. This is the study's most consequential finding".
- README, Extension 1: "The universal IND test failure is the study's clearest finding. Violation clustering appears in all 13 models, indicating the data-generating process has at least two volatility regimes that single-state GARCH cannot span."
- README, Sub-Period: "COVID violations (GJR-GARCH-t: 2.77%) reflect the March 2020 regime break — volatility tripled in 72 hours, a transition speed no daily-updating GARCH can fully absorb."
- README, Extension 2: "The delta-normal Component VaR uses an **unconditional** covariance matrix, ignoring the documented increase in asset correlations during market stress. This understates equity concentration risk precisely when it matters most."
- README, Extension 3: "VaR is not sub-additive — it can penalise diversification."
- README, Component VaR: "Static delta-normal Component VaR underestimates crisis-period risk because it ignores the correlation regime shift."
- README, Return properties: "Jarque-Bera p < 0.001 (normality strongly rejected)".
- NB cell 26: "**Normal models fail at 99%** | GARCH-N and GJR-GARCH-N over-represent violations in stress periods -- fat tails are not optional".
- NB cell 26: "**HS is anti-conservative in crises** | Rolling quantile too slow to update -- violation clustering persists during drawdowns".
- NB cell 26: "**FIGARCH long-memory adds marginal value** | AIC/BIC improvement is small; backtest performance similar to GJR-GARCH-t".
- `metrics.py:104-106`: "Note: All models studied here fail the IND test (p < 0.01), indicating persistent violation clustering. This is a systemic property of single-regime GARCH, not a model-specific defect".
- `metrics.py:256-258`: models outside M* "are *formally inferior*" (framing of the MCS).
- Commit `3e58762` message (bugs disclosed by the project's own history): tick-loss sign inversion "causing MCS to eliminate best models and retain worst"; bootstrap/observed variance "inconsistency corrupted p-value calibration"; the old status label "'POF ✓' was shown when POF passed but CC/IND failed, implying a passing grade"; EGARCH-t "was omitted" from the ES test "despite being a Student-t model"; Component VaR was "falling back to norm.ppf(ALPHA)".

### 4.2 Unexpected or negative results visible in saved output (read directly, not re-derived)

- **0 of 13 models pass the joint Christoffersen CC test** at 5% (NB cell 16).
- The only models that pass IND are the three **Normal** models. These are also the models that fail POF worst (79–85 violations vs 48.1 expected).
- The t-GARCH models are **over-conservative** in the full sample: 35–41 violations vs 48.1 expected. EGARCH-t (35 violations, 0.73%) **fails POF (p = 0.0459) from the conservative side** yet is in the saved MCS.
- The two EVT hybrids have the closest-to-nominal violation counts (48 and 47; p_POF 0.9873 and 0.8717) but still fail IND and CC, and are **not** in the saved MCS.
- **Every model is flagged in COVID 2020** (2.37%–3.95% violation rates).
- Post-COVID 2021–24: the t-GARCH models and APARCH-EVT are flagged for **too few** violations (0.30%–0.50%).
- EGARCH-t has the lowest AIC and BIC of the seven GARCH fits (10605.8 / 10638.5), but the README never mentions it.
- The fitted EVT shape parameters are small (ξ = 0.0570 and 0.0979). README instead states ξ ≈ 0.18.
- In the saved GFC Component VaR, **GLD's risk share rises** (11.95% → 13.36%) and **QQQ's falls** (20.19% → 13.47%). This runs against README's narrative.
- The three t-GARCH ES tests all "accept" with **positive** t-stats (0.887, 1.083, 0.819). By the notebook's own legend, "t > 0 = model underestimates tail losses", but none is significant.

### 4.3 Values that do not exist anywhere in the repo

- The MCS p-values, elimination order and per-model tick-loss means
- **Any MCS result computed with the corrected (post-`3e58762`) code**
- Fitted GARCH ω, α, β, γ and persistence values
- The McNeil-Frey result for EGARCH-t or any non-t model
- Formal per-sub-period backtest statistics (only raw rates are saved)
- Rolling Component VaR numbers (plot only)
- Any correlation matrices, normal vs. GFC
- A decomposition of the +89% VaR widening into correlation vs. volatility effects
- 2022 Component VaR / TLT correlation figures
- Any Expected Shortfall levels themselves
- Any runtime, seed or data-download timestamp
- Any MS-GARCH, DCC or CVaR-optimization results (these are proposals only)

---

## 5. TEST SUITE STATUS (run now)

Command: `python3 -m pytest tests/ -v`. Environment: Python 3.11.15, pytest 9.1.1. The suite imports only numpy, pandas and scipy.

Summary line (verbatim):
```
============================== 27 passed in 1.19s ==============================
```
- **27 collected, 27 passed (100%).** Breakdown: `TestChristoffersenTest` 8, `TestMcNeilFreyESTest` 5, `TestModelConfidenceSet` 6, `TestRunBacktest` 8.
- README says "26 unit tests" in two places. The 27th, `test_mcs_with_tick_loss_retains_good_model`, was added in commit `3e58762` as a regression test for the tick-loss sign bug.
- Scope: the tests cover only `src/metrics.py` on **synthetic** data (seeded normal and exponential draws). **No test exercises the notebook, the VaR models, the data pipeline, or reproduces any reported number.**

---

## 6. FILE INVENTORY (files with prose)

| File | What's in it |
|---|---|
| `README.md` | The main write-up. Title/abstract; 3 "Research Contribution" claims; portfolio table; LaTeX math for GARCH, GJR, EGARCH, FIGARCH, APARCH, FHS, EWMA, EVT/POT, Component VaR, Christoffersen, ES and MCS; Key Results tables (full-sample, sub-period, Component VaR); 3 Proposed Extensions (MS-GARCH, DCC-GARCH, CVaR optimisation); tech stack; structure; references table. Parts conflict with saved output (§3.2–3.3). |
| `notebooks/portfolio_var_analysis.ipynb` | The end-to-end analysis and the **only source of saved results**. Markdown: research question and "Key result" (cell 0), model table (cell 9), FHS/EVT explanations (cells 11, 13), backtest/MCS description (cell 15), Component VaR note (cell 23), "Key Insights" table and recommendation (cell 26). Saved text outputs are in cells 4, 10, 12, 14, 16, 17, 18 and 24; 7 PNG figures are embedded. |
| `src/metrics.py` | Module docstring (statistical background plus 4 references) and detailed function docstrings for the Christoffersen, McNeil-Frey and MCS tests and `run_backtest`, including the "all models fail IND" note. |
| `tests/test_metrics.py` | Short docstrings describing each test's intent, e.g. the tick-loss regression test's rationale. |
| `requirements.txt` | Dependency list: numpy, pandas, scipy, statsmodels, arch, yfinance, matplotlib, tabulate, pytest. No prose. |
| `LICENSE` | MIT License, "Copyright (c) 2026 rxj0102". |
| `src/__init__.py`, `tests/__init__.py` | One-line comment or empty. |
| **Git history** (not files, but prose in the repo) | Commit `3e58762`'s message is a detailed bug disclosure (§3.4). `git show e568cf7:README.md` is an earlier README with a Basel III/IV motivation paragraph, an "Interpretation" section (t₅ critical value, ξ ≈ 0.18, EVT ~15% claim) and a "Limitations & Extensions" list. `git show 743a9ac:03_portfolio_var.ipynb` is the original notebook, with the same outputs as the current one. |

**Not present in the repo:** no paper draft (.md/.tex/.pdf SSRN-style), no `docs/` directory, no results/output directory, no CSV/JSON/log files, no saved data.
