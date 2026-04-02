"""
metrics.py — VaR backtesting and model evaluation metrics.

Implements:
  christoffersen_test  : Christoffersen (1998) LR framework (POF, IND, CC tests)
  mcneil_frey_es_test  : McNeil & Frey (2000) Expected Shortfall specification test
  model_confidence_set : Hansen, Lunde & Nason (2011) MCS procedure
  run_backtest         : orchestrates the full backtest pipeline on a model suite

Statistical Background
----------------------
A correctly specified VaR model at level α must satisfy two properties:
  1. Unconditional correctness : E[I_t] = α
  2. Conditional correctness   : E[I_t | F_{t-1}] = α  (violations are unpredictable)

where I_t = 1{r_t < -VaR_t} is the daily hit/violation indicator.

The Christoffersen (1998) framework tests both simultaneously via nested LR tests
on the first-order Markov structure of {I_t}. The Model Confidence Set then
formally ranks surviving models by their tick-loss performance, providing a
statistically rigorous answer to "which model is best" rather than a binary
pass/fail verdict.

References
----------
Christoffersen, P. (1998). Evaluating interval forecasts.
    International Economic Review, 39(4), 841–862.
McNeil, A. J., & Frey, R. (2000). Estimation of tail-related risk measures for
    heteroscedastic financial time series. Journal of Empirical Finance, 7(3-4), 271–300.
Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set.
    Econometrica, 79(2), 453–497.
Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite,
    heteroskedasticity and autocorrelation consistent covariance matrix.
    Econometrica, 55(3), 703–708.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2


# ── Internal helpers ──────────────────────────────────────────────────────────

def _newey_west_var(x: np.ndarray, lag: int) -> float:
    """Newey-West HAC variance estimate (Bartlett kernel).

    Parameters
    ----------
    x   : 1-D array, assumed mean-zero or will be demeaned implicitly via caller.
    lag : truncation lag.
    """
    n = len(x)
    x_dm = x - x.mean()
    var = float(np.dot(x_dm, x_dm) / n)
    for l in range(1, lag + 1):
        gamma_l = float(np.dot(x_dm[l:], x_dm[:-l]) / n)
        var += 2.0 * (1.0 - l / (lag + 1)) * gamma_l  # Bartlett weight
    return max(var, 1e-12)


def _tick_loss(returns: np.ndarray, var: np.ndarray, alpha: float) -> np.ndarray:
    """Tick (quantile) loss for VaR evaluation.

    L_t = (I_t − alpha) * (r_t + VaR_t),  I_t = 1{r_t < −VaR_t}

    A proper scoring rule for the alpha-quantile; used internally by
    model_confidence_set and run_backtest.
    """
    hit = (returns < -var).astype(float)
    return (hit - alpha) * (returns + var)


# ── Public API ────────────────────────────────────────────────────────────────

def christoffersen_test(
    returns: pd.Series,
    var_series: pd.Series,
    alpha: float = 0.01,
) -> dict:
    """Christoffersen (1998) conditional coverage test for VaR models.

    Runs three nested likelihood-ratio tests on the hit sequence
    I_t = 1{r_t < −VaR_t}:

      POF  (Kupiec 1995) : H0: π̂ = α  (unconditional coverage)     chi²(1)
      IND               : H0: π01 = π11 (serial independence)         chi²(1)
      CC                : joint conditional coverage                  chi²(2)

    **POF statistic:**
      LR_POF = −2 [n1·ln(α/π̂) + n0·ln((1−α)/(1−π̂))]
      where n1 = Σ I_t, n0 = T − n1, π̂ = n1/T.

    **IND statistic** (first-order Markov transition probabilities):
      π̂_ij = n_ij / Σ_j n_ij  (empirical transition rates)
      LR_IND = −2 [ℓ(π̂) − ℓ(π̂01, π̂11)]
      ℓ(π̂)         = (n00+n10)·ln(1−π̂) + (n01+n11)·ln(π̂)
      ℓ(π̂01, π̂11) = n00·ln(1−π̂01) + n01·ln(π̂01)
                      + n10·ln(1−π̂11) + n11·ln(π̂11)

    **CC statistic:**
      LR_CC = LR_POF + LR_IND  ~  chi²(2)

    Note: All models studied here fail the IND test (p < 0.01), indicating
    persistent violation clustering. This is a systemic property of single-regime
    GARCH, not a model-specific defect — see README §Research Contribution.

    Parameters
    ----------
    returns    : Signed portfolio returns (e.g. log-returns).
    var_series : VaR forecasts in *loss* convention (positive values).
    alpha      : Tail probability, e.g. 0.01 for 99% VaR.

    Returns
    -------
    dict with keys: n_violations, violation_rate, p_POF, p_IND, p_CC.
    """
    idx = returns.index.intersection(var_series.dropna().index)
    r = returns.reindex(idx).values
    v = var_series.reindex(idx).values

    hit = (r < -v).astype(int)
    n, n1 = len(hit), int(hit.sum())
    n0 = n - n1
    pi_hat = n1 / n

    # --- POF test (Kupiec 1995) ---
    if 0 < pi_hat < 1:
        lr_pof = -2.0 * (
            n1 * np.log(alpha / pi_hat) + n0 * np.log((1.0 - alpha) / (1.0 - pi_hat))
        )
        p_pof = float(chi2.sf(max(lr_pof, 0.0), df=1))
    else:
        lr_pof = p_pof = np.nan

    # --- IND test (Christoffersen 1998) ---
    n00 = n01 = n10 = n11 = 0
    for t in range(1, n):
        prev, curr = hit[t - 1], hit[t]
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi_c = (n01 + n11) / (n - 1)  # unconditional rate over lag-1 sample

    eps = 1e-10
    try:
        ll_h0 = (
            (n00 + n10) * np.log(1.0 - pi_c + eps)
            + (n01 + n11) * np.log(pi_c + eps)
        )
        ll_h1 = (
            n00 * np.log(1.0 - pi01 + eps)
            + n01 * np.log(pi01 + eps)
            + n10 * np.log(1.0 - pi11 + eps)
            + n11 * np.log(pi11 + eps)
        )
        lr_ind = max(-2.0 * (ll_h0 - ll_h1), 0.0)
        p_ind = float(chi2.sf(lr_ind, df=1))
    except Exception:
        lr_ind = p_ind = np.nan

    # --- CC test ---
    if np.isfinite(lr_pof) and np.isfinite(lr_ind):
        p_cc = float(chi2.sf(lr_pof + lr_ind, df=2))
    else:
        p_cc = np.nan

    return {
        "n_violations": n1,
        "violation_rate": float(pi_hat),
        "p_POF": p_pof,
        "p_IND": p_ind,
        "p_CC": p_cc,
    }


def mcneil_frey_es_test(
    returns: pd.Series,
    var_series: pd.Series,
    es_series: pd.Series,
) -> dict:
    """McNeil & Frey (2000) test for correct ES specification.

    On VaR exceedance days (r_t < −VaR_t), tests whether the ES is correctly
    specified via H₀: E[−r_t − ES_t | exceed] = 0.

    Exceedance residuals: e_t = −r_t − ES_t  on days where r_t < −VaR_t.
    Under correct specification: E[e_t] = 0.
    Test statistic: t = ē / (s_e / √n_exceed)  ~  t(n_exceed − 1).

    **Interpretation of t-statistic sign:**
      t > 0  :  actual losses exceed ES forecasts  → model underestimates tail risk
      t < 0  :  ES forecasts exceed actual losses   → model is conservative (overstates risk)

    For GARCH-t, the parametric ES has the closed form:
      ES_t = σ_{t+1|t} · [f_ν(t_ν^{-1}(α)) / α] · (ν + (t_ν^{-1}(α))²) / (ν − 1)
    where f_ν and t_ν^{-1} are the Student-t pdf and quantile function (df = ν).
    This exceeds the normal ES by ~30% for ν=5, explaining why GARCH-t passes
    this test while GARCH-N does not.

    Parameters
    ----------
    returns    : Signed portfolio returns.
    var_series : VaR estimates (positive, loss convention).
    es_series  : ES estimates (positive; should satisfy ES ≥ VaR).

    Returns
    -------
    dict with keys: n_exceed, t_stat, p_val.
    """
    idx = (
        returns.index
        .intersection(var_series.dropna().index)
        .intersection(es_series.dropna().index)
    )
    r = returns.reindex(idx)
    v = var_series.reindex(idx)
    es = es_series.reindex(idx)

    exceed_mask = r < -v
    n_exceed = int(exceed_mask.sum())

    if n_exceed < 2:
        return {"n_exceed": n_exceed, "t_stat": np.nan, "p_val": np.nan}

    # Exceedance residuals: positive when actual loss > ES forecast
    e = -r[exceed_mask] - es[exceed_mask]
    t_stat, p_val = stats.ttest_1samp(e.values, popmean=0.0)

    return {
        "n_exceed": n_exceed,
        "t_stat": float(t_stat),
        "p_val": float(p_val),
    }


def model_confidence_set(
    loss_df: pd.DataFrame,
    alpha: float = 0.10,
    B: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Model Confidence Set (Hansen, Lunde & Nason 2011).

    Constructs the smallest set M* of models such that:
      P(best model ∈ M*) ≥ 1 − α.

    This is strictly stronger than a backtest pass/fail: models in M* are
    statistically indistinguishable in predictive ability; models outside M*
    are *formally inferior*.

    **Algorithm:**
      1. Compute relative loss: d_{i,t} = L_{i,t} − L̄_t  (deviation from cross-sectional mean)
      2. t-bar statistic for model i:
           t̄_i = d̄_i / (ω̂_i / √T)
         where d̄_i = mean(d_{i,t}),  ω̂_i² = Newey-West variance (lag = T^(1/3))
      3. Test statistic: T_M = max_i t̄_i
      4. Bootstrap p-value: resample rows of L, re-center, compute T_boot^(b)
           p = #{T_boot^(b) ≥ T_obs} / B
      5. If p ≤ α: eliminate argmax_i t̄_i and repeat.
         If p > α: stop — remaining models form the MCS.

    **Loss function:** Tick (quantile) loss L_t = (I_t − α)(r_t + VaR_t).
    This is the unique proper scoring rule for the α-quantile (Gneiting & Raftery 2007),
    ensuring that a correctly specified model minimises expected loss.

    Parameters
    ----------
    loss_df : DataFrame (T × M) of per-period loss values per model.
    alpha   : MCS significance level for the elimination test (default 0.10).
    B       : IID bootstrap replications for p-value calibration (default 500).
    seed    : Random seed for reproducibility.

    Returns
    -------
    DataFrame indexed by model name with column ``in_MCS`` ('★' or '').
    """
    models = list(loss_df.columns)
    in_mcs = set(models)
    rng = np.random.default_rng(seed)

    while len(in_mcs) > 1:
        active = [m for m in models if m in in_mcs]
        L = loss_df[active].values.astype(float)  # (T, n_m)
        T, n_m = L.shape

        # Relative loss: deviation of each model from the cross-sectional mean
        L_bar_cross = L.mean(axis=1, keepdims=True)  # (T, 1)
        D = L - L_bar_cross                           # (T, n_m)
        mu_D = D.mean(axis=0)                         # (n_m,)

        # HAC variance for each model's relative loss (Newey-West, lag = T^(1/3))
        lags = max(1, int(T ** (1.0 / 3.0)))
        var_D = np.array([_newey_west_var(D[:, i], lags) for i in range(n_m)])
        t_bars = mu_D / np.sqrt(var_D / T)
        T_obs = float(t_bars.max())

        # Bootstrap distribution of T_max under H₀ (iid bootstrap, re-centered)
        T_boot = np.empty(B)
        for b in range(B):
            idx_b = rng.integers(0, T, T)
            L_b = L[idx_b]
            D_b = L_b - L_b.mean(axis=1, keepdims=True)
            mu_b = D_b.mean(axis=0) - mu_D          # re-center at observed mean
            var_b = np.maximum(D_b.var(axis=0, ddof=1), 1e-12)
            T_boot[b] = (mu_b / np.sqrt(var_b / T)).max()

        p_val = float((T_boot >= T_obs).mean())

        if p_val > alpha:
            break  # H₀ not rejected; all remaining models are in the MCS

        # Eliminate the model with the highest t-bar (worst relative performance)
        worst_idx = int(t_bars.argmax())
        in_mcs.discard(active[worst_idx])

    return pd.DataFrame(
        {"in_MCS": ["★" if m in in_mcs else "" for m in models]},
        index=models,
    )


def run_backtest(
    var_df: pd.DataFrame,
    alpha: float = 0.01,
    run_mcs: bool = True,
    mcs_alpha: float = 0.10,
    mcs_B: int = 500,
) -> pd.DataFrame:
    """Run the full Christoffersen (1998) backtest suite on a model set.

    Parameters
    ----------
    var_df    : DataFrame with a ``'Portfolio Returns'`` column plus one column
                per VaR model. VaR values must be positive (loss convention).
    alpha     : Tail probability (default 0.01 = 99% VaR).
    run_mcs   : Whether to append Model Confidence Set membership (default True).
    mcs_alpha : MCS elimination significance level (default 0.10).
    mcs_B     : Bootstrap replications for MCS (default 500).

    Returns
    -------
    DataFrame indexed by model name with columns:
        Violations, ViolRate, p_POF, p_IND, p_CC, status, in_MCS (if run_mcs).
    """
    returns = var_df["Portfolio Returns"]
    model_cols = [c for c in var_df.columns if c != "Portfolio Returns"]

    records = {}
    for m in model_cols:
        if var_df[m].isnull().all():
            continue
        res = christoffersen_test(returns, var_df[m], alpha=alpha)

        p_pof = res["p_POF"]
        p_cc = res["p_CC"]
        if np.isfinite(p_pof) and p_pof > 0.05:
            status = "CC ✓" if (np.isfinite(p_cc) and p_cc > 0.05) else "POF ✓"
        else:
            status = "FAIL"

        records[m] = {
            "Violations": res["n_violations"],
            "ViolRate": res["violation_rate"],
            "p_POF": res["p_POF"],
            "p_IND": res["p_IND"],
            "p_CC": res["p_CC"],
            "status": status,
        }

    summary = pd.DataFrame(records).T
    # Ensure numeric dtypes for statistical columns
    for col in ["Violations", "ViolRate", "p_POF", "p_IND", "p_CC"]:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")

    if run_mcs:
        valid = [m for m in model_cols if m in summary.index]
        losses = {}
        for m in valid:
            idx = returns.index.intersection(var_df[m].dropna().index)
            r = returns.reindex(idx).values
            v = var_df[m].reindex(idx).values
            losses[m] = pd.Series(_tick_loss(r, v, alpha), index=idx)
        loss_df = pd.DataFrame(losses).dropna()
        mcs_result = model_confidence_set(loss_df, alpha=mcs_alpha, B=mcs_B)
        summary = summary.join(mcs_result)

    return summary
