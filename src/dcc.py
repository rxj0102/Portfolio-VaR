"""DCC-GARCH (Engle 2002) for time-varying covariance and dynamic Component VaR.

Steps
-----
1. Univariate GARCH(1,1)-t per asset gives sigma_{i,t|t-1} and standardised
   residuals z_{i,t} = r_{i,t} / sigma_{i,t|t-1}.
2. DCC(1,1) on z:  Q_t = (1 - a - b) Qbar + a z_{t-1} z_{t-1}' + b Q_{t-1},
   R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}.  (a, b) by Gaussian quasi-ML.
3. Sigma_t = D_t R_t D_t with D_t = diag(sigma_{t|t-1}). Every Sigma_t uses
   information through t-1 only (parameters are full-sample estimates, as for
   the univariate models in the notebook).
4. Component VaR_{i,t} = -crit * w_i (Sigma_t w)_i / sigma_{p,t}, which sums to
   the delta-normal portfolio VaR -crit * sigma_{p,t}.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize


def univariate_garch(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit GARCH(1,1)-t to each column (returns in decimals).

    Returns (sigma, z): one-step-ahead conditional vol (decimal) aligned so that
    sigma.loc[t] uses data through t-1, and standardised residuals r_t / sigma_t.
    """
    sig, z = {}, {}
    for c in returns.columns:
        r = returns[c].dropna() * 100.0
        res = arch_model(r, vol="Garch", p=1, q=1, dist="t", mean="Constant").fit(disp="off")
        s = res.conditional_volatility / 100.0  # arch: sigma_t is a function of info through t-1
        sig[c] = s
        z[c] = (returns[c].reindex(s.index) - res.params["mu"] / 100.0) / s
    return pd.DataFrame(sig), pd.DataFrame(z)


def _dcc_recursion(z: np.ndarray, a: float, b: float, qbar: np.ndarray) -> np.ndarray:
    T, n = z.shape
    R = np.empty((T, n, n))
    Q = qbar.copy()
    for t in range(T):
        if t > 0:
            zz = np.outer(z[t - 1], z[t - 1])
            Q = (1 - a - b) * qbar + a * zz + b * Q
        d = 1.0 / np.sqrt(np.diag(Q))
        R[t] = Q * np.outer(d, d)
    return R


def _neg_loglik(params: np.ndarray, z: np.ndarray, qbar: np.ndarray) -> float:
    a, b = params
    if a < 0 or b < 0 or a + b >= 0.999:
        return 1e10
    R = _dcc_recursion(z, a, b, qbar)
    sign, logdet = np.linalg.slogdet(R)
    if np.any(sign <= 0):
        return 1e10
    quad = np.einsum("ti,tij,tj->t", z, np.linalg.inv(R), z)
    return float(0.5 * np.sum(logdet + quad))


def fit_dcc(z: pd.DataFrame) -> dict:
    """Estimate DCC(1,1) by quasi-ML. Returns dict with a, b and R (T x n x n)."""
    zz = z.dropna().values
    qbar = np.corrcoef(zz.T)
    best = None
    for a0, b0 in [(0.02, 0.95), (0.05, 0.90), (0.01, 0.98)]:
        res = minimize(_neg_loglik, x0=[a0, b0], args=(zz, qbar), method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
        if best is None or res.fun < best.fun:
            best = res
    a, b = best.x
    return {"a": float(a), "b": float(b), "R": _dcc_recursion(zz, a, b, qbar),
            "index": z.dropna().index, "columns": list(z.columns), "loglik": -float(best.fun)}


def dynamic_component_var(sigma: pd.DataFrame, dcc: dict, weights: np.ndarray,
                          crit: float) -> tuple[pd.DataFrame, pd.Series]:
    """Daily Component VaR (decimal loss, columns = assets) and portfolio VaR.

    crit is the (negative) standardised quantile used for VaR, e.g. norm.ppf(0.01).
    """
    idx = dcc["index"]
    S = sigma.reindex(idx)[dcc["columns"]].values
    comp = np.empty_like(S)
    pvar = np.empty(len(idx))
    for t in range(len(idx)):
        D = np.diag(S[t])
        cov = D @ dcc["R"][t] @ D
        sw = cov @ weights
        sp = np.sqrt(weights @ sw)
        comp[t] = -crit * weights * sw / sp
        pvar[t] = -crit * sp
    return pd.DataFrame(comp, index=idx, columns=dcc["columns"]), pd.Series(pvar, index=idx, name="DCC VaR")
