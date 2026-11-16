"""Out-of-sample VaR forecasts.

A VaR backtest only means something if the forecast for day t uses nothing from day t or later.
Here every model is re-estimated every REFIT days on all returns before the refit day, and its
parameters, residual quantiles and tail fits are then held fixed until the next refit. The
volatility for day t is the model's one-step forecast from returns up to day t-1.

fit_once() does it the wrong way on purpose: one fit on the whole sample, used to forecast that
same sample. The gap between the two is reported, because it is the most common mistake in
VaR backtests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import genpareto

from src.garch_var import var_es_multipliers

SCALE = 100.0
SPECS = {
    "GARCH-N": dict(vol="GARCH", p=1, o=0, q=1, dist="normal"),
    "GARCH-t": dict(vol="GARCH", p=1, o=0, q=1, dist="t"),
    "GJR-GARCH-t": dict(vol="GARCH", p=1, o=1, q=1, dist="t"),
    "EGARCH-t": dict(vol="EGARCH", p=1, o=1, q=1, dist="t"),
}
# semi-parametric models: a volatility filter, then the empirical or GPD tail of its residuals
FILTERS = {"FHS-GARCH": "GARCH-N", "FHS-GJR": "GJR-GARCH-t", "GJR-EVT": "GJR-GARCH-t"}
MODELS = ["HS", "EWMA-N", *SPECS, "FHS-EWMA", *FILTERS]
ES_MODELS = ["GARCH-t", "GJR-GARCH-t", "EGARCH-t", "FHS-GARCH", "FHS-GJR", "GJR-EVT"]


@dataclass
class Config:
    alpha: float = 0.01
    first_day: str = "2001-01-02"
    refit: int = 21
    hs_window: int = 250
    ewma_lambda: float = 0.94
    evt_threshold: float = 0.90


def ewma_sigma(r: pd.Series, lam: float) -> pd.Series:
    """Volatility forecast for day t from squared returns up to day t-1."""
    return np.sqrt((r**2).ewm(alpha=1 - lam, adjust=False).mean().shift(1))


def evt_tail(z: np.ndarray, alpha: float, threshold: float) -> tuple[float, float]:
    """Quantile and expected shortfall of the loss tail of standardised residuals (both positive).

    Peaks over threshold: a generalised Pareto distribution fitted to losses beyond the
    threshold quantile (McNeil and Frey 2000).
    """
    loss = -z[np.isfinite(z)]
    u = float(np.quantile(loss, threshold))
    exceed = loss[loss > u] - u
    xi, _, beta = genpareto.fit(exceed, floc=0)
    p_u = len(exceed) / len(loss)
    q = u + beta / xi * ((alpha / p_u) ** (-xi) - 1) if abs(xi) > 1e-8 else u - beta * np.log(alpha / p_u)
    es = (q + beta - xi * u) / (1 - xi)
    return float(q), float(es)


def empirical_tail(z: np.ndarray, alpha: float) -> tuple[float, float]:
    """Empirical quantile and expected shortfall of the loss tail (both positive)."""
    loss = -z[np.isfinite(z)]
    q = float(np.quantile(loss, 1 - alpha))
    return q, float(loss[loss >= q].mean())


def _fit(r: np.ndarray, spec: dict):
    return arch_model(r * SCALE, mean="Zero", rescale=False, **spec).fit(disp="off", show_warning=False)


def _sigma(r: np.ndarray, spec: dict, params: pd.Series) -> np.ndarray:
    """One-step volatility path for fixed parameters; entry t uses returns before t."""
    fixed = arch_model(r * SCALE, mean="Zero", rescale=False, **spec).fix(params)
    return np.asarray(fixed.conditional_volatility) / SCALE


def _block(r: np.ndarray, train_end: int, stop: int, cfg: Config) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """VaR and ES for days train_end .. stop-1 from models estimated on days before train_end."""
    out = {}
    sig, resid = {}, {}
    for name, spec in SPECS.items():
        res = _fit(r[:train_end], spec)
        s = _sigma(r[:stop], spec, res.params)
        sig[name], resid[name] = s, r[:train_end] / s[:train_end]
        nu = float(res.params["nu"]) if spec["dist"] == "t" else None
        q, es = var_es_multipliers(cfg.alpha, spec["dist"], nu)
        out[name] = (-q * s[train_end:stop], es * s[train_end:stop])
    for name, base in FILTERS.items():
        tail = evt_tail(resid[base], cfg.alpha, cfg.evt_threshold) if name.endswith("EVT") \
            else empirical_tail(resid[base], cfg.alpha)
        s = sig[base][train_end:stop]
        out[name] = (tail[0] * s, tail[1] * s)
    return out


def walk_forward(r: pd.Series, cfg: Config, progress: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-sample VaR and ES (positive numbers) for every model from cfg.first_day on."""
    x = r.to_numpy()
    start = int(r.index.searchsorted(pd.Timestamp(cfg.first_day)))
    var = pd.DataFrame(index=r.index[start:], columns=MODELS, dtype=float)
    es = pd.DataFrame(index=r.index[start:], columns=MODELS, dtype=float)
    for k in range(start, len(x), cfg.refit):
        stop = min(k + cfg.refit, len(x))
        for name, (v, e) in _block(x, k, stop, cfg).items():
            var.iloc[k - start: stop - start, var.columns.get_loc(name)] = v
            es.iloc[k - start: stop - start, es.columns.get_loc(name)] = e
        if progress and ((k - start) // cfg.refit) % 50 == 0:
            print(f"  refit at {r.index[k].date()}", flush=True)
    # models without estimated parameters
    q_n, es_n = var_es_multipliers(cfg.alpha, "normal")
    s = ewma_sigma(r, cfg.ewma_lambda)
    var["EWMA-N"], es["EWMA-N"] = (-q_n * s)[start:], (es_n * s)[start:]
    hs = -r.rolling(cfg.hs_window).quantile(cfg.alpha).shift(1)
    var["HS"] = hs[start:]
    es["HS"] = (-r.rolling(cfg.hs_window).apply(lambda w: w[w <= np.quantile(w, cfg.alpha)].mean(), raw=True)
                .shift(1))[start:]
    # FHS with the EWMA filter: rolling empirical quantile of past standardised returns
    z = r / s
    zq = -z.rolling(cfg.hs_window).quantile(cfg.alpha).shift(1)
    var["FHS-EWMA"] = (zq * s)[start:]
    es["FHS-EWMA"] = np.nan
    return var, es


def fit_once(r: pd.Series, cfg: Config) -> pd.DataFrame:
    """The in-sample shortcut: fit each model once on the whole sample and forecast that sample."""
    x = r.to_numpy()
    start = int(r.index.searchsorted(pd.Timestamp(cfg.first_day)))
    out = pd.DataFrame(index=r.index[start:])
    sig, resid = {}, {}
    for name, spec in SPECS.items():
        res = _fit(x, spec)
        s = np.asarray(res.conditional_volatility) / SCALE
        sig[name], resid[name] = s, x / s
        nu = float(res.params["nu"]) if spec["dist"] == "t" else None
        out[name] = (-var_es_multipliers(cfg.alpha, spec["dist"], nu)[0] * s)[start:]
    for name, base in FILTERS.items():
        tail = evt_tail(resid[base], cfg.alpha, cfg.evt_threshold) if name.endswith("EVT") \
            else empirical_tail(resid[base], cfg.alpha)
        out[name] = (tail[0] * sig[base])[start:]
    return out


def basel_zones(hits: pd.Series, window: int = 250) -> dict[str, float]:
    """Share of days in each Basel traffic-light zone: exceptions in the last 250 days.

    Green 0 to 4, yellow 5 to 9, red 10 or more (Basel Committee 1996).
    """
    n = hits.astype(float).rolling(window).sum().dropna()
    return {"green": float((n <= 4).mean()), "yellow": float(((n >= 5) & (n <= 9)).mean()),
            "red": float((n >= 10).mean()), "max_exceptions": int(n.max())}
