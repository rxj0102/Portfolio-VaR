"""Parametric VaR / ES multipliers for GARCH models fitted with ``arch``.

``arch`` parameterises ``dist="t"`` as a *standardised* Student-t (unit
variance), and ``conditional_volatility[t]`` is already the one-step-ahead
standard deviation for day ``t`` (a function of information through ``t-1``).
Two consequences that are easy to get wrong:

* the quantile must be the raw Student-t quantile rescaled by
  ``sqrt((nu - 2) / nu)``; using the raw quantile overstates VaR by
  ``sqrt(nu / (nu - 2))`` (about 29% for nu = 5);
* the VaR for day ``t`` uses ``conditional_volatility[t]`` directly - lagging
  it by one more day would use a stale forecast.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def var_es_multipliers(alpha: float, dist: str, nu: float | None = None) -> tuple[float, float]:
    """Return ``(q, es_mult)`` for a unit-variance innovation distribution.

    ``q`` is the alpha-quantile (negative); ``VaR = -sigma * q`` and
    ``ES = sigma * es_mult`` with ``es_mult`` positive.
    """
    if dist == "normal":
        q = float(stats.norm.ppf(alpha))
        return q, float(stats.norm.pdf(q) / alpha)
    if dist == "t":
        if nu is None or nu <= 2:
            raise ValueError("Student-t needs nu > 2")
        scale = np.sqrt((nu - 2.0) / nu)
        q_raw = float(stats.t.ppf(alpha, df=nu))
        es_raw = stats.t.pdf(q_raw, df=nu) / alpha * (nu + q_raw**2) / (nu - 1.0)
        return q_raw * float(scale), float(scale * es_raw)
    raise ValueError(f"unsupported dist {dist!r}")
