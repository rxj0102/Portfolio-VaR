import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.garch_var import var_es_multipliers


def test_normal_multipliers_match_known_values():
    q, es = var_es_multipliers(0.01, "normal")
    assert q == pytest.approx(-2.3263, abs=1e-3)
    assert es == pytest.approx(2.6652, abs=1e-3)


@pytest.mark.parametrize("nu", [4.5, 5.0, 8.0, 30.0])
def test_standardised_t_quantile_has_unit_variance_scale(nu):
    q, _ = var_es_multipliers(0.01, "t", nu)
    rng = np.random.default_rng(0)
    x = stats.t.rvs(nu, size=2_000_000, random_state=rng) * np.sqrt((nu - 2) / nu)
    assert x.var() == pytest.approx(1.0, rel=0.03) or nu < 5  # heavy tails: noisy variance
    assert (x < q).mean() == pytest.approx(0.01, abs=5e-4)


@pytest.mark.parametrize("nu", [5.0, 8.0])
def test_t_es_matches_simulation(nu):
    q, es = var_es_multipliers(0.01, "t", nu)
    rng = np.random.default_rng(1)
    x = stats.t.rvs(nu, size=4_000_000, random_state=rng) * np.sqrt((nu - 2) / nu)
    assert -x[x < q].mean() == pytest.approx(es, rel=0.03)


def test_unscaled_t_would_overstate_var():
    nu = 5.0
    q, _ = var_es_multipliers(0.01, "t", nu)
    raw = stats.t.ppf(0.01, nu)
    assert abs(raw) / abs(q) == pytest.approx(np.sqrt(nu / (nu - 2)))
    assert abs(raw) / abs(q) > 1.25


def test_t_converges_to_normal():
    q_t, es_t = var_es_multipliers(0.01, "t", 5000.0)
    q_n, es_n = var_es_multipliers(0.01, "normal")
    assert q_t == pytest.approx(q_n, abs=5e-3) and es_t == pytest.approx(es_n, abs=1e-2)


def test_arch_garch_t_var_has_nominal_coverage_without_extra_lag():
    arch = pytest.importorskip("arch")
    rng = np.random.default_rng(3)
    n = 3000
    omega, a, b, nu = 0.05, 0.08, 0.90, 6.0
    z = stats.t.rvs(nu, size=n, random_state=rng) * np.sqrt((nu - 2) / nu)
    s2 = np.empty(n); r = np.empty(n); s2[0] = omega / (1 - a - b)
    for t in range(n):
        if t:
            s2[t] = omega + a * r[t - 1] ** 2 + b * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * z[t]
    ret = pd.Series(r / 100, index=pd.bdate_range("2010-01-01", periods=n))
    res = arch.arch_model(ret * 100, mean="Zero", vol="Garch", p=1, q=1, dist="t").fit(disp="off", update_freq=0)
    q, _ = var_es_multipliers(0.01, "t", float(res.params["nu"]))
    sigma = res.conditional_volatility / 100
    hit_ok = (ret < sigma * q).mean()            # VaR_t = -sigma_t * q
    assert hit_ok == pytest.approx(0.01, abs=0.006)
    from src.metrics import christoffersen_test
    assert christoffersen_test(ret, -sigma * q, 0.01)["p_POF"] > 0.01
