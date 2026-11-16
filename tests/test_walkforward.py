"""The backtest must be out of sample, and the tail estimators must return known answers."""
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import walkforward as wf

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def returns() -> pd.Series:
    R = pd.read_csv("data/snapshot/asset_returns.csv", index_col="date", parse_dates=True)
    return R.mean(axis=1).loc["1998-01-01":"2002-12-31"]


@pytest.fixture(scope="module")
def cfg() -> wf.Config:
    return wf.Config(first_day="2001-01-02", refit=63)


def test_forecasts_do_not_change_when_the_future_does(returns, cfg):
    """Altering returns from a date on must leave every earlier forecast untouched."""
    var, es = wf.walk_forward(returns, cfg)
    cut = pd.Timestamp("2002-01-02")
    changed = returns.copy()
    changed.loc[cut:] = changed.loc[cut:] * 4.0
    var2, es2 = wf.walk_forward(changed, cfg)
    before = var.index[var.index <= cut]          # the forecast for the cut day itself uses only earlier data
    pd.testing.assert_frame_equal(var.loc[before], var2.loc[before])
    pd.testing.assert_frame_equal(es.loc[before], es2.loc[before])
    assert (var2.loc["2002-06-01":, "GARCH-t"] > var.loc["2002-06-01":, "GARCH-t"]).all()


def test_the_in_sample_shortcut_does_look_ahead(returns, cfg):
    """The contrast case: one fit on the whole sample reacts to later data."""
    changed = returns.copy()
    changed.loc["2002-01-02":] = changed.loc["2002-01-02":] * 4.0
    a, b = wf.fit_once(returns, cfg), wf.fit_once(changed, cfg)
    assert not np.allclose(a.loc[:"2001-12-31", "GJR-EVT"], b.loc[:"2001-12-31", "GJR-EVT"])


def test_every_model_gives_a_positive_var_and_a_larger_es(returns, cfg):
    var, es = wf.walk_forward(returns, cfg)
    assert list(var.columns) == wf.MODELS and var.index[0] == pd.Timestamp("2001-01-02")
    assert var.notna().all().all() and (var > 0).all().all()
    for m in wf.ES_MODELS:
        assert (es[m] > var[m]).all()


def test_ewma_uses_yesterday():
    r = pd.Series([0.01, -0.02, 0.03, 0.0])
    s = wf.ewma_sigma(r, 0.94)
    assert np.isnan(s.iloc[0]) and s.iloc[1] == pytest.approx(0.01)
    assert s.iloc[2] == pytest.approx(np.sqrt(0.94 * 0.01**2 + 0.06 * 0.02**2))


def test_empirical_and_evt_tails_recover_a_known_distribution():
    z = stats.t.rvs(5, size=400000, random_state=1) * np.sqrt(3 / 5)      # unit variance Student-t
    true_q = -stats.t.ppf(0.01, 5) * np.sqrt(3 / 5)
    q_emp, es_emp = wf.empirical_tail(z, 0.01)
    q_evt, es_evt = wf.evt_tail(z, 0.01, 0.90)
    assert q_emp == pytest.approx(true_q, rel=0.02) and q_evt == pytest.approx(true_q, rel=0.03)
    tail = -z[-z >= true_q]
    assert es_emp == pytest.approx(tail.mean(), rel=0.02) and es_evt == pytest.approx(tail.mean(), rel=0.05)
    assert es_evt > q_evt


def test_basel_zones():
    hits = pd.Series([False] * 500)
    assert wf.basel_zones(hits) == {"green": 1.0, "yellow": 0.0, "red": 0.0, "max_exceptions": 0}
    hits.iloc[300:312] = True
    z = wf.basel_zones(hits)
    assert z["max_exceptions"] == 12 and z["red"] > 0 and z["yellow"] > 0
    assert z["green"] + z["yellow"] + z["red"] == pytest.approx(1.0)
