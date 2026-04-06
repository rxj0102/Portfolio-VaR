"""
test_metrics.py — Unit tests for src/metrics.py.

Covers: christoffersen_test, mcneil_frey_es_test,
        model_confidence_set, run_backtest.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from metrics import (
    christoffersen_test,
    mcneil_frey_es_test,
    model_confidence_set,
    run_backtest,
    _tick_loss,
)

# ── Shared fixtures ───────────────────────────────────────────────────────────

RNG = np.random.default_rng(0)
ALPHA = 0.01
N = 2000  # enough observations for stable statistics


def _returns(n: int = N, sigma: float = 0.01) -> pd.Series:
    return pd.Series(RNG.normal(0, sigma, n))


def _var_series(value: float, n: int = N) -> pd.Series:
    return pd.Series(np.full(n, value))


# ── christoffersen_test ───────────────────────────────────────────────────────

class TestChristoffersenTest:

    def test_output_keys(self):
        result = christoffersen_test(_returns(), _var_series(0.02), alpha=ALPHA)
        assert set(result.keys()) == {"n_violations", "violation_rate", "p_POF", "p_IND", "p_CC"}

    def test_violation_rate_in_range(self):
        result = christoffersen_test(_returns(), _var_series(0.015), alpha=ALPHA)
        assert 0.0 <= result["violation_rate"] <= 1.0

    def test_n_violations_consistent_with_rate(self):
        r = _returns()
        result = christoffersen_test(r, _var_series(0.015), alpha=ALPHA)
        assert result["n_violations"] == int(round(result["violation_rate"] * N))

    def test_correct_coverage_passes_pof(self):
        """A well-calibrated VaR (using the empirical quantile) should not
        reject the POF test at 5% significance."""
        r = _returns(5000)
        threshold = float(np.quantile(-r, 1.0 - ALPHA))
        var = pd.Series(np.full(len(r), threshold))
        result = christoffersen_test(r, var, alpha=ALPHA)
        assert result["p_POF"] > 0.05

    def test_zero_var_fails_pof(self):
        """VaR near zero is violated almost every day — POF must reject."""
        result = christoffersen_test(_returns(), _var_series(1e-6), alpha=ALPHA)
        assert result["p_POF"] < 0.05

    def test_very_large_var_fails_pof(self):
        """VaR so large it is never breached — POF must reject."""
        result = christoffersen_test(_returns(), _var_series(1.0), alpha=ALPHA)
        # pi_hat == 0, special case: should return nan or < 0.05
        p = result["p_POF"]
        assert np.isnan(p) or p < 0.05

    def test_index_alignment(self):
        """Test handles partial index overlap without error."""
        r = pd.Series(RNG.normal(0, 0.01, 100), index=pd.date_range("2020", periods=100))
        v = pd.Series(np.full(80, 0.015), index=pd.date_range("2020", periods=80))
        result = christoffersen_test(r, v, alpha=ALPHA)
        assert result["n_violations"] >= 0

    def test_p_values_are_numeric(self):
        result = christoffersen_test(_returns(), _var_series(0.015), alpha=ALPHA)
        for key in ("p_POF", "p_IND", "p_CC"):
            assert isinstance(result[key], float)


# ── mcneil_frey_es_test ───────────────────────────────────────────────────────

class TestMcNeilFreyESTest:

    def test_output_keys(self):
        result = mcneil_frey_es_test(_returns(), _var_series(0.015), _var_series(0.025))
        assert set(result.keys()) == {"n_exceed", "t_stat", "p_val"}

    def test_n_exceed_positive_for_low_var(self):
        """Very low VaR means many exceedances."""
        result = mcneil_frey_es_test(_returns(), _var_series(0.001), _var_series(0.015))
        assert result["n_exceed"] > 0

    def test_no_exceedances_returns_nan(self):
        """VaR so large there are zero exceedances."""
        result = mcneil_frey_es_test(_returns(), _var_series(10.0), _var_series(15.0))
        assert result["n_exceed"] == 0
        assert np.isnan(result["t_stat"])
        assert np.isnan(result["p_val"])

    def test_conservative_es_gives_negative_t(self):
        """When ES greatly overestimates tail losses, t-stat should be negative."""
        r = _returns(1000)
        v = _var_series(0.001, 1000)  # many violations
        es = _var_series(5.0, 1000)   # wildly conservative ES
        result = mcneil_frey_es_test(r, v, es)
        assert result["t_stat"] < 0

    def test_index_alignment(self):
        r = pd.Series(RNG.normal(0, 0.01, 200), index=pd.date_range("2020", periods=200))
        v = pd.Series(np.full(150, 0.005), index=pd.date_range("2020", periods=150))
        es = pd.Series(np.full(180, 0.010), index=pd.date_range("2020", periods=180))
        result = mcneil_frey_es_test(r, v, es)
        assert isinstance(result["n_exceed"], int)


# ── model_confidence_set ──────────────────────────────────────────────────────

class TestModelConfidenceSet:

    def _loss_df(self, n: int = 500, n_models: int = 4) -> pd.DataFrame:
        return pd.DataFrame(
            {f"model_{i}": RNG.exponential(i + 1, n) for i in range(n_models)}
        )

    def test_output_shape(self):
        ldf = self._loss_df()
        result = model_confidence_set(ldf, B=50)
        assert isinstance(result, pd.DataFrame)
        assert "in_MCS" in result.columns
        assert len(result) == ldf.shape[1]

    def test_mcs_values_are_star_or_empty(self):
        result = model_confidence_set(self._loss_df(), B=50)
        assert set(result["in_MCS"].unique()).issubset({"★", ""})

    def test_clearly_best_model_survives(self):
        """Model with much lower losses should always be retained."""
        n = 800
        ldf = pd.DataFrame({
            "best": RNG.exponential(0.1, n),
            "worse": RNG.exponential(5.0, n),
            "worst": RNG.exponential(20.0, n),
        })
        result = model_confidence_set(ldf, B=200)
        assert result.loc["best", "in_MCS"] == "★"

    def test_single_model_always_in_mcs(self):
        ldf = pd.DataFrame({"only": RNG.exponential(1, 200)})
        result = model_confidence_set(ldf, B=50)
        assert result.loc["only", "in_MCS"] == "★"

    def test_two_identical_models_both_survive(self):
        """Models with identical losses should both be in the MCS."""
        x = RNG.exponential(1, 300)
        ldf = pd.DataFrame({"m1": x, "m2": x})
        result = model_confidence_set(ldf, B=50)
        assert result.loc["m1", "in_MCS"] == "★"
        assert result.loc["m2", "in_MCS"] == "★"

    def test_mcs_with_tick_loss_retains_good_model(self):
        """MCS fed with tick losses from _tick_loss must retain the well-calibrated model.

        Regression test for sign-convention bug: tick loss is always non-negative and
        lower = better, so the model with fewer violations should survive elimination.
        """
        rng = np.random.default_rng(99)
        T = 1200
        r = rng.normal(0, 0.01, T)
        # Well-calibrated: ~1 % violation rate at 99 % VaR
        var_good = np.full(T, np.quantile(-r, 0.99))
        # Badly calibrated: VaR far too low → many violations
        var_bad = np.full(T, np.quantile(-r, 0.60))

        ldf = pd.DataFrame({
            "good": _tick_loss(r, var_good, alpha=0.01),
            "bad":  _tick_loss(r, var_bad,  alpha=0.01),
        })
        # Tick loss must be non-negative
        assert (ldf >= 0).all().all(), "Tick losses must be non-negative"
        # Good model must have strictly lower mean tick loss
        assert ldf["good"].mean() < ldf["bad"].mean()
        # MCS must retain the good model
        result = model_confidence_set(ldf, B=200, seed=7)
        assert result.loc["good", "in_MCS"] == "★", "MCS must retain the well-calibrated model"


# ── run_backtest ──────────────────────────────────────────────────────────────

class TestRunBacktest:

    def _var_df(self, n: int = N) -> pd.DataFrame:
        r = _returns(n)
        return pd.DataFrame({
            "Portfolio Returns": r,
            "model_ok": np.full(n, 0.018),   # reasonable VaR
            "model_bad": np.full(n, 0.001),  # far too low
        })

    def test_expected_output_columns(self):
        summary = run_backtest(self._var_df(), run_mcs=False)
        for col in ("Violations", "ViolRate", "p_POF", "p_IND", "p_CC", "status"):
            assert col in summary.columns, f"Missing column: {col}"

    def test_models_are_index(self):
        summary = run_backtest(self._var_df(), run_mcs=False)
        assert "model_ok" in summary.index
        assert "model_bad" in summary.index
        assert "Portfolio Returns" not in summary.index

    def test_bad_model_has_more_violations(self):
        summary = run_backtest(self._var_df(), run_mcs=False)
        assert summary.loc["model_bad", "Violations"] > summary.loc["model_ok", "Violations"]

    def test_bad_model_fails_status(self):
        summary = run_backtest(self._var_df(), run_mcs=False)
        assert summary.loc["model_bad", "status"] == "FAIL"

    def test_mcs_column_present_when_requested(self):
        summary = run_backtest(self._var_df(), run_mcs=True, mcs_B=50)
        assert "in_MCS" in summary.columns

    def test_mcs_column_absent_when_not_requested(self):
        summary = run_backtest(self._var_df(), run_mcs=False)
        assert "in_MCS" not in summary.columns

    def test_violation_rate_consistent_with_violations(self):
        df = self._var_df()
        summary = run_backtest(df, run_mcs=False)
        n = len(df)
        for m in ("model_ok", "model_bad"):
            expected = summary.loc[m, "Violations"] / n
            assert abs(summary.loc[m, "ViolRate"] - expected) < 1e-6

    def test_null_model_skipped(self):
        df = self._var_df()
        df["all_nan"] = np.nan
        summary = run_backtest(df, run_mcs=False)
        assert "all_nan" not in summary.index
