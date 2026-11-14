import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dcc import _dcc_recursion, dynamic_component_var  # noqa: E402


def test_dcc_recursion_gives_valid_correlations():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((300, 3))
    R = _dcc_recursion(z, 0.05, 0.9, np.corrcoef(z.T))
    assert np.allclose(np.diagonal(R, axis1=1, axis2=2), 1.0)
    assert np.all(np.linalg.eigvalsh(R) > 0)


def test_component_var_sums_to_portfolio_var():
    idx = pd.bdate_range("2024-01-01", periods=5)
    sigma = pd.DataFrame({"A": [0.01] * 5, "B": [0.02] * 5}, index=idx)
    R = np.tile(np.array([[1.0, 0.3], [0.3, 1.0]]), (5, 1, 1))
    dcc = {"index": idx, "columns": ["A", "B"], "R": R}
    comp, pvar = dynamic_component_var(sigma, dcc, np.array([0.6, 0.4]), crit=-2.326)
    assert np.allclose(comp.sum(axis=1).values, pvar.values)
    sp = np.sqrt(0.6**2 * 1e-4 + 0.4**2 * 4e-4 + 2 * 0.6 * 0.4 * 0.3 * 0.01 * 0.02)
    assert np.isclose(pvar.iloc[0], 2.326 * sp)
