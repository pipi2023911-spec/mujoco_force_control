import numpy as np
import pytest

from experiments.pi_gain_design import fit_stiffness


def test_reference_units_and_local_slope():
    x = np.array([.0138, .014, .0142])
    result = fit_stiffness(x, 2000*x-20.)
    assert result["有效刚度_N每m"] == pytest.approx(2000.)
    assert result["截距_N"] == pytest.approx(-20.)
    assert result["拟合最大残差_N"] < 1e-12


def test_unusable_fit_rejected():
    for x, y in [([0, 1], [1, 2]), ([1, 1, 1], [1, 2, 3]), ([0, 1, 2], [1, np.nan, 3])]:
        with pytest.raises(ValueError):
            fit_stiffness(x, y)
