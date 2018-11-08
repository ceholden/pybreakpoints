""" Test for :py:mod:`pybreakpoints.stats`
"""
import numpy as np
import pytest

from pybreakpoints.stats import core

TOL = 1e-9


def test_mad():
    # value tested against output from R stats::mad
    x = np.random.RandomState(42).randint(0, 100, 100)
    assert (core.mad(x) - 37.064999999999997) < TOL


@pytest.mark.parametrize('n', range(10, 300, 20))
def test_var_std(n):
    x = np.random.RandomState(n).rand(n)
    assert (core._var(x, ddof=1) - np.var(x, ddof=1)) < TOL
    assert (core._std(x, ddof=1) - np.std(x, ddof=1)) < TOL
