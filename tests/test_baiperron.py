""" Tests for :py:mod:`pybreakpoints.baiperron`
"""
import numpy as np
import pandas as pd

from pybreakpoints.baiperron import breakpoint


def test_baiperron_nile(nile_river):
    """ Test on Nile river data

    .. code-block:: R

        > breakpoints(Nile ~ 1)

             Optimal 2-segment partition:

        Call:
        breakpoints.formula(formula = Nile ~ 1)

        Breakpoints at observation number: 28

        Corresponding to breakdates: 1898

        > summary(bp_nile)

             Optimal (m+1)-segment partition:

        Call:
        breakpoints.formula(formula = Nile ~ 1)

        Breakpoints at observation number:

        m = 1      28
        m = 2      28       83
        m = 3      28    68 83
        m = 4      28 45 68 83
        m = 5   15 30 45 68 83

        Corresponding to breakdates:

        m = 1        1898
        m = 2        1898           1953
        m = 3        1898      1938 1953
        m = 4        1898 1915 1938 1953
        m = 5   1885 1900 1915 1938 1953

        Fit:

        m   0       1       2       3       4       5
        RSS 2835157 1597457 1552924 1538097 1507888 1659994
        BIC    1318    1270    1276    1285    1292    1311

    """
    X = np.ones_like(nile_river)
    bp_all, bp_rss, bp_bic = breakpoint(X, nile_river)
    bp_win = bp_all[np.argmin(bp_bic)][0]

    # Best model
    assert bp_win == 27  # 1 less because of index on 0
    assert nile_river.index[bp_win] == pd.to_datetime('1898-01-01')

    # BIC
    expected_bic = np.array([1318, 1270, 1276,
                             1285, 1292, 1311])
    assert np.all(np.round(bp_bic) == expected_bic)
    # RSS
    expected_rss = np.array([2835157, 1597457, 1552924,
                             1538097, 1507888, 1659994])
    assert all(np.round(bp_rss) == expected_rss)

    # All the models
    all_breaks = {
        1: [28],
        2: [28, 83],
        3: [28, 68, 83],
        4: [28, 45, 68, 83],
        5: [15, 30, 45, 68, 83]
    }
    for n in all_breaks:
        assert all([expect - 1 == ans for expect, ans in
                    zip(all_breaks[n], bp_all[n])])
