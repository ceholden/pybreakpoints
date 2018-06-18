""" Bai & Perron test for detecting multiple unknown breakpoints
"""
import logging
import math

import numpy as np

from .stats import BIC
from .recresid_ import recresid

logger = logging.getLogger(__name__)


def breakpoint(X, y, h=0.15, breaks=None):
    """ Bai & Perron breakpoint estimation

    Parameters
    ----------
    X : array-like, shape (n, p)
        Independent variables
    y : array-like, shape (n, )
        Dependent variable
    h : int or float, optional
        Minimum segment size. If float, assumed to be fraction
        of total number of observations
    breaks : int, optional
        Number of breaks to fit. If None, will fit as many as
        possible given ``h`` minimum segment size

    Returns
    -------
    dict[int, np.ndarray]
        Locations of detected breaks for all possible numbers of breaks
        (1 - ``breaks``)
    np.ndarray
        Total RSS for each number of fitted breaks
    np.ndarray
        Bayesian Information Criterion (BIC) for each number of fitted
        breaks (lower is better)

    TODO
    ----
    * Dispatch / wrapper on X/y for pandas/xarray dtypes so we can
      translate the output break indices into index (i.e., date) values
    * Add references
    """
    n, k = X.shape

    if h is None:
        h = k + 1
    elif isinstance(h, float):
        h = int(math.floor(n * h))
    if h <= k:
        raise ValueError('Minimum segment size `h` must be greater than '
                         'number of regressors')
    if h > math.floor(n / 2):
        raise ValueError('Minimum segment size `h` must be smaller than '
                         'half the number of observations')

    breaks_default = int(math.ceil(n / h) - 2)
    if breaks is None:
        breaks = breaks_default
    else:
        if breaks > breaks_default:
            breaks0 = breaks
            breaks = breaks_default
            logger.debug('Number of breaks requested {0} is too large, '
                         'reducing to {1}'.format(breaks0, breaks))

    X_ = np.asarray(X, dtype=np.float)
    y_ = np.asarray(y, dtype=np.float)

    # Calculate RSS for all possible points
    RSS_tri = np.full((n - h + 1, n), np.nan)
    for i in range(n - h + 1):
        ssr = recresid(X_[i:n, :], y_[i:n])
        RSS_tri[i, (i + k):] = np.nancumsum(ssr**2)

    # Find breaks
    index = np.arange(h - 1, n - h)
    RSS_table = np.full((index.size, breaks), np.nan)
    RSS_index = np.full((index.size, breaks), np.nan)

    RSS_table[:, 0] = RSS_tri[0, index]
    RSS_index[:, 0] = index

    # Loop over possible breaks
    for m in range(1, breaks):
        # Indexes for consideration
        idx = np.arange((m + 1) * h - 1, n - h)
        for idx_ in idx:
            pot_index = np.arange(m * h - 1, idx_ - h + 1)
            break_RSS = (RSS_table[pot_index - h + 1, m - 1] +
                         RSS_tri[pot_index + 1, idx_])
            bp = np.argmin(break_RSS)

            RSS_table[idx_ - h + 1, m] = break_RSS[bp]
            RSS_index[idx_ - h + 1, m] = pot_index[bp]

    # Find breaks
    # bp = extract_breaks(RSS_tri, RSS_table, RSS_index, breaks, h)
    bp_all = {
        breaks_: _extract_breaks(RSS_tri, RSS_table, RSS_index, breaks_, h)
        for breaks_ in range(breaks, 0, -1)
    }

    # Find RSS and calculate BIC
    bp_RSS = np.full((breaks + 1), np.nan)
    bp_BIC = np.full((breaks + 1), np.nan)

    # 0 breaks
    bp_RSS[0] = _breaks_RSS(RSS_tri, n, [])
    loglik, k_ = _log_likelihood(bp_RSS[0], n, k, 0)
    bp_BIC[0] = BIC(loglik, k_, n)

    # 1 - `breaks`
    for break_, bp_ in bp_all.items():
        bp_RSS[break_] = _breaks_RSS(RSS_tri, n, bp_)
        loglik, k_ = _log_likelihood(bp_RSS[break_], n, k, break_)
        bp_BIC[break_] = BIC(loglik, k_, n)

    return bp_all, bp_RSS, bp_BIC


def _extract_breaks(RSS_tri, RSS_table, RSS_index, breaks, h):
    index = RSS_index[:, 0].astype(np.int)
    break_RSS = RSS_table[index - h + 1, breaks - 1] + RSS_tri[index + 1, -1]

    bp = [int(index[np.nanargmin(break_RSS)])]

    if breaks > 1:
        for m in range(breaks - 1, 0, -1):
            bp_ = RSS_index[bp[0] - h + 1, m]
            bp.insert(0, int(bp_))

    return np.array(bp)


def _breaks_RSS(RSS_tri, n, bp):
    """ Calculate RSS for series given `bp` break locations
    """
    points = np.concatenate(([-1], bp, [n - 1])).astype(np.int)
    return np.sum([
        RSS_tri[points[i] + 1, points[i + 1]] for i in range(len(points) - 1)
    ])


def _log_likelihood(rss, n, k, breaks):
    """ Return log-likelihood and degrees of freedom for breakpoint model

    Parameters
    ----------
    rss : float
        RSS
    n : int
        Number of observations
    k : int
        Number of regressors
    breaks : int
        Number of breaks fit

    Returns
    -------
    tuple (float, int)
        Log-likelihood and number of parameters estimated
    """
    k_ = (k + 1) * (breaks + 1)
    ll = -0.5 * n * (np.log(rss) + 1 - np.log(n) + np.log(2 * np.pi))
    return ll, k_
