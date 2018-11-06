""" Commonly used, basic statistics functions
"""
import math

import numpy as np

from .compat import HAS_JIT, jit


@jit(nopython=True, nogil=True)
def mad(x, c=1.4826):
    """ Calculate Median-Absolute-Deviation (MAD) of x

    Parameters
    ----------
    x : array-like
        1D array
    c : float
        Scale factor to get to ~standard normal (default: 1.4826)
        (i.e. 1 / 0.75iCDF ~= 1.4826)

    Returns
    -------
    float
        MAD 'robust' variance estimate

    Reference
    ---------
        http://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    return np.median(np.fabs(x - np.median(x))) * 1.4826


# Numba-compatible (if we have it) var/std with ddof != 0
@jit(nopython=True, nogil=True)
def _var(x, ddof=0):
    """ Calculate variance for an array

    Uses Welford's algorithm[1] for online variance calculation.

    Parameters
    ----------
    x : array-like
        The data
    ddof : int
        Degrees of freedom

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
    """
    n = 0.0
    mean_ = M2 = 0.0

    for x_ in x:
        n += 1
        delta = x_ - mean_
        mean_ += delta / n
        delta2 = x_ - mean_
        M2 += delta * delta2
    return M2 / max(n - ddof, 0)


@jit(nopython=True, nogil=True)
def _std(x, ddof=0):
    return _var(x, ddof=ddof) ** 0.5


# ONLY USE THESE IF WE HAVE NUMBA
if HAS_JIT:
    var = _var
    std = _std
else:
    var = np.var
    std = np.std


# Model criteria
@jit(nopython=True, nogil=True)
def rmse(resid):
    """ Calculate RMSE from either y and yhat or residuals

    Parameters
    ----------
    resid : np.ndarray
        Residuals

    Returns
    -------
    float
        Root mean squared error
    """
    return ((resid ** 2).sum() / y.size) ** 0.5


# INFORMATION CRITERION
@jit(nopython=True, nogil=True)
def AIC(loglik, k):
    """ Akaike Information Criterion

    Parameters
    ----------
    loglik : float
        Log likelihood
    k : int
        Parameters estimated

    Returns
    -------
    float
        AIC, the smaller the better
    """
    return -2 * loglik + 2 * k


@jit(nopython=True, nogil=True)
def BIC(loglik, k, n):
    """ Bayesian Information Criterion

    Parameters
    ----------
    loglik : float
        Log likelihood
    k : int
        Parameters estimated
    n : int
        Number of observations

    Parameters
    ----------
    float
        BIC, the smaller the better
    """
    return -2 * loglik + math.log(n) * k
