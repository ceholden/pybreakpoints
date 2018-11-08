""" Least squares models using Numba

TODO:
    * Hide (with "_" prefix) JIT'd functions?

"""
import numba as nb
import numpy as np
from scipy import stats


@nb.njit
def scale(wresid, df):
    return (wresid @ wresid) / df


@nb.njit
def normalized_cov(X):
    pinv = np.linalg.pinv(X)
    return pinv @ pinv.T


@nb.njit
def params_cov(X, resid, df):
    scale_ = scale(resid, df)
    return normalized_cov(X) * scale_


@nb.njit
def params_stderr(params_cov):
    return np.sqrt(np.diag(params_cov))


@nb.njit
def tvalues(params, stderr):
    return params / stderr


def pvalues(params, stderr, df, use_t=True):
    tvalues_ = np.abs(tvalues(params, stderr))
    if use_t:
        return stats.t.sf(tvalues_, df) * 2
    else:
        return stats.norm.sf(tvalues_) * 2


@nb.njit
def _lstsq_model(X, y):
    coef, sse, rank, s = np.linalg.lstsq(X, y)
    resid = y - (X @ coef)

    df = y.size - rank

    p_cov = params_cov(X, resid, df)
    p_stderr = params_stderr(p_cov)
    p_tval = tvalues(coef, p_stderr)

    return coef, df, resid, p_stderr, p_tval


# TODO: we can use the index on/off wrapper here
def lstsq_model(X, y, use_t=False):
    """ Fit a least squares regression model and return model info

    Parameters
    ----------
    X : np.ndarray, (n, p)
        Independent features
    y : np.ndarray (n, )
        Dependent variable
    use_t : bool, optional
        Use Student's t distribution instead of standard normal when calculating
        p-values

    Returns
    -------
    coef : np.ndarray, (p, )
        Coefficient estimates
    resid : np.ndarray, (n, )
        Residuals
    stderr : np.ndarray, (p, )
        Coefficient estimate standard errors
    tval : np.ndarray, (p, )
        Coefficient estimate t-values
    pval : np.ndarray, (p, )
        Coefficient estimate p-values

    See Also
    --------
    np.linalg.lstsq
    """
    X_ = getattr(X, 'values', X)
    y_ = getattr(y, 'values', y)

    coef, df, resid, stderr, tval = _lstsq_model(X_, y_)
    pval = pvalues(coef, stderr, df, use_t=use_t)

    return coef, resid, stderr, tval, pval
