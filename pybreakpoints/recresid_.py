# -*- coding: utf-8 -*-
u""" Recursive residuals computation

Citations:

- Brown, RL, J Durbin, and JM Evans. 1975. Techniques for Testing the
  Consistency of Regression Relationships over Time. Journal of the Royal
  Statistical Society. Series B (Methodological) 37 (2): 149-192.

- Judge George G., William E. Griffiths, R. Carter Hill, Helmut Lütkepohl,
  and Tsoung-Chao Lee. 1985. The theory and practice of econometrics.
  New York: Wiley. ISBN: 978-0-471-89530-5

"""
import numpy as np
import pandas as pd
import xarray as xr

from .core import PANDAS_LIKE
from .compat import jit


@jit(nopython=True, nogil=True)
def _recresid(X, y, span):
    nobs, nvars = X.shape

    recresid_ = np.nan * np.zeros((nobs))
    recvar = np.nan * np.zeros((nobs))

    X0 = X[:span, :]
    y0 = y[:span]

    # Initial fit
    XTX_j = np.linalg.inv(np.dot(X0.T, X0))
    XTY = np.dot(X0.T, y0)
    beta = np.dot(XTX_j, XTY)

    yhat_j = np.dot(X[span - 1, :], beta)
    recresid_[span - 1] = y[span - 1] - yhat_j
    recvar[span - 1] = 1 + np.dot(X[span - 1, :],
                                  np.dot(XTX_j, X[span - 1, :]))
    for j in range(span, nobs):
        x_j = X[j:j+1, :]
        y_j = y[j]

        # Prediction with previous beta
        resid_j = y_j - np.dot(x_j, beta)

        # Update
        XTXx_j = np.dot(XTX_j, x_j.T)
        f_t = 1 + np.dot(x_j, XTXx_j)
        XTX_j = XTX_j - np.dot(XTXx_j, XTXx_j.T) / f_t  # eqn 5.5.15

        beta = beta + (XTXx_j * resid_j / f_t).ravel()  # eqn 5.5.14
        recresid_[j] = resid_j.item()
        recvar[j] = f_t.item()

    return recresid_ / np.sqrt(recvar)


def recresid(X, y, span=None):
    """ Return standardized recursive residuals for y ~ X

    Parameters
    ----------
    X : array like
        2D (n_obs x n_features) design matrix
    y : array like
        1D independent variable
    span : int, optional
        Minimum number of observations for initial regression. If ``span``
        is None, use the number of features in ``X``

    Returns
    -------
    array like
        np.ndarray, pd.Series, or xr.DataArray containing recursive residuals
        standardized by prediction error variance

    Notes
    -----
    For a matrix :math:`X_t` of :math:`T` total observations of :math:`n`
    variables, the :math:`t` th recursive residual is the forecast prediction
    error for :math:`y_t` using a regression fit on the first :math:`t - 1`
    observations. Recursive residuals are scaled and standardized so they are
    N(0, 1) distributed.

    Using notation from Brown, Durbin, and Evans (1975) and Judge, et al
    (1985):

    .. math::
        w_r =
            \\frac{y_r - \\boldsymbol{x}_r^{\prime}\\boldsymbol{b}_{r-1}}
                  {\sqrt{(1 + \\boldsymbol{x}_r^{\prime}
                   S_{r-1}\\boldsymbol{x}_r)}}
            =
            \\frac
                {y_r - \\boldsymbol{x}_r^{\prime}\\boldsymbol{b}_r}
                {\sqrt{1 - \\boldsymbol{x}_r^{\prime}S_r\\boldsymbol{x}_r}}

        r = k + 1, \ldots, T,

    where :math:`S_{r}` is the residual sum of squares after
    fitting the model on :math:`r` observations.

    A quick way of calculating :math:`\\boldsymbol{b}_r` and
    :math:`S_r` is using an update formula (Equations 4 and 5 in
    Brown, Durbin, and Evans; Equation 5.5.14 and 5.5.15 in Judge et al):

    .. math::
        \\boldsymbol{b}_r
            =
            b_{r-1} +
            \\frac
                {S_{r-1}\\boldsymbol{x}_j
                    (y_r - \\boldsymbol{x}_r^{\prime}\\boldsymbol{b}_{r-1})}
                {1 + \\boldsymbol{x}_r^{\prime}S_{r-1}x_r}

    .. math::
        S_r =
            S_{j-1} -
            \\frac{S_{j-1}\\boldsymbol{x}_r\\boldsymbol{x}_r^{\prime}S_{j-1}}
                  {1 + \\boldsymbol{x}_r^{\prime}S_{j-1}\\boldsymbol{x}_r}

    See the recursive residuals implementation that this follows,
    `recursive_olsresiduals`, within the `statsmodels.stats.diagnostic` module.

    """
    if not span:
        span = X.shape[1]
    _X = X.values if isinstance(X, PANDAS_LIKE) else X
    _y = y.values.ravel() if isinstance(y, PANDAS_LIKE) else y.ravel()

    rresid = _recresid(_X, _y, span)[span:]

    if isinstance(y, PANDAS_LIKE):
        if isinstance(y, (pd.Series, pd.DataFrame)):
            rresid = pd.Series(data=rresid,
                               index=y.index[span:],
                               name='recresid')
        elif isinstance(y, xr.DataArray):
            rresid = xr.DataArray(rresid,
                                  coords={'time': y.get_index('time')[span:]},
                                  dims=('time', ),
                                  name='recresid')

    return rresid
