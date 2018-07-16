import xarray as xr
import numpy as np
import numba as nb

from .baiperron import breakpoint


def bp_wrapper(X, y, **kwargs):
    """ Bai & Perron breakpoint estimation

    Runs Bai & Perron breakpoint estimation in a single ``np.array``

    Parameters
    ----------
    X : array-like, shape (n, p)
        Independent variables
    y : array-like, shape (n, )
        Dependent variable
    **kwargs
        Additional arguments passed to ``baiperron.breakpoint``

    Returns
    -------
    np.ndarray
        array of shape (n) with values corresponding to optimal segmentation
        segment number
    """
    loc, _, bic = breakpoint(X, y, **kwargs)
    nseg = np.argmin(bic)
    out = np.zeros_like(y)
    if nseg > 0:
        for i,num in enumerate(loc[nseg], start=1):
            out[num+1:] = i
    return out


#TODO: How to pass additional arguments to guvectorize?
@nb.guvectorize(["void(float64[:], float64[:,:], uint8[:])"],
                 "(n),(n,m)->(n)")
def _bp_ufunc(y, X, out):
    out[:] = bp_wrapper(X, y)


def _breakpoint_xarray(X, y, dim, **kwargs):
    """
    Example
    -------
    >>> from pybreapoints.datasets import make_xarray
    >>> from pybreapoints.wrappers import _breakpoint_xarray
    >>> import numpy as np

    >>> xarr = make_xarray()
    >>> # Run breakpoint with intercept
    >>> X = np.ones_like(xarr.time).astype(np.float64)
    >>> X = np.expand_dims(X, 1)
    >>> xarr_out = _breakpoint_xarray(X, xarr, 'time')
    """
    out = xr.apply_ufunc(_bp_ufunc,
                         y, X,
                         input_core_dims=[[dim], []],
                         output_core_dims=[['time']])
    return out


def breakpoint_wrapper(X, y, dim, **kwargs):
    if isinstance(y, xr.DataArray):
        return _breakpoint_xarray(X, y, dim=dim, **kwargs)
    else:
        raise NotImplemented('Unsuported type for y')
