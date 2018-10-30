from collections import defaultdict
from functools import singledispatch, wraps
import logging
import re
import string

import numpy as np
import pandas as pd
import xarray as xr

from .compat import (HAS_DASK, da, ddf,
                     HAS_TOOLZ, toolz as tz)
from .utils import register_multi_singledispatch

logger = logging.getLogger(__name__)


TYPES_ARRAY = (np.ndarray, )
TYPES_SERIES = (pd.Series, )
TYPES_DATAFRAME = (pd.DataFrame, )
if HAS_DASK:
    TYPES_ARRAY += (da.Array, )
    TYPES_SERIES += (ddf.Series, )
    TYPES_DATAFRAME += (ddf.DataFrame, )


# Modified to allow comma and spaces after signature (e.g., `(n, )`
# instead of just `(n)`)
# See:
# http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
_DIMENSION_NAME = r'\w+'
_CORE_DIMENSION_LIST = '(?:{0:}(?:,{0:})*)?'.format(_DIMENSION_NAME)
_ARGUMENT = r'\({0},*\s*\)'.format(_CORE_DIMENSION_LIST)
_ARGUMENT_LIST = '{0:}(?:,{0:})*'.format(_ARGUMENT)
_SIGNATURE = '^{0:}->{0:}$'.format(_ARGUMENT_LIST)


def parse_gufunc_signature(signature):
    """ Parse a NumPy gufunc signature

    Parameters
    ----------
    signature : str
        Generalized universal function signature
        (e.g., ``(m, n, ), (n, p, )-> (m, p)``)

    Returns
    -------
    List[Tuple[str, ...]]
        Input core dimensions parsed from signature
    List[Tuple[str, ...]]
        Output core dimensions parsed from signature
    """
    signature = ''.join([c for c in signature if c not in string.whitespace])
    if not re.match(_SIGNATURE, signature):
        raise ValueError(f'Invalid gufunc signature: {signature}')

    return tuple([
        tuple(re.findall(_DIMENSION_NAME, arg))
        for arg in re.findall(_ARGUMENT, arg_list)
    ] for arg_list in signature.split('->'))


def pair_gufunc_indices(signature, data):
    """ Record the index (if any) for GUFunc dimensions

    Parameters
    ----------
    signature : List[Tuple[str, ...]]
        GUFunc input argument signature
        (e.g., ``[('m', 'n', ), ('n', 'p', )]``)
    data : List[array-like]
        Function input or output

    Returns
    -------
    dict[str, pd.Index]
        Mapping of dimension signature label to index

    Raises
    ------
    ValueError
        Raised if data have incorrect or inconsistent shapes
    """
    if len(signature) != len(data):
        raise ValueError('Number of items in signature and data do not match '
                         f'({len(signature)} vs {len(data)})')

    sig_dict = defaultdict(list)
    for sig, dat in zip(signature, data):
        # Only try if signature present
        if sig:
            # get_index returns tuple of dimensions in right order
            idx = get_index(dat)
            if len(idx) != len(sig):
                raise ValueError('Wrong number of dimensions for input '
                                 f'({sig} but got {len(idx)} dimensions)')
            for key, idx_, shp in zip(sig, idx, dat.shape):
                # index could be None for np.ndarray/da.Array
                sig_dict[key].append((idx_, shp))

    sig_dict_ = {}
    for key, idx_size in sig_dict.items():
        indices, sizes = zip(*idx_size)

        sizes_ = set(sizes)
        indices_ = [i for i in indices if i is not None] or [None]

        if len(sizes_) == 1:
            sig_dict_[key] = list(indices_)[0]
        elif len(sizes_) >= 1:
            raise ValueError(f'Dimension "{key}" had inconsistent sizes '
                             f'({", ".join(map(str, sizes_))})')

    return sig_dict_


def wrap_index(signature, promote_wrapper_type=True):
    """ A decorator that handles input argument checks and handles array types

    Parameters
    ----------
    signature : str
        Generalized universal function signature (e.g.,
        ``(m, n, ), (n, p, )-> (m, p)``). Specify an empty
        tuple to ignore/skip dimension checking/handling.
    promote_wrapper_type : bool, optional
        By default, will reapply index using most flexible data container
        available (a :py:class:`xr.DataArray`). If ``False``, will
        wrap 1D outputs as :py:class:`pd.Series`, 2D as
        :py:class:`pd.DataFrame`, and 3D or above as :py:class:`xr.DataArray`

    Returns
    -------
    callable
        Returns a wrapper function
    """
    sig_ins, sig_out = parse_gufunc_signature(signature)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwds):
            # Pair up dimension key with index
            dim_idx_ins = pair_gufunc_indices(sig_ins, args)

            # Remove any index on data
            args_ = [strip_index(arg_) if sig else arg_
                     for arg_, sig in zip(args, sig_ins)]

            # Run
            ans = func(*args_, **kwds)
            if not isinstance(ans, tuple):
                assert len(sig_out) == 1
                ans = (ans, )

            # Repair with any existing on output
            dim_idx_out = pair_gufunc_indices(sig_out, ans)

            # TODO: do we want to merge output index?
            # Merge dimensions - prefer from output
            dim_idx = tz.merge_with(_merge_check, dim_idx_ins, dim_idx_out)

            # Put back on
            ans_ = [reapply_index(a, sig_, dim_idx,
                                  promote_wrapper_type=promote_wrapper_type)
                    if sig_ else a
                    for a, sig_ in zip(ans, sig_out)]

            # Return, unpacked as needed
            if len(sig_out) == 1:
                return ans_[0]
            else:
                return ans_

        return wrapper

    return decorator


def _merge_check(values):
    values_ = [v for v in values if v is not None]
    n = set(len(v) for v in values_)
    if len(n) > 1:
        raise ValueError(f'Values are not same shape {tuple(n)}')
    elif len(n) == 0:
        return values[-1]
    else:
        return list([v for v in values_])[-1]


# ============================================================================
# Array index on/off helper
def reapply_index(obj, signature, indices, promote_wrapper_type=True):
    """ (Re)Apply an index to ``obj``

    Parameters
    ----------
    obj : array-like
        Array-like data to reapply index to
    signature : Tuple[str, ]
        Dimension labels (should correspond to keys in ``indices``)
    indices : dict[str, pd.Index]
        Index per dimension in ``signature``
    promote_wrapper_type : bool, optional
        By default, will reapply index using most flexible data container
        available (a :py:class:`xr.DataArray`). If ``False``, will
        wrap 1D outputs as :py:class:`pd.Series`, 2D as
        :py:class:`pd.DataFrame`, and 3D or above as :py:class:`xr.DataArray`

    Returns
    -------
    np.ndarray, pd.Series, pd.DataFrame, or xr.DataArray
        If ``promote_wrapper_type`` is ``True``, will always return a
        ``xr.DataArray``. Otherwise depends on input dimensions

    Raises
    ------
    ValueError
        Raised if the object is not the correct shape for the indices
        or signature
    """
    ndim, shape = obj.ndim, obj.shape
    if not ndim == len(signature):
        raise ValueError(f'Input data shape does not match signature ({ndim} '
                         f'vs {len(signature)}')

    indices_ = {k: v for k, v in indices.items() if v is not None}
    if not indices_:
        logger.debug('No indices to put back on')
        return obj

    names_ = {k: (v.name if k in indices_ else k) or k
              for k, v in indices.items()}

    if promote_wrapper_type or ndim >= 3:
        dims_ = tuple(names_.get(k, k) for k in signature)
        coords_ = {dim: indices[k] for dim, k in zip(dims_, signature)
                   if k in indices}
        obj_ = xr.DataArray(obj, dims=dims_, coords=coords_)
    else:
        index = indices.get(signature[0], None)
        if ndim == 1:
            obj_ = pd.Series(obj, index=index)
        else:
            columns = indices.get(signature[1], None)
            obj_ = pd.DataFrame(obj, index=index, columns=columns)

    return obj_


@singledispatch
def strip_index(obj):
    """ Remove index an object (if it has one)

    Parameters
    ----------
    arg : array-like argument
        Argument (some kind of array)

    Returns
    -------
    np.ndarray or da.Array
        A NumPy or Dask array (i.e. no index)
    """
    raise TypeError('Only works for Xarray or Pandas datatypes, '
                    f'not "{type(obj)}"')


_TYPES_VALUES = TYPES_ARRAY + TYPES_SERIES + TYPES_DATAFRAME
@register_multi_singledispatch(strip_index, _TYPES_VALUES)
def _strip_index_values(obj):
    return getattr(obj, 'values', obj)


_TYPES_DATA = (xr.DataArray, )
@register_multi_singledispatch(strip_index, _TYPES_DATA)
def _strip_index_data(obj):
    return getattr(obj, 'data', obj)


@singledispatch
def get_index(obj):
    """ Return the index of an array-like object (if any)

    Parameters
    ----------
    obj : np.ndarray, pd.Series, pd.DataFrame, xr.DataArray, dask.array.Array,
          dask.dataframe.Series, dask.dataframe.DataFrame
        Array-like object

    Returns
    -------
    tuple[pd.Index]
        Object's index/indices, if any
    """
    raise TypeError("Only works for NumPy/Pandas/xarray types")


@register_multi_singledispatch(get_index, TYPES_SERIES)
def _get_index_series(obj):
    return (obj.index, )


@register_multi_singledispatch(get_index, TYPES_DATAFRAME)
def _get_index_dataframe(obj):
    return (obj.index, obj.columns)


@get_index.register(xr.DataArray)
def _get_index_xarray(obj):
    idx = tuple(obj.get_index(d) for d in obj.dims)
    for i, d in zip(idx, obj.dims):
        i.name = d
    return idx


@register_multi_singledispatch(get_index, TYPES_ARRAY)
def _get_index_ndarray(obj):
    return tuple(None for _ in obj.shape)
