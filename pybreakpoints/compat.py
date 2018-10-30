""" Various compatibility assists, including hide that Numba is optional
"""
# Numba
HAS_JIT = True
try:
    from numba import jit
except ImportError:
    HAS_JIT = False
    def jit(*args, **kwds):  # noqa
        def _(func):
            return func
        return _


# Dask
HAS_DASK = True
try:
    import dask.array as da
    import dask.dataframe as ddf
except ImportError:
    HAS_DASK = False
    da, ddf = None, None


# Toolz
HAS_TOOLZ = True
try:
    import cytoolz as toolz
except ImportError:
    try:
        import toolz
    except ImportError:
        HAS_TOOLZ = False
        toolz = None
