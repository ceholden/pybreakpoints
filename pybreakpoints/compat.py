""" Various compatibility assists, including hide that Numba is optional
"""
HAS_JIT = True
try:
    from numba import jit
except ImportError:
    HAS_JIT = False
    def jit(*args, **kwds):  # noqa
        def _(func):
            return func
        return _
