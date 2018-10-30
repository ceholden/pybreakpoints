""" Functions, etc useful to project
"""
import functools


def register_multi_singledispatch(func, types):
    """ Register multiple types for singledispatch

    Parameters
    ----------
    func : callable
        Function
    types : tuple
        Multiple types to register

    Returns
    -------
    func : callable
        Decorated function
    """
    if not hasattr(func, 'registry'):
        raise TypeError("Function must be dispatchable (missing "
                        "`func.registry` from wrapping with `singledispatch`)")

    def _wrapper(dispatch_func):
        for type_ in types:
            dispatch_func = func.register(type_, dispatch_func)

        @functools.wraps
        def wrapper(*args, **kwds):
            return dispatch_func(*args, **kwds)
        return wrapper
    return _wrapper
