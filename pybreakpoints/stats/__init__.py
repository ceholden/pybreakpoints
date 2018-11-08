""" Common statistical functions (non-breakpoint oriented)
"""
from . core import (
    mad,
    var,
    std,
    rmse,
    AIC,
    BIC
)
from .lstsq import lstsq_model


__all__ = [
    'mad',
    'var',
    'std',
    'rmse',
    'AIC',
    'BIC',
    'lstsq_model'
]
