from collections import namedtuple

import pandas as pd
import xarray as xr

PANDAS_LIKE = (pd.DataFrame, pd.Series, xr.DataArray)

# TODO: some sort of wrapper for removing Pandas/xarray index for analysis,
#       and putting it back on for the return results

#: namedtuple: Structural break detection results
StructuralBreakResult = namedtuple('StructuralBreakResult', [
    'method',
    'index',
    'score',
    'process',
    'boundary',
    'pvalue',
    'signif'
])
