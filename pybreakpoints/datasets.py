import random
import datetime as dt

import numpy as np
import xarray as xr


def make_xarray(shape=(100, 10, 10), max_breaks=4,
                begin=dt.datetime(2000, 1, 1),
                delta=10):
    def ts_prep(x, max_breaks):
        nbreaks = random.randint(0, max_breaks)
        if nbreaks == 0:
            return x
        breakpos = sorted(random.sample(range(x.shape[0]), nbreaks))
        shifts = [random.uniform(-1, 1) for x in breakpos]
        for t in zip(breakpos, shifts):
            x[t[0]:] = x[t[0]:] + t[1]
        return x

    arr = np.random.rand(np.prod(shape)).reshape(shape).astype(np.float64)
    arr = np.apply_along_axis(ts_prep, 0, arr, max_breaks)
    dates_list = [begin + dt.timedelta(d*delta) for d in range(shape[0])]
    xarr = xr.DataArray(arr, dims=['time', 'x', 'y'],
                        coords={'time': dates_list,
                                'x': np.arange(shape[2]),
                                'y': np.arange(shape[1])},
                        name = 'pybreakpoints_test')
    return xarr

