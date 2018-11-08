from pathlib import Path

import pandas as pd
import pytest

try:
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    pandas2ri.activate()
except ImportError:
    has_rpy2 = False
else:
    has_rpy2 = True

HERE = Path(__file__).parent.resolve()


@pytest.fixture('module')
def rpy2_strucchange():
    if has_rpy2:
        try:
            base = importr('base')
            utils = importr('utils')
            has_pkg = base.require('strucchange')[0]
            if not has_pkg:
                utils.install_packages(
                    'strucchange',
                    repos='http://cran.revolutionanalytics.com/'
                )
        except Exception as exc:
            pytest.skip('Unable to install "strucchange"')
        else:
            return True
    return False


@pytest.fixture(scope='module')
def nile_river():
    filename = str(HERE.joinpath('data', 'nile.csv'))
    nile = pd.read_csv(filename, index_col=0)
    nile.index = pd.date_range('1871', '1970', freq='AS')


    return nile
