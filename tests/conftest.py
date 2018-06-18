from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).parent.resolve()


@pytest.fixture(scope='module')
def nile_river():
    filename = str(HERE.joinpath('data', 'nile.csv'))
    nile = pd.read_csv(filename, index_col=0)
    nile.index = pd.date_range('1871', '1970', freq='AS')
    return nile
