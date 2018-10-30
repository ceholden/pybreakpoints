""" Tests for :py:mod:`pybreakpoints.signatures`
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pybreakpoints import signatures


# ============================================================================
# parse_gufunc_signature
def test_parse_gufunc_signature_1():
    # regression
    in_, out_ = signatures.parse_gufunc_signature('(n, p), (n) -> (p, )')
    assert len(in_) == 2
    assert in_[0] == ('n', 'p', )
    assert in_[1] == ('n', )
    assert len(out_) == 1
    assert out_[0] == ('p', )


def test_parse_gufunc_signature_2():
    # probability prediction
    in_, out_ = signatures.parse_gufunc_signature('(n, p) -> (n, m)')
    assert len(in_) == 1
    assert in_[0] == ('n', 'p', )
    assert len(out_) == 1
    assert out_[0] == ('n', 'm', )


# ============================================================================
# pair_gufunc_indices
@pytest.mark.parametrize('example', [
    pytest.lazy_fixture('sig_ols'),
    pytest.lazy_fixture('sig_proba')
])
def test_pair_gufunc_indices_1(example):
    sig, ins_, out_ = example
    sig_ins, sig_out = signatures.parse_gufunc_signature(sig)

    ans = signatures.pair_gufunc_indices(sig_ins, ins_)
    if 'n' in ans:
        assert ans['n'].size == DIM_N
    if 'p' in ans:
        assert ans['p'].size == DIM_P
    if 'm' in ans:
        assert ans['m'].size == DIM_M

    ans = signatures.pair_gufunc_indices(sig_out, out_)
    if 'n' in ans:
        assert ans['n'].size == DIM_N
    if 'p' in ans:
        assert ans['p'].size == DIM_P
    if 'm' in ans:
        assert ans['m'].size == DIM_M


def test_pair_gufunc_indices_2():
    # test this works with unhandled dimensions
    ans = signatures.pair_gufunc_indices(
        [(), ('n', 'p', ), ('n', )],
        ['do not touch', np.zeros((10, 5)), np.zeros((10, ))]
    )
    assert len(ans) == 2

    ans = signatures.pair_gufunc_indices(
        [(), ('n', 'p', ), ('n', )],
        [np.zeros((5, 5, 5)), np.zeros((10, 5)), np.zeros((10, ))]
    )
    assert len(ans) == 2


def test_pair_gufunc_indices_fail_1():
    # Wrong number of signature items vs data 
    with pytest.raises(ValueError, match=r'Number of items.*do not match'):
        ans = signatures.pair_gufunc_indices(
            [('n', 'p'), ('n', )],
            [np.zeros((100, 5))]
        )


def test_pair_gufunc_indices_fail_2():
    # Wrong number of dimensions for input
    with pytest.raises(ValueError, match=r'Wrong number of dimensions.*'):
        ans = signatures.pair_gufunc_indices(
            [('n', 'p'), ('m', 'n')],
            [np.zeros((100, 5)), np.zeros((100))]
        )


def test_pair_gufunc_indices_fail_3():
    # Inconsistent sizes
    with pytest.raises(ValueError, match=r'Dimension "n" had inconsistent.*'):
        ans = signatures.pair_gufunc_indices(
            [('n', 'p'), ('n', )],
            [np.zeros((100, 5)), np.zeros((99, ))]
        )

# =============================================================================
# wrap_index
@pytest.mark.parametrize('example', [
    pytest.lazy_fixture('sig_ols'),
    pytest.lazy_fixture('sig_proba')
])
def test_wrap_index_1(example):
    # With type promotion
    sig, ins_, outs_ = example

    @signatures.wrap_index(sig, promote_wrapper_type=True)
    def test_f(*args, **kwds):
        for a in args:
            assert not isinstance(a, (pd.DataFrame, pd.Series))
        return outs_

    ans = test_f(*ins_)
    if not isinstance(ans, tuple):
        ans = (ans, )

    for a in ans:
        assert isinstance(a, xr.DataArray)


@pytest.mark.parametrize('example', [
    pytest.lazy_fixture('sig_ols'),
    pytest.lazy_fixture('sig_proba')
])
def test_wrap_index_2(example):
    # Without type promotion
    sig, ins_, outs_ = example

    @signatures.wrap_index(sig, promote_wrapper_type=False)
    def test_f(*args, **kwds):
        for a in args:
            assert not isinstance(a, (pd.DataFrame, pd.Series))
        return outs_

    ans = test_f(*ins_)
    if not isinstance(ans, tuple):
        ans = (ans, )

    for a in ans:
        if a.ndim == 1:
            assert isinstance(a, pd.Series)
        elif a.ndim == 2:
            assert isinstance(a, pd.DataFrame)
        else:
            assert isinstance(a, xr.DataArray)


@pytest.mark.parametrize('example', [
    pytest.lazy_fixture('sig_ols'),
    pytest.lazy_fixture('sig_proba')
])
def test_wrap_index_3(example):
    # With passing any indexed data
    sig, ins_, outs_ = example

    @signatures.wrap_index(sig, promote_wrapper_type=False)
    def test_f(*args, **kwds):
        for a in args:
            assert not isinstance(a, (pd.DataFrame, pd.Series))
        # Remove any indexed output from return value
        # so indexes aren't grabbed from the output
        outs_ndarray = tuple(getattr(o, 'values', o) for o in outs_)
        return outs_ndarray

    ans = test_f(*[getattr(i, 'values', i) for i in ins_])
    if not isinstance(ans, tuple):
        ans = (ans, )
    for a in ans:
        assert isinstance(a, np.ndarray)


# =============================================================================
# Fixtures
DIM_N = 100  # SAMPLES
DIM_P = 5    # FEATURES
DIM_M = 3    # CLASSES


def get_time(n):
    return pd.date_range('2000', '2010', n)


def get_coef(p):
    return ['b_%s' % i for i in range(p)]


def get_class(m):
    return ['cls_%s' % i for i in range(m)]


@pytest.fixture
def sig_ols(request):
    # X, y -> coef
    sig = '(n,p),(n)->(p)'

    time = get_time(DIM_N)
    coef = get_coef(DIM_P)

    X = pd.DataFrame(np.random.random((DIM_N, DIM_P)),
                     index=time, columns=coef)
    y = pd.Series(np.random.random((DIM_N, )), index=time)

    out = pd.Series(np.random.random((DIM_P, )), index=coef)

    return sig, (X, y), (out, )


@pytest.fixture
def sig_proba(request):
    # X -> probabilities
    sig = '(n, p) -> (n, m)'

    time = get_time(DIM_N)
    coef = get_coef(DIM_P)
    class_ = get_class(DIM_M)

    X = pd.DataFrame(np.random.random((DIM_N, DIM_P)),
                     index=time, columns=coef)
    proba = pd.DataFrame(np.random.random((DIM_N, DIM_M)),
                         index=time, columns=class_)
    return sig, (X, ), (proba, )
