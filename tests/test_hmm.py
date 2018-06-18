""" Tests for :py:mod:`plants_change.stats.hmm`
"""
import numpy as np
import pytest
from scipy.special import logsumexp

from pybreakpoints import hmm


def test__logsumexp():
    a = np.random.rand(100)
    assert (hmm.logsumexp(a) - logsumexp(a)) < 1e-9


@pytest.fixture
def weather_data():
    obs_string = ['walk', 'shop', 'walk', 'clean']
    obs_coding = {
        'walk': 0,
        'shop': 1,
        'clean': 2
    }

    obs = [obs_coding[s] for s in obs_string]

    states = ['rain', 'sun']
    trans_proba = np.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])
    start_proba = np.array([0.6, 0.4])
    emissions = np.array([
        [0.1, 0.6],
        [0.4, 0.3],
        [0.5, 0.1]
    ])
    obs_proba = np.stack([
        emissions[o, :] for o in obs
    ])
    return {
        'states': states,
        'obs': obs,
        'obs_proba': obs_proba,
        'trans_proba': trans_proba,
        'start_proba': start_proba
    }


def test_fwd_bwd(weather_data):
    args = (
        np.log(weather_data['obs_proba']),
        np.log(weather_data['start_proba']),
        np.log(weather_data['trans_proba']),
    )

    # Tested against R's HMM
    fwd = np.array([
        [-2.813410717, -2.896792326, -5.148519001, -4.89029553],
        [-1.427116356, -3.024131748, -3.596045064, -6.30883087]
    ]).T
    bwd = np.array([
        [-3.495156475, -2.611831343, -0.9675840263, 0],
        [-3.463243013, -2.218243945, -1.3470736480, 0]
    ]).T
    posterior = np.array([
        [0.1949426941, 0.4338284422, 0.2363159788, 0.8051087012],
        [0.8050573059, 0.5661715578, 0.7636840212, 0.1948912988]
    ]).T

    fwd_ = hmm.forward(*args)
    bwd_ = hmm.backward(*args)
    posterior_ = np.exp(hmm.forward_backward(*args))

    np.testing.assert_allclose(fwd, fwd_)
    np.testing.assert_allclose(bwd, bwd_)
    np.testing.assert_allclose(posterior, posterior_)
