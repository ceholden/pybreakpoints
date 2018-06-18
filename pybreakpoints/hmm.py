""" Hidden Markov Model (HMM) algorithms, using Numba (hope you have it)

Note that all these functions are calculated in log space for numerical
stability (avoiding underflow), and each function assumes probabilities
have been passed as log probabilities.

References
----------
https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
"""
import numpy as np

from .compat import jit


@jit(nopython=True, nogil=True)
def logsumexp(a):
    """ Return the log of the sum of exponentials of `a`

    Parameters
    ----------
    a : np.ndarray, 1D
        1 dimensional array

    Returns
    -------
    float
        log(sum(exp(a)))

    See Also
    --------
    scipy.special.logsumexp
        A more feature complete version (without Numba)
    """
    a_max = np.amax(a)
    out = np.log(np.sum(np.exp(a - a_max)))
    out += a_max
    return out


@jit(nopython=True, nogil=True)
def forward(obs_proba, start_proba, trans_proba):
    """ Calculate forward probabilities

    Parameters
    ----------
    obs_proba : np.ndarray, (n_obs, n_states)
        Probability of observing each sample for each state
    start_proba : np.ndarray (n_states)
        Starting probabilities for each state
    trans_proba : np.ndarray (n_states, n_states)
        Transition probabilities among classes

    Returns
    -------
    np.ndarray : forward_lattice (n_obs, n_states)
        Lattice of forward log probabilities
    """
    n_obs, n_states = obs_proba.shape
    fwd = np.zeros((n_obs, n_states))

    fwd[0, :] = start_proba + obs_proba[0, :]
    for i in range(1, n_obs):
        for j in range(n_states):
            previous = fwd[i - 1, :] + trans_proba[:, j]
            fwd[i, j] = logsumexp(previous) + obs_proba[i, j]

    return fwd


@jit(nopython=True, nogil=True)
def backward(obs_proba, start_proba, trans_proba):
    """ Calculate backward probabilities

    Parameters
    ----------
    obs_proba : np.ndarray, (n_obs, n_states)
        Probability of observing each sample for each state
    start_proba : np.ndarray (n_states)
        Starting probabilities for each state
    trans_proba : np.ndarray (n_states, n_states)
        Transition probabilities among classes

    Returns
    -------
    np.ndarray : backward_lattice (n_obs, n_states)
        Lattice of backward log probabilities
    """
    n_obs, n_states = obs_proba.shape
    bwd = np.zeros((n_obs, n_states))

    bwd[n_obs - 1, :] = 0.0
    for i in range(n_obs - 2, -1, -1):
        for j in range(n_states):
            previous = trans_proba[j, :] + obs_proba[i + 1, :] + bwd[i + 1, :]
            bwd[i, j] = logsumexp(previous)

    return bwd


@jit(nopython=True, nogil=True)
def forward_backward(obs_proba, start_proba, trans_proba):
    """ Calculate smoothed (forward-backward) probabilities

    Parameters
    ----------
    obs_proba : np.ndarray, (n_obs, n_states)
        Probability of observing each sample for each state
    start_proba : np.ndarray (n_states)
        Starting probabilities for each state
    trans_proba : np.ndarray (n_states, n_states)
        Transition probabilities among classes

    Returns
    -------
    np.ndarray : smoothed (n_obs, n_states)
        Lattice of smoothed log probabilities
    """
    n_obs, n_states = obs_proba.shape

    fwd = forward(obs_proba, start_proba, trans_proba)
    bwd = backward(obs_proba, start_proba, trans_proba)

    fwd += bwd  # combine
    a_lse = np.empty(n_obs)
    for i in range(n_obs):
        a_lse[i] = logsumexp(fwd[i, :])

    fwd -= a_lse.reshape((-1, 1))

    return fwd


def smoothed_probabilities(obs_proba,
                           start_proba='naive',
                           trans_proba=0.05,
                           bounds=None,
                           log_proba=False):
    """ Smooth a time series of observation probabilities with a HMM

    Probabilities are assumed to be in normal space [0-1] (not log).

    Parameters
    ----------
    obs_proba : np.ndarray, (n_obs, n_states)
        Probability of observing each sample for each state
    start_proba : np.ndarray (n_states) or {'naive'}
        Starting probabilities for each state. The default strategy,
        'naive', sets all classes as equally likely.
    trans_proba : np.ndarray (n_states, n_states), or float
        Transition probabilities among classes. If float is passed,
        sets all off-diagonals (the class transitions) to this number.
    bounds : tuple, or None
        Pass (lower, upper) bounds of the smoothed probabilities. If not
        `None`, will :py:func:`np.clip` the probabilities using these bounds.

    Returns
    -------
    np.ndarray, (n_obs, n_states)
        Smoothed log probabilities
    """
    obs_proba = np.asarray(obs_proba, dtype=float)
    if not log_proba:
        obs_proba = np.log(obs_proba)

    n_obs, n_states = obs_proba.shape

    if start_proba == 'naive':
        start_proba = np.log(np.full(n_states, 1 / n_states))
    else:
        start_proba = np.asarray(start_proba, dtype=float)
        if not log_proba:
            start_proba = np.log(start_proba)
    assert start_proba.shape == (n_states, )

    if isinstance(trans_proba, float):
        trans_proba = np.full((n_states, n_states), trans_proba)
        trans_proba[np.eye(n_states, dtype=bool)] = 1
        trans_proba /= trans_proba.sum(axis=1)
        trans_proba = np.log(trans_proba)
    else:
        trans_proba = np.asarray(trans_proba, dtype=float)
        if not log_proba:
            trans_proba = np.log(trans_proba)
    assert trans_proba.shape == (n_states, n_states)

    if bounds:
        if not log_proba:
            bounds = np.log(bounds)
        obs_proba_ = np.clip(obs_proba, *bounds)
    else:
        obs_proba_ = obs_proba

    smoothed = forward_backward(obs_proba_, start_proba, trans_proba)

    return smoothed
