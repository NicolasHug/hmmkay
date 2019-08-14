import numbers

from numba import njit
import numpy as np


def _check_array_sums_to_1(a, name="array"):
    a_sum = a.sum()
    if not (1 - 1e-5 < a_sum < 1 + 1e-5):
        err_msg = f"{name} must sum to 1. Got \n{a}.sum() = {a_sum}"
        raise ValueError(err_msg)


def _allocate_or_reuse(array, requested_shape, dtype=np.float):
    # Return array if requested shape is smaller than frame.shape else
    # allocate new array
    if array is None or any(a < b for (a, b) in zip(array.shape, requested_shape)):
        return np.empty(shape=requested_shape, dtype=dtype)
    else:  # reuse
        return array


@njit(cache=True)
def _choice(p):
    """return i with probability p[i]"""
    # inspired from https://github.com/numba/numba/issues/2539
    # p must sum to 1
    return np.searchsorted(np.cumsum(p), np.random.random(), side="right")


@njit(cache=True)
def _logsumexp(a):
    # stolen from pygbm \o/

    a_max = np.amax(a)
    if not np.isfinite(a_max):
        a_max = 0

    s = np.sum(np.exp(a - a_max))
    return np.log(s) + a_max


@njit(cache=True)
def _argmax(a):
    # Apparently much faster than np.argmax in our context
    curr_max = a[0]
    curr_max_idx = 0
    for i in range(1, len(a)):
        if a[i] > curr_max:
            curr_max = a[i]
            curr_max_idx = i
    return curr_max_idx


def _get_hmm_learn_model(hmm):
    """Return equivalent hmm_learn model"""
    try:
        import hmmlearn.hmm  # noqa
    except ImportError as ie:
        raise RuntimeError("Please install hmmlearn to run tests/benchmarks.") from ie

    hmm_learn_model = hmmlearn.hmm.MultinomialHMM(
        n_components=hmm.A.shape[0], init_params="", tol=0, n_iter=hmm.n_iter
    )
    hmm_learn_model.startprob_ = hmm.pi
    hmm_learn_model.transmat_ = hmm.A
    hmm_learn_model.emissionprob_ = hmm.B

    return hmm_learn_model


def _to_weird_format(sequences):
    # Please don't ask
    if type(sequences) == list:
        # list of lists, potentially different lenghts
        X = np.concatenate(sequences).reshape(-1, 1)
        lenghts = [len(seq) for seq in sequences]
    else:  # 2d array, sequences are of same length
        X = np.array(sequences).ravel().reshape(-1, 1)
        lenghts = [sequences.shape[1]] * sequences.shape[0]

    return {"X": X, "lengths": lenghts}


def _make_random_parameters(n_hidden_states, n_observable_states, random_state=None):
    """Randomly generate probability matrices."""

    rng = _check_random_state(random_state)
    pi = rng.rand(n_hidden_states)
    pi /= pi.sum()

    A = rng.rand(n_hidden_states, n_hidden_states)
    A /= A.sum(axis=1, keepdims=True)

    B = rng.rand(n_hidden_states, n_observable_states)
    B /= B.sum(axis=1, keepdims=True)

    return pi, A, B


def _make_random_sequences_observations(
    n_seq, n_observable_states, n_obs_min, n_obs_max=None, random_state=None
):
    """Randomly generate observation sequences.

    Return 2D array with observations of size n_obs_min if n_obs_max is None.
    Else a list of lists (with n_obs_min <= length < n_obs_max) is returned.
    """

    rng = _check_random_state(random_state)
    if n_obs_max is None:
        # return 2d numpy array, all observations have same length
        return rng.randint(n_observable_states, size=(n_seq, n_obs_min))
    else:
        return [
            rng.randint(
                n_observable_states, size=rng.randint(n_obs_min, n_obs_max)
            ).tolist()
            for _ in range(n_seq)
        ]


def _check_random_state(seed):
    # Stolen from scikit-learn
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState" " instance" % seed
    )
