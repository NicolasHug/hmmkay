from numba import njit
import numpy as np


def _check_array_sums_to_1(a, name="array"):
    a_sum = a.sum()
    if not (1 - 1e-5 < a_sum < 1 + 1e-5):
        err_msg = f"{name} must sum to 1. Got \n{a}.sum() = {a_sum}"
        raise ValueError(err_msg)


def _allocate_or_reuse(frame, requested_shape, dtype=np.float):
    # Return frame if requested shape is smaller than frame.shape else
    # allocate new array
    if frame is None or any(a < b for (a, b) in zip(frame.shape, requested_shape)):
        return np.empty(shape=requested_shape, dtype=dtype)
    else:  # reuse
        return frame


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
    return {
        "X": np.array(sequences).ravel().reshape(-1, 1),
        "lengths": [sequences.shape[1]] * sequences.shape[0],
    }
