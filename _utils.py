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
