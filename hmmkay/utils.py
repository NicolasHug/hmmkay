"""
The utils module contains helpers for input checking, parameter generation and
sequence generation.
"""
import numpy as np
import numpy.typing as npt
import hmmlearn.hmm as hl

from hmm import HMM
from numba import njit, types
from numba.typed import List
from typing import Union, cast
from _typing import FormattedSequences, Seed, Sequences


__all__ = ["make_observation_sequences", "make_proba_matrices", "check_sequences"]


def _check_array_sums_to_1(a: np.ndarray, name: str = "array") -> None:
    a_sum = a.sum()
    if not (1 - 1e-5 < a_sum < 1 + 1e-5):
        err_msg = f"{name} must sum to 1. Got \n{a}.sum() = {a_sum}"
        raise ValueError(err_msg)


@njit(cache=True)
def _choice(p: np.ndarray) -> np.intp:
    """return i with probability p[i]"""
    # inspired from https://github.com/numba/numba/issues/2539
    # p must sum to 1
    return np.searchsorted(np.cumsum(p), np.random.random(), side="right")


@njit(cache=True)
def _logsumexp(a: np.ndarray) -> float:
    # stolen from pygbm \o/

    a_max = np.amax(a)
    if not np.isfinite(a_max):
        a_max = 0

    s = np.sum(np.exp(a - a_max))
    return np.log(s) + a_max


@njit(cache=True)
def _argmax(a: np.ndarray) -> int:
    # Apparently much faster than np.argmax in our context
    curr_max = a[0]
    curr_max_idx = 0
    for i in range(1, len(a)):
        if a[i] > curr_max:
            curr_max = a[i]
            curr_max_idx = i
    return curr_max_idx


# TODO: Review return type when hmmlearn is not imported above.
def _get_hmm_learn_model(hmm: HMM) -> hl.MultinomialHMM:
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


def _to_weird_format(
    sequences: Sequences,
) -> dict[str, Union[np.ndarray, list[int]]]:
    # Please don't ask
    if isinstance(sequences, (list, List)):
        # list of lists, potentially different lengths
        X = np.array(np.concatenate(sequences)).reshape(-1, 1)
        lengths = [len(seq) for seq in sequences]
    else:  # 2d array, sequences are of same length
        X_prev = np.array(sequences)
        X = X_prev.ravel().reshape(-1, 1)
        lengths = [X_prev.shape[1]] * X_prev.shape[0]

    return {"X": X, "lengths": lengths}


def make_proba_matrices(
    n_hidden_states: int = 4, n_observable_states: int = 3, random_state: Seed = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate random probability matrices.

    Parameters
    ----------
    n_hidden_states : int, default=4
        Number of hidden states
    n_observable_states : int, default=3
        Number of observable states
    random_state: int or np.random.RandomState instance, default=None
        Controls the RNG, see `scikt-learn glossary
        <https://scikit-learn.org/stable/glossary.html#term-random-state>`_
        for details.

    Returns
    -------
    init_probas : array-like of shape (n_hidden_states,)
        The initial probabilities.
    transitions : array-like of shape (n_hidden_states, n_hidden_states)
        The transition probabilities. ``transitions[i, j] = P(st+1 = j / st = i)``.
    emissions : array-like of shape (n_hidden_states, n_observable_states)
        The probabilities of symbol emission. ``emissions[i, o] = P(Ot = o /
        st = i)``.
    """

    rng = _check_random_state(random_state)
    pi = rng.rand(n_hidden_states)
    pi /= pi.sum()

    A = rng.rand(n_hidden_states, n_hidden_states)
    A /= A.sum(axis=1, keepdims=True)

    B = rng.rand(n_hidden_states, n_observable_states)
    B /= B.sum(axis=1, keepdims=True)

    return pi, A, B


def make_observation_sequences(
    n_seq: int = 10,
    n_observable_states: int = 3,
    n_obs_min: int = 10,
    n_obs_max: int = None,
    random_state: int = None,
) -> np.ndarray:
    """Generate random observation sequences.

    Parameters
    ----------
    n_seq : int, default=10
        Number of sequences to generate
    n_observable_states : int, default=3
        Number of observable states.
    n_obs_min : int, default=10
        Minimum length of each sequence.
    n_obs_max : int or None, default=None
        If None (default), all sequences are of length ``n_obs_min`` and a 2d
        ndarray is returned. If an int, the length of each sequence is
        chosen randomly with ``n_obs_min <= length < n_obs_max``. A numba typed
        list of arrays is returned in this case.
    random_state: int or np.random.RandomState instance, default=None
        Controls the RNG, see `scikt-learn glossary
        <https://scikit-learn.org/stable/glossary.html#term-random-state>`_
        for details.

    Returns
    -------
    sequences : ndarray of shape (n_seq, n_obs_min,) or numba typed list of \
            ndarray of variable length
        The generated sequences of observable states
    """
    # TODO: generate a typed list instead of a list.

    rng = _check_random_state(random_state)
    if n_obs_max is None:
        # return 2d numpy array, all observations have same length
        return rng.randint(n_observable_states, size=(n_seq, n_obs_min), dtype=np.int32)
    else:
        sequences = List.empty_list(types.int32[:])
        for _ in range(n_seq):
            sequences.append(
                rng.randint(
                    n_observable_states,
                    size=rng.randint(n_obs_min, n_obs_max),
                    dtype=np.int32,
                )
            )
        return sequences


def _check_random_state(seed: Seed) -> np.random.RandomState:
    # Stolen from scikit-learn
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, int):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState" " instance" % seed
    )


def check_sequences(sequences: Sequences) -> tuple[FormattedSequences, int]:
    """Convert sequences into appropriate format.

    This helper is called before any method that uses sequences. It is
    recommended to convert your sequences once and for all before using the
    ``HMM`` class, to avoid repeated convertions.

    Parameters
    ----------
    sequences : array-like of shape (n_seq, n_obs) or list/typed list of iterables of \
            variable length.
        Lists of iterables are converted to typed lists of
        numpy arrays, which can have different lengths. 2D arrays are
        untouched (all sequences have the same length).

    Returns
    -------
    sequences : ndarray of shape (n_seq, n_obs) or typed list of ndarray of \
            variable length
        The sequences converted either to ndarray or numba typed list of ndarrays.
    """
    if isinstance(sequences, List):  # typed list
        longest_seq_length = max(len(seq) for seq in sequences)
    elif isinstance(sequences, list):  # regular list, convert to numba typed list
        longest_seq_length = max(len(seq) for seq in sequences)
        new_sequences = cast(List, List.empty_list(types.int32[:]))
        for seq in sequences:
            new_sequences.append(np.asarray(seq, dtype=np.int32))

        sequences = new_sequences
    elif isinstance(sequences, np.ndarray):
        longest_seq_length = sequences.shape[1]
    else:
        raise ValueError(
            "Accepted sequences types are 2d numpy arrays, "
            "lists of iterables, or numba typed lists of iterables."
        )

    return sequences, longest_seq_length
