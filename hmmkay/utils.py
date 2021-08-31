"""
The utils module contains helpers for input checking, parameter generation and
sequence generation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Union, cast

import hmmlearn.hmm as hl
import numpy as np
from numba import njit, types
from numba.typed import List
from numpy.random import mtrand

from ._typing import FormattedSequences, Seed, Sequences

if TYPE_CHECKING:
    from hmm import HMM

__all__ = ["make_observation_sequences", "make_proba_matrices", "check_sequences"]


def check_array_sums_to_1(a: np.ndarray, name: str = "array") -> None:
    """It checks if a probability array sums to 1.

    Parameters
    ----------
    a: numpy.ndarray
        Probability array.
    name: str
        Name of the array. Default: "array".

    Raises
    ------
    ValueError
        When the array does not sum to 1.
    """

    a_sum = a.sum()
    if not (1 - 1e-5 < a_sum < 1 + 1e-5):
        err_msg = f"{name} must sum to 1. Got \n{a}.sum() = {a_sum}"
        raise ValueError(err_msg)


@njit(cache=True)
def choice(p: np.ndarray) -> np.intp:
    """It returns i with probability p[i]. Inspired in https://bit.ly/3gsxZPx

    Parameters
    ----------
    p: numpy.ndarray
        Probabilities array.

    Returns
    -------
    numpy.intp
        The chosen number.
    """

    # p must sum to 1

    return np.searchsorted(np.cumsum(p), np.random.random(), side="right")


@njit(cache=True)
def logsumexp(a: np.ndarray) -> float:
    """It calculates the logsumexp of an array a. From pygbm

    logsumexp(a) = ln( sum_{i=1}^n exp(a_i - a*) ),
    where a* = max_{i=1,...,n} a_i

    Parameters
    ----------
    a: numpy.ndarray
        The array to apply the function to.

    Returns
    -------
    float
        The logsumexp of the input array.
    """

    a_max = np.amax(a)
    if not np.isfinite(a_max):
        a_max = 0

    s = np.sum(np.exp(a - a_max))
    return np.log(s) + a_max


@njit(cache=True)
def argmax(a: np.ndarray) -> int:
    """It returns the argmax of an array.

    Apparently much faster than numpy.argmax in our context.

    Parameters
    ----------
    a: numpy.ndarray
        The array to apply the function to.

    Returns
    -------
    int
        The index of the maximum number on the array.
    """

    curr_max = a[0]
    curr_max_idx = 0
    for i in range(1, len(a)):
        if a[i] > curr_max:
            curr_max = a[i]
            curr_max_idx = i
    return curr_max_idx


# TODO: Review return type when hmmlearn is not imported above.
def get_hmm_learn_model(hmm: HMM) -> hl.MultinomialHMM:
    """It returns the equivalent model in hmmlearn.

    Parameters
    ----------
    hmm: hmmkay.HMM
        Input hidden markov model.

    Returns
    -------
    hmmlearn.hmm.MultinomialHMM
        The equivalent model.

    Raises
    ------
    RuntimeError
        When the hmmlearn library is not installed.
    """

    try:
        import hmmlearn.hmm
    except ImportError as ie:
        raise RuntimeError("Please install hmmlearn to run tests/benchmarks.") from ie

    hmm_learn_model = hmmlearn.hmm.MultinomialHMM(
        n_components=hmm.A.shape[0], init_params="", tol=0, n_iter=hmm.n_iter
    )
    hmm_learn_model.startprob_ = hmm.pi
    hmm_learn_model.transmat_ = hmm.A
    hmm_learn_model.emissionprob_ = hmm.B

    return hmm_learn_model


def to_weird_format(
    sequences: Sequences,
) -> dict[str, Union[np.ndarray, list[int]]]:
    """It formats the set of sequences for hmmlearn (required for tests).

    Parameters
    ----------
    sequences: hmmkay._typing.sequences
        The set of sequences.

    Returns
    -------
    dict[str, numpy.ndarray | list[int]]
    """
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
    n_hidden_states: int, optional
        Number of hidden states. Default: 4.
    n_observable_states: int, optional
        Number of observable states. Default 3.
    random_state: int | numpy.random.RandomState, optional
        Controls the RNG, see `scikit-learn glossary
        <https://scikit-learn.org/stable/glossary.html#term-random-state>`_
        for details. Default: None.

    Returns
    -------
    init_probas: numpy.ndarray
        The initial probabilities.
    transitions: numpy.ndarray
        The transition probabilities ``transitions[i, j] = P(s_{t+1} = j | s_t = i)``.
    emissions: numpy.ndarray
        The emission probabilities ``emissions[i, o] = P(O_t = o | s_t = i)``.
    """

    rng = check_random_state(random_state)
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
) -> FormattedSequences:
    """Generate random observation sequences.

    Parameters
    ----------
    n_seq: int, optional
        Number of sequences to generate. Default: 10.
    n_observable_states: int, optional
        Number of observable states. Default: 3.
    n_obs_min: int, optional
        Minimum length of each sequence. Default: 10.
    n_obs_max: int | None, optional
        If None (default), all sequences are of length ``n_obs_min`` and a 2d
        ndarray is returned. If an int, the length of each sequence is
        chosen randomly with ``n_obs_min <= length < n_obs_max``. A numba typed
        list of arrays is returned in this case. Default: None.
    random_state: int | np.random.RandomState, optional
        Controls the RNG, see `scikit-learn glossary
        <https://scikit-learn.org/stable/glossary.html#term-random-state>`_
        for details. Default: None.

    Returns
    -------
    sequences : numpy.ndarray | numba.typed.List
        The generated sequences of observable states
    """

    # TODO: generate a typed list instead of a list.

    rng = check_random_state(random_state)
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


def check_random_state(seed: Seed) -> np.random.RandomState:
    """It returns a RandomState sequence according to the input. From scikit-learn.

    Parameters
    ----------
    seed: hmmkay._typing.Seed
        The input to return the RandomState instance from.

    Returns
    -------
    numpy.random.RandomState
        The random state instance.
    """

    if seed is None or seed is np.random:
        return mtrand._rand
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
    sequences: hmmkay._typing.Sequences
        Sequences to be formatted.

    Returns
    -------
    hmmkay._typing.FormattedSequences
        The formatted sequences to either ndarray or numba typed list of ndarrays.
    int
        The length of the longest sequence.
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
