import pytest
import numpy as np
from numba import types
from numba.typed import List

from hmmkay.utils import (
    _check_array_sums_to_1,
    make_proba_matrices,
    make_observation_sequences,
    check_sequences,
)


def test_make_observation_sequences():
    # basic tests for shape, types and values

    # test with constant number of observations
    n_seq, n_observable_states, n_obs_min, n_obs_max = 10, 3, 12, None
    sequences = make_observation_sequences(
        n_seq=n_seq,
        n_observable_states=n_observable_states,
        n_obs_min=n_obs_min,
        n_obs_max=n_obs_max,
        random_state=0,
    )
    assert isinstance(sequences, np.ndarray)
    assert sequences.dtype == np.int32
    assert sequences.shape == (n_seq, n_obs_min)
    assert np.all(np.unique(sequences) == np.arange(n_observable_states))

    # test with non-constant number of observations
    n_seq, n_observable_states, n_obs_min, n_obs_max = 10, 3, 12, 20
    sequences = make_observation_sequences(
        n_seq=n_seq,
        n_observable_states=n_observable_states,
        n_obs_min=n_obs_min,
        n_obs_max=n_obs_max,
        random_state=0,
    )
    assert isinstance(sequences, List)
    for seq in sequences:
        assert isinstance(seq, np.ndarray)
        assert seq.dtype == np.int32
        assert seq.ndim == 1
        assert n_obs_min <= seq.shape[0] < n_obs_max
        assert np.all(np.unique(seq) == np.arange(n_observable_states))


def test_make_proba_matrices():
    # Make sure matrices rows sum to 1

    n_hidden_states = 10
    pi, A, B = make_proba_matrices(n_hidden_states=n_hidden_states, random_state=0)

    _check_array_sums_to_1(pi)
    for s in range(n_hidden_states):
        _check_array_sums_to_1(A[s])
        _check_array_sums_to_1(B[s])


def _make_typed_list():  # helper for test below
    l = List.empty_list(types.int32[:])
    l.append(np.array([1, 2, 3], dtype=np.int32))
    l.append(np.array([0, 1, 3, 5], dtype=np.int32))
    return l


@pytest.mark.parametrize(
    "sequences, expected_type, expected_longest_length",
    [
        ([[1, 2], [1, 2, 3]], List, 3),
        (_make_typed_list(), List, 4),
        (np.arange(20).reshape(4, 5), np.ndarray, 5),
    ],
)
def test_check_sequences(sequences, expected_type, expected_longest_length):
    sequences, longest_length = check_sequences(sequences, return_longest_length=True)
    assert isinstance(sequences, expected_type)
    assert longest_length == expected_longest_length
