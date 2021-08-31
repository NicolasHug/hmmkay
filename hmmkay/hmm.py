from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit

from ._typing import FormattedSequences, Seed, Sequences

from .utils import (
    argmax,
    check_array_sums_to_1,
    check_random_state,
    check_sequences,
    choice,
    logsumexp,
)

__all__ = ["HMM"]


class HMM:
    """Discrete Hidden Markov Model.

    The number of hidden and observable states are determined by the shapes
    of the probability matrices passed as parameters.

    Parameters
    ----------
    init_probas: numpy.typing.ArrayLike
        The initial probabilities.
    transitions: numpy.typing.ArrayLike
        The transition probabilities ``transitions[i, j] = P(s_{t+1} = j | s_t = i)``.
    emissions: numpy.typing.ArrayLike
        The probabilities of symbol emission:
        ``emissions[i, o] = P(O_t = o | s_t = i)``.
    current_state: int, optional.
        Starting state of the HMM. Negative values mean unspecified. Default: -1.
    n_iter: int, optional
        Number of iterations to run for the EM algorithm (in ``fit()``). Default: 10.
    """

    def __init__(
        self,
        init_probas: npt.ArrayLike,
        transitions: npt.ArrayLike,
        emissions: npt.ArrayLike,
        n_iter: int = 10,
    ) -> None:
        self.init_probas: np.ndarray = np.array(init_probas, dtype=np.float64)
        self.transitions: np.ndarray = np.array(transitions, dtype=np.float64)
        self.emissions: np.ndarray = np.array(emissions, dtype=np.float64)

        self.n_iter: int = n_iter

        self.n_hidden_states: int = self.A.shape[0]
        self.n_observable_states: int = self.B.shape[1]

        # TODO: wtf is this if
        if not (
            self.A.shape[0] == self.A.shape[1] == self.pi.shape[0] == self.B.shape[0]
        ):
            raise ValueError("inconsistent number of hidden states.")

        self._check_matrices_conditioning()

    # pi, A and B are respectively init_probas, transitions and emissions
    # matrices. _log_pi, _log_A and _log_B are updated each time pi, A, or B
    # are updated, respectively. Consider these private (and bug-prone :)),
    # Updating transitions would not update _log_A.
    @property
    def pi(self) -> np.ndarray:
        """It returns the initial probabilities array.

        Returns
        -------
        numpy.ndarray
            The initial probabilities array.
        """

        return self.init_probas

    @pi.setter
    def pi(self, value: np.ndarray) -> None:
        """It updates the value of the initial probabilities array.

        Parameters
        ----------
        value: numpy.ndarray
            The new initial probabilities.
        """

        self.init_probas = value
        self._recompute_log_pi = True

    @property
    def _log_pi(self) -> np.ndarray:
        """It updates and returns the log of the initial probabilities array.

        Returns
        -------
        numpy.ndarray
            The log of the initial probabilities array.
        """

        if getattr(self, "_recompute_log_pi", True):
            self.__log_pi = np.log(self.pi)
            self._recompute_log_pi = False
        return self.__log_pi

    @property
    def A(self) -> np.ndarray:
        """It returns the hidden states transition matrix.

        Returns
        -------
        numpy.ndarray
            The hidden states transition matrix.
        """

        return self.transitions

    @A.setter
    def A(self, value: np.ndarray) -> None:
        """It updates the hidden states transition matrix.

        Parameters
        ----------
        value: numpy.ndarray
            The new transition matrix.
        """

        self.transitions = value
        self._recompute_log_A = True

    @property
    def _log_A(self) -> np.ndarray:
        """It updates and returns the logarithm of the hidden states transition matrix.

        Returns
        -------
        numpy.ndarray
            The logarithm of the hidden states transition matrix.
        """

        if getattr(self, "_recompute_log_A", True):
            self.__log_A = np.log(self.A)
            self._recompute_log_A = False
        return self.__log_A

    @property
    def B(self) -> np.ndarray:
        """It returns the emission matrix.

        Returns
        -------
        numpy.ndarray
            The emission matrix.
        """

        return self.emissions

    @B.setter
    def B(self, value: np.ndarray) -> None:
        """It updates the emission matrix.

        Parameters
        ----------
        value: numpy.ndarray
            The new emission matrix.
        """

        self.emissions = value
        self._recompute_log_B = True

    @property
    def _log_B(self) -> np.ndarray:
        """It updates and returns the logarithm of the emission matrix.

        Returns
        -------
        numpy.ndarray
            The logarithm of the emission matrix.
        """

        if getattr(self, "_recompute_log_B", True):
            self.__log_B = np.log(self.B)
            self._recompute_log_B = False
        return self.__log_B

    def log_likelihood(self, sequences: Sequences) -> float:
        """Compute log-likelihood of sequences.

        Parameters
        ----------
        sequences: hmmkay._typing.Sequences
            The sequences of observable states.

        Returns
        -------
        float
            The total log-likelihood.
        """

        total_log_likelihood = 0
        sequences, n_obs_max = check_sequences(sequences)
        log_alpha = np.empty(shape=(self.n_hidden_states, n_obs_max), dtype=np.float32)
        for seq in sequences:
            total_log_likelihood += self._forward(seq, log_alpha)
        return total_log_likelihood

    def decode(self, sequences: Sequences) -> tuple[FormattedSequences, np.ndarray]:
        """Decode sequences using the Viterbi algorithm.

        Given a sequence of observable states, return the sequence of hidden
        states that most-likely generated the input.

        Parameters
        ----------
        sequences: hmmkay._typing.FormattedSequences
            The sequences of observable states.

        Returns
        -------
        hmmkay._typing.Sequences
            The most likely sequences of hidden states.
        numpy.ndarray
            log-probabilities of the joint sequences of observable and hidden
            states.
        """

        sequences, n_obs_max = check_sequences(sequences)

        hidden_states_sequences = []
        log_probas = []

        log_V = np.empty(shape=(self.n_hidden_states, n_obs_max), dtype=np.float32)
        back_path = np.empty(shape=(self.n_hidden_states, n_obs_max), dtype=np.int32)

        for seq in sequences:
            n_obs = seq.shape[0]  # type: ignore
            self._viterbi(seq, log_V, back_path)
            best_path = np.empty(n_obs, dtype=np.int32)
            log_proba = _get_best_path(log_V, back_path, best_path)
            hidden_states_sequences.append(best_path)
            log_probas.append(log_proba)

        if isinstance(sequences, np.ndarray):
            # All sequences have the same length
            hidden_states_sequences = np.array(hidden_states_sequences)

        return hidden_states_sequences, np.array(log_probas)

    def sample(
        self, n_seq: int = 10, n_obs: int = 10, random_state: Seed = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample sequences of hidden and observable states.

        Parameters
        ----------
        n_seq : int, optional
            Number of sequences to sample. Default: 10.
        n_obs : int, optional
            Number of observations per sequence. Default: 10.
        random_state: int or np.random.RandomState instance, optional
            Controls the RNG, see `scikt-learn glossary
            <https://scikit-learn.org/stable/glossary.html#term-random-state>`_
            for details. Default: None.

        Returns
        -------
        numpy.ndarray
            Hidden states sequences.
        numpy.ndarray
            Observable states sequences.
        """
        # TODO: allow n_obs_max

        rng = check_random_state(random_state)
        sequences = np.array(
            [
                _sample_one(n_obs, self.pi, self.A, self.B, seed=rng.tomaxint())
                for _ in range(n_seq)
            ]
        )
        # Unzip array of (hidden_states, observation) into tuple of arrays
        sequences = sequences.swapaxes(0, 1)
        return sequences[0], sequences[1]

    def next_state(self, current_state: int = -1) -> tuple[int, int]:
        """It runs a step of the HMM.

        Parameters
        ----------
        current_state: int, optional
            The current state of the HMM. Default: -1.

        Returns
        -------
        int
            The new current state of the HMM.
        int
            The observed state.
        """
        return _get_next_state(current_state, self.pi, self.A, self.B)

    def fit(self, sequences: Sequences) -> HMM:
        """Fit model to sequences.

        The probabilities matrices ``init_probas``, ``transitions`` and
        ``emissions`` are estimated with the EM algorithm.

        Parameters
        ----------
        sequences: hmmkay._typing.Sequences
            The sequences of observable states.

        Returns
        -------
        hmmkay.HMM
            The estimated hidden markov model instance.
        """

        sequences, n_obs_max = check_sequences(sequences)
        log_alpha = np.empty(shape=(self.n_hidden_states, n_obs_max))
        log_beta = np.empty(shape=(self.n_hidden_states, n_obs_max))
        # E[i, j, t] = P(s_t = i, s_{t+1} = j | O, lambda)
        log_E = np.empty(
            shape=(self.n_hidden_states, self.n_hidden_states, n_obs_max - 1)
        )
        # g[i, t] = P(s_t = i | O, lambda)
        log_gamma = np.empty(shape=(self.n_hidden_states, n_obs_max))

        for _ in range(self.n_iter):

            self.pi, self.A, self.B = _do_EM_step(
                sequences,
                self._log_pi,
                self._log_A,
                self._log_B,
                log_alpha,
                log_beta,
                log_E,
                log_gamma,
            )
            self._check_matrices_conditioning()
        return self

    def _viterbi(self, seq, log_V, back_path) -> None:
        # dummy wrapper for convenience
        return _viterbi(seq, self._log_pi, self._log_A, self._log_B, log_V, back_path)

    def _forward(self, seq, log_alpha) -> float:
        # dummy wrapper for convenience
        return _forward(seq, self._log_pi, self._log_A, self._log_B, log_alpha)

    def _backward(self, seq, log_beta) -> None:
        # dummy wrapper for convenience
        return _backward(seq, self._log_pi, self._log_A, self._log_B, log_beta)

    def _check_matrices_conditioning(self) -> None:
        """Check the matrices in the model instance.

        Raises
        ------
        ValueError
            When a probabilities array does not sum to 1.
        """

        check_array_sums_to_1(self.pi, "init_probas")
        for s in range(self.n_hidden_states):
            check_array_sums_to_1(self.A[s], f"Row {s} of A")
            check_array_sums_to_1(self.B[s], f"Row {s} of B")


@njit(cache=True)
def _get_next_state(
    current_state: int, pi: np.ndarray, A: np.ndarray, B: np.ndarray
) -> tuple[int, int]:
    """Returns the next and the observed state in the HMM.

    Parameters
    ----------
    current_state: int
        The current state of the HMM.
    pi: numpy.ndarray
        The initial probabilities array.
    A: numpy.ndarray
        The hidden states transition matrix.
    B: numpy.ndarray
        The emission matrix.

    Returns
    -------
    int
        The new state of the HMM.
    int
        The observed state.
    """

    if current_state < 0:
        current_state = choice(pi)
    else:
        current_state = choice(A[current_state])
    emission = choice(B[current_state])
    return current_state, emission


@njit(cache=True)
def _sample_one(
    n_obs: int, pi: np.ndarray, A: np.ndarray, B: np.ndarray, seed: int
) -> tuple[list[np.intp], list[np.intp]]:
    """Returns both the sampled hidden states and observable sequences.

    Parameters
    ----------
    n_obs: int
        The number of observations in the sequences.
    pi: numpy.ndarray
        The initial probabilities array.
    A: numpy.ndarray
        The hidden states transition matrix.
    B: numpy.ndarray
        The emission matrix.
    seed: int
        The seed for the RNG.

    Returns
    -------
    list[numpy.intp]
        The sampled hidden states sequence.
    list[numpy.intp]
        The sampled observable states sequence.
    """

    # TODO: review this RNG
    np.random.seed(seed)  # local to this numba function, not global numpy
    observations = []
    hidden_states = []
    s = choice(pi)
    for _ in range(n_obs):
        hidden_states.append(s)
        obs = choice(B[s])
        observations.append(obs)
        s = choice(A[s])
    return hidden_states, observations


@njit(cache=True)
def _forward(
    seq: np.ndarray,
    log_pi: np.ndarray,
    log_A: np.ndarray,
    log_B: np.ndarray,
    log_alpha: np.ndarray,
) -> float:
    """It fills log_alpha array with log-probabilities and returns log-likelihood.

    Parameters
    ----------
    seq: numpy.ndarray
        A sequence.
    log_pi: numpy.ndarray
        The logarithm of the initial probabilities array.
    log_A: numpy.ndarray
        The logarithm of the hidden states transition matrix.
    log_B: numpy.ndarray
        The logarithm of the emission matrix.
    log_alpha: numpy.ndarray
        Empty array to fill.

    Returns
    -------
    float
        The log-likelihood calculated from log_alpha.
    """

    # alpha[i, t] = P(O_1, ... O_t, s_t = i | lambda)
    # reccursion is alpha[i, t] = B[i, Ot] * sum_j(alpha[j, t - 1] * A[i, j])
    # which becomes (when applying log)
    # log_alpha[i, t] = log(B[i, O_t]) +
    #                   log(sum_j(exp(log_alpha[j, t - 1] + _log_A[i, j])))
    # since log(sum(ai . bj)) =
    #       log(sum(exp(log_ai + log_bi)))

    n_obs = len(seq)
    n_hidden_states = log_pi.shape[0]
    log_alpha[:, 0] = log_pi + log_B[:, seq[0]]
    buffer = np.empty(shape=n_hidden_states)
    for t in range(1, n_obs):
        for s in range(n_hidden_states):
            for ss in range(n_hidden_states):
                buffer[ss] = log_alpha[ss, t - 1] + log_A[ss, s]
            log_alpha[s, t] = logsumexp(buffer) + log_B[s, seq[t]]
    return logsumexp(log_alpha[:, n_obs - 1])


@njit(cache=True)
def _backward(
    seq: np.ndarray,
    log_pi: np.ndarray,
    log_A: np.ndarray,
    log_B: np.ndarray,
    log_beta: np.ndarray,
) -> None:
    """It fills log_beta array with log-probabilities.

    Parameters
    ----------
    seq: numpy.ndarray
        A sequence.
    log_pi: numpy.ndarray
        The logarithm of the initial probabilities array.
    log_A: numpy.ndarray
        The logarithm of the hidden states transition matrix.
    log_B: numpy.ndarray
        The logarithm of the emission matrix.
    log_beta: numpy.ndarray
        Empty array to fill.
    """

    # beta[i, t] = P(O_{t+1}, ... O_T, / s_t = i, lambda)

    n_obs = seq.shape[0]
    n_hidden_states = log_pi.shape[0]
    log_beta[:, n_obs - 1] = np.log(1)
    buffer = np.empty(shape=n_hidden_states)
    for t in range(n_obs - 2, -1, -1):
        for s in range(n_hidden_states):
            for ss in range(n_hidden_states):
                buffer[ss] = log_A[s, ss] + log_B[ss, seq[t + 1]] + log_beta[ss, t + 1]
            log_beta[s, t] = logsumexp(buffer)


@njit(cache=True)
def _viterbi(
    seq: np.ndarray,
    log_pi: np.ndarray,
    log_A: np.ndarray,
    log_B: np.ndarray,
    log_V: np.ndarray,
    back_path: np.ndarray,
) -> None:
    """It fills both log_V array with log-probabilities and back_path with back links.

    Parameters
    ----------
    seq: numpy.ndarray
        A sequence.
    log_pi: numpy.ndarray
        The logarithm of the initial probabilities array.
    log_A: numpy.ndarray
        The logarithm of the hidden states transition matrix.
    log_B: numpy.ndarray
        The logarithm of the emission matrix.
    log_V: numpy.ndarray
        Empty array to fill with log probabilities.
    back_path: numpy.ndarray
        Empty array to fill with back links.
    """

    # V[i, t] = max_{s_1...s{t-1}} P(O_1, ... O_t, s_1, ... s_{t-1}, s_t=i | lambda)

    n_obs = seq.shape[0]
    n_hidden_states = log_pi.shape[0]
    log_V[:, 0] = log_pi + log_B[:, seq[0]]
    buffer = np.empty(shape=n_hidden_states)
    for t in range(1, n_obs):
        for s in range(n_hidden_states):
            for ss in range(n_hidden_states):
                buffer[ss] = log_V[ss, t - 1] + log_A[ss, s]
            best_prev = argmax(buffer)
            back_path[s, t] = best_prev
            log_V[s, t] = buffer[best_prev] + log_B[s, seq[t]]


@njit(cache=True)
def _get_best_path(
    log_V: np.ndarray, back_path: np.ndarray, best_path: np.ndarray
) -> float:
    """It fills out the best_path array.

    Parameters
    ----------
    log_V: np.ndarray
        Matrix of log-probabilities.
    back_path: np.ndarray
        The back links.

    Returns
    -------
    float
        The maximum log-probability.
    """
    n_obs = best_path.shape[0]
    s = argmax(log_V[:, n_obs - 1])
    out = log_V[s, n_obs - 1]
    for t in range(n_obs - 1, -1, -1):
        best_path[t] = s
        s = back_path[s, t]
    return out


@njit(cache=True)
def _do_EM_step(
    sequences: FormattedSequences,
    log_pi: np.ndarray,
    log_A: np.ndarray,
    log_B: np.ndarray,
    log_alpha: np.ndarray,
    log_beta: np.ndarray,
    log_E: np.ndarray,
    log_gamma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return A, B and C after one EM step.

    Parameters
    ----------
    sequences: hmmkay._typing.FormattedSequences
        The formatted input sequences.
    log_pi: numpy.ndarray
        The logarithm of the initial probabilities array.
    log_A: numpy.ndarray
        The logarithm of the hidden states transition matrix.
    log_B: numpy.ndarray
        The logarithm of the emission matrix.
    TODO: Review parameters below.
    log_alpha: numpy.ndarray
        The log-alpha-probabilities matrix to be filled in the forward step.
    log_beta: numpy.ndarray
        The log-beta-probabilities matrix to be filled in the backward step.
    log_E: numpy.ndarray
        The tensor of log-values combinations to be filled.
    log_gamma: numpy.ndarray
        The matrix of alpha and beta log-probabilities to be filled.

    Returns
    -------
    numpy.ndarray
        The new initial probabilities array.
    numpy.ndarray
        The new hidden states transition matrix.
    numpy.ndarray
        The new emission matrix.
    """

    # E STEP (over all sequences)
    # Accumulators for parameters of the hmm. They are summed over for
    # each sequence, then normalized in the M-step.
    # These are homogeneous to probabilities, not log-probabilities.

    pi_acc = np.zeros_like(log_pi)
    A_acc = np.zeros_like(log_A)
    B_acc = np.zeros_like(log_B)

    n_hidden_states = log_pi.shape[0]

    for seq_idx in range(len(sequences)):  # numba can't iterate over 2D arrays
        seq = sequences[seq_idx]
        n_obs = seq.shape[0]
        log_likelihood = _forward(seq, log_pi, log_A, log_B, log_alpha)
        _backward(seq, log_pi, log_A, log_B, log_beta)

        # Compute E
        for t in range(n_obs - 1):
            for i in range(n_hidden_states):
                for j in range(n_hidden_states):
                    log_E[i, j, t] = (
                        log_alpha[i, t]
                        + log_A[i, j]
                        + log_B[j, seq[t + 1]]
                        + log_beta[j, t + 1]
                        - log_likelihood
                    )

        # compute gamma
        log_gamma = log_alpha + log_beta - log_likelihood

        # M STEP accumulators
        pi_acc += np.exp(log_gamma[:, 0])
        A_acc += np.sum(np.exp(log_E[:, :, : n_obs - 1]), axis=-1)
        for t in range(n_obs):
            B_acc[:, seq[t]] += np.exp(log_gamma[:, t])

    # M STEP (mostly done in the accumulators already)
    pi = pi_acc / pi_acc.sum()
    # equivalent to X / X.sum(axis=1, keepdims=True) but not supported
    A = A_acc / A_acc.sum(axis=1).reshape(-1, 1)
    B = B_acc / B_acc.sum(axis=1).reshape(-1, 1)

    return pi, A, B
