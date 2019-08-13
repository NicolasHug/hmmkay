from numba import njit
from numba.typed import List
import numpy as np
from scipy.special import logsumexp

from _utils import _choice, _logsumexp, _check_array_sums_to_1, _allocate_or_reuse


class HMM:
    def __init__(self, init_probas=None, transitions=None, emissions=None, n_iter=10):

        self.init_probas = np.array(init_probas, dtype=np.float64)
        self.transitions = np.array(transitions, dtype=np.float64)
        self.emissions = np.array(emissions, dtype=np.float64)

        self.n_iter = n_iter

        self.n_hidden_states = self.A.shape[0]
        self.n_observable_states = self.B.shape[1]

        if not (
            self.A.shape[0] == self.A.shape[1] == self.pi.shape[0] == self.B.shape[0]
        ):
            raise ValueError("inconsistent number of hidden states.")

        self._check_matrices_conditioning()

    def likelihood(self, sequences):
        return np.exp(self.log_likelihood(sequences))

    def log_likelihood(self, sequences):
        sequences = np.array(sequences)
        n_obs = sequences.shape[1]
        log_alpha = np.empty(shape=(self.n_hidden_states, n_obs))

        total_log_likelihood = 0
        for seq in sequences:
            n_obs = len(seq)
            total_log_likelihood += self._forward(seq, log_alpha)
        return total_log_likelihood

    def decode(self, sequences, return_log_probas=False):
        sequences = np.array(sequences)
        n_obs = sequences.shape[1]
        log_V = np.empty(shape=(self.n_hidden_states, n_obs))
        back_path = np.empty(shape=(self.n_hidden_states, n_obs), dtype=np.int32)

        hidden_states_sequences = []
        log_probas = []
        for seq in sequences:
            self._viterbi(seq, log_V, back_path)
            best_path = np.empty(n_obs, dtype=np.int)
            log_proba = _get_best_path(log_V, back_path, best_path)
            hidden_states_sequences.append(best_path)
            if return_log_probas:
                log_probas.append(log_proba)

        hidden_states_sequences = np.array(hidden_states_sequences)
        if return_log_probas:
            return hidden_states_sequences, np.array(log_probas)
        else:
            return hidden_states_sequences

    def sample(self, n_seq, n_obs, seed=0):

        rng = np.random.RandomState(seed)
        sequences = np.array(
            [
                _sample_one(n_obs, self.pi, self.A, self.B, seed=rng.randint(10000))
                for _ in range(n_seq)
            ]
        )
        # Unzip array of (observations, hidden_states) into tuple of arrays
        sequences = sequences.swapaxes(0, 1)
        return sequences[0], sequences[1]

    def EM(self, sequences, n_iter=100):
        sequences = np.array(sequences)
        n_obs = sequences.shape[1]
        log_alpha = np.empty(shape=(self.n_hidden_states, n_obs))
        log_beta = np.empty(shape=(self.n_hidden_states, n_obs))
        # E[i, j, t] = P(st = i, st+1 = j / O, lambda)
        log_E = np.empty(shape=(self.n_hidden_states, self.n_hidden_states, n_obs - 1))
        # g[i, t] = P(st = i / O, lambda)
        log_gamma = np.empty(shape=(self.n_hidden_states, n_obs))

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

    def _viterbi(self, seq, log_V, back_path):
        # dummy wrapper for conveniency
        _viterbi(seq, self._log_pi, self._log_A, self._log_B, log_V, back_path)

    def _forward(self, seq, log_alpha):
        # dummy wrapper for conveniency
        return _forward(seq, self._log_pi, self._log_A, self._log_B, log_alpha)

    def _backward(self, seq, log_beta):
        # dummy wrapper for conveniency
        return _backward(seq, self._log_pi, self._log_A, self._log_B, log_beta)

    def _check_matrices_conditioning(self):

        _check_array_sums_to_1(self.pi, "init_probas")
        for s in range(self.n_hidden_states):
            _check_array_sums_to_1(self.A[s], f"Row {s} of A")
            _check_array_sums_to_1(self.B[s], f"Row {s} of B")

    # pi, A and B are respectively init_probas, transitions and emissions
    # matrices. _log_pi, _log_A and _log_B are updated each time pi, A, or B
    # are updated, respectively. Consider these private (and bug-prone :)),
    # Updating transitions would not update _log_A.
    @property
    def pi(self):
        return self.init_probas

    @pi.setter
    def pi(self, value):
        self.init_probas = value
        self._recompute_log_pi = True

    @property
    def _log_pi(self):
        if getattr(self, "_recompute_log_pi", True):
            self.__log_pi = np.log(self.pi)
        return self.__log_pi

    @property
    def A(self):
        return self.transitions

    @A.setter
    def A(self, value):
        self.transitions = value
        self._recompute_log_A = True

    @property
    def _log_A(self):
        if getattr(self, "_recompute_log_A", True):
            self.__log_A = np.log(self.A)
        return self.__log_A

    @property
    def B(self):
        return self.emissions

    @B.setter
    def B(self, value):
        self.emissions = value
        self._recompute_log_B = True

    @property
    def _log_B(self):
        if getattr(self, "_recompute_log_B", True):
            self.__log_B = np.log(self.B)
        return self.__log_B


@njit(cache=True)
def _sample_one(n_obs, pi, A, B, seed):
    """Return (observations, hidden_states) sample"""
    np.random.seed(seed)  # local to this numba function, not global numpy

    observations = []
    hidden_states = []
    s = _choice(pi)
    for _ in range(n_obs):
        hidden_states.append(s)
        obs = _choice(B[s])
        observations.append(obs)
        s = _choice(A[s])

    return observations, hidden_states


@njit(cache=True)
def _forward(seq, log_pi, log_A, log_B, log_alpha):
    """Fill log_alpha array with log probabilities, return log-likelihood"""
    # alpha[i, t] = P(O1, ... Ot, st = i / lambda)
    # reccursion is alpha[i, t] = B[i, Ot] * sumj(alpha[j, t - 1] * A[i, j])
    # which becomes (when applying log)
    # log_alpha[i, t] = log(B[i, Ot]) +
    #                   logsum_jexp(log_alpha[j, t - 1] + _log_A[i, j])
    # since log(sum(ai . bj)) =
    #       log(sum(exp(log_ai + log_bi)))

    n_obs = len(seq)
    n_hidden_states = log_pi.shape[0]
    log_alpha[:, 0] = log_pi + log_B[:, seq[0]]
    for t in range(1, n_obs):
        for s in range(n_hidden_states):
            log_alpha[s, t] = _logsumexp(log_alpha[:, t - 1] + log_A[:, s])
            log_alpha[s, t] += log_B[s, seq[t]]
    return _logsumexp(log_alpha[:, n_obs - 1])


@njit(cache=True)
def _backward(seq, log_pi, log_A, log_B, log_beta):
    """Fills beta array with log probabilities"""
    # beta[i, t] = P(Ot+1, ... OT, / st = i, lambda)

    n_obs = seq.shape[0]
    n_hidden_states = log_pi.shape[0]
    log_beta[:, -1] = np.log(1)
    for t in range(n_obs - 2, -1, -1):
        for s in range(n_hidden_states):
            log_beta[s, t] = _logsumexp(
                log_A[s, :] + log_B[:, seq[t + 1]] + log_beta[:, t + 1]
            )


@njit(cache=True)
def _viterbi(seq, log_pi, log_A, log_B, log_V, back_path):
    """Fill V array with log probabilities and back_path with back links"""
    # V[i, t] = max_{s1...st-1} P(O1, ... Ot, s1, ... st-1, st=i / lambda)
    n_obs = seq.shape[0]
    n_hidden_states = log_pi.shape[0]
    log_V[:, 0] = log_pi + log_B[:, seq[0]]
    for t in range(1, n_obs):
        for s in range(n_hidden_states):
            work_buffer = log_V[:, t - 1] + log_A[:, s]
            best_prev = np.argmax(work_buffer)
            back_path[s, t] = best_prev
            log_V[s, t] = work_buffer[best_prev] + log_B[s, seq[t]]


@njit(cache=True)
def _get_best_path(log_V, back_path, best_path):
    """Fill out best_path array"""
    s = np.argmax(log_V[:, -1])
    out = log_V[s, -1]
    for t in range(back_path.shape[1] - 1, -1, -1):
        best_path[t] = s
        s = back_path[s, t]
    return out


@njit(cache=True)
def _do_EM_step(sequences, log_pi, log_A, log_B, log_alpha, log_beta, log_E, log_gamma):
    """Return A, B and C after EM step."""
    # E STEP (over all sequences)
    # Accumulators for parameters of the hmm. They are summed over for
    # each sequence, then normalized in the M-step.
    # These are homogeneous to probabilities, not log-probabilities.
    pi_acc = np.zeros_like(log_pi)
    A_acc = np.zeros_like(log_A)
    B_acc = np.zeros_like(log_B)

    n_obs = sequences.shape[1]
    n_hidden_states = log_pi.shape[0]

    for seq_idx in range(sequences.shape[0]):
        seq = sequences[seq_idx]
        _forward(seq, log_pi, log_A, log_B, log_alpha)
        _backward(seq, log_pi, log_A, log_B, log_beta)
        log_likelihood = _logsumexp(log_alpha[:, -1])

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
        A_acc += np.sum(np.exp(log_E), axis=-1)
        for t in range(n_obs):
            B_acc[:, seq[t]] += np.exp(log_gamma[:, t])

    # M STEP (mostly done in the accumulators already)
    pi = pi_acc / pi_acc.sum()
    # equivalent to X / X.sum(axis=1, keepdims=True) but not supported
    A = A_acc / A_acc.sum(axis=1).reshape(-1, 1)
    B = B_acc / B_acc.sum(axis=1).reshape(-1, 1)

    return pi, A, B
