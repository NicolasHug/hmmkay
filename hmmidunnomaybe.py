import numpy as np


class HMM:
    def __init__(self, init_probas=None, transitions=None, emissions=None, n_iter=10):

        # bad names, see Rabiner's paper.
        self.pi = np.array(init_probas)
        self.A = np.array(transitions)
        self.B = np.array(emissions)
        self.n_iter = n_iter

        self.n_hidden_states = self.A.shape[0]
        self.n_observable_states = self.B.shape[1]

        if not (
            self.A.shape[0] == self.A.shape[1] == self.pi.shape[0] == self.B.shape[0]
        ):
            raise ValueError("inconsistent number of hidden states.")

        self._check_matrices_conditioning()

    def likelihood(self, sequences):
        sequences = np.array(sequences)
        n_obs = sequences.shape[1]
        alpha = np.empty(shape=(self.n_hidden_states, n_obs))
        out = 1
        for seq in sequences:
            self._forward(seq, alpha)
            out *= alpha[:, -1].sum()
        return out

    def decode(self, sequences):
        return np.array([self._decode_one(seq) for seq in sequences])

    def _decode_one(self, seq):
        n_obs = seq.shape[0]
        V = np.empty(shape=(self.n_hidden_states, n_obs))
        back_path = np.empty(shape=(self.n_hidden_states, n_obs), dtype=np.int32)

        self._viterbi(seq, V, back_path)

        best_path = []
        s = np.argmax(V[:, -1])
        for t in range(n_obs - 1, -1, -1):
            best_path.append(s)
            s = back_path[s, t]
        return list(reversed(best_path))

    def _viterbi(self, seq, V, back_path):
        V[:, 0] = self.pi * self.B[:, seq[0]]
        n_obs = V.shape[1]
        for t in range(1, n_obs):
            for s in range(self.n_hidden_states):
                zob = V[:, t - 1] * self.A[:, s]
                best_prev = np.argmax(zob)
                back_path[s, t] = best_prev
                V[s, t] = zob[best_prev] * self.B[s, seq[t]]

    def _forward(self, seq, alpha):
        alpha[:, 0] = self.pi * self.B[:, seq[0]]
        n_obs = alpha.shape[1]
        for t in range(1, n_obs):
            for s in range(self.n_hidden_states):
                alpha[s, t] = np.sum(alpha[:, t - 1] * self.A[:, s])
                alpha[s, t] *= self.B[s, seq[t]]

    def _backward(self, seq, beta):
        beta[:, -1] = 1
        n_obs = beta.shape[1]
        for t in range(n_obs - 2, -1, -1):
            for s in range(self.n_hidden_states):
                beta[s, t] = np.sum(
                    self.A[s, :] * self.B[:, seq[t + 1]] * beta[:, t + 1]
                )

    def sample(self, n_seq, n_obs, seed=0):

        rng = np.random.RandomState(seed)

        sequences = []
        for _ in range(n_seq):
            observations = []
            s = rng.choice(self.n_hidden_states, p=self.pi)
            for _ in range(n_obs):
                obs = rng.choice(self.n_observable_states, p=self.B[s])
                observations.append(obs)
                s = rng.choice(self.n_hidden_states, p=self.A[s])
            sequences.append(observations)
        return np.array(sequences)

    def EM(self, sequences, n_iter=100):
        sequences = np.array(sequences)
        n_seq, n_obs = sequences.shape
        alpha = np.empty(shape=(self.n_hidden_states, n_obs))
        beta = np.empty(shape=(self.n_hidden_states, n_obs))

        for _ in range(self.n_iter):
            # E STEP
            # E[i, j, t] = P(st = i, st+1 = j / O, lambda)
            E = np.empty(shape=(self.n_hidden_states, self.n_hidden_states, n_obs - 1))
            # g[i, t] = P(st = i / O, lambda)
            gamma = np.empty(shape=(self.n_hidden_states, n_obs))

            # Accumulators for parameters of the hmm. They are summed over for
            # each sequence, then normalized in the M-step
            pi_acc = np.zeros_like(self.pi)
            A_acc = np.zeros_like(self.A)
            B_acc = np.zeros_like(self.B)

            for seq in sequences:
                self._forward(seq, alpha)
                self._backward(seq, beta)
                likelihood = alpha[:, -1].sum()

                # Compute E
                for t in range(n_obs - 1):
                    for i in range(self.n_hidden_states):
                        for j in range(self.n_hidden_states):
                            E[i, j, t] = (
                                alpha[i, t]
                                * self.A[i, j]
                                * self.B[j, seq[t + 1]]
                                * beta[j, t + 1]
                                / likelihood
                            )

                # compute gamma
                gamma = alpha * beta / likelihood

                if getattr(self, "_enable_sanity_checks", False):
                    for t in range(E.shape[-1]):
                        _check_array_sums_to_1(E[:, :, t], f"E at t={t}")

                    for t in range(gamma.shape[1]):
                        _check_array_sums_to_1(gamma[:, t], f"gamma at t={t}")

                    for t in range(n_obs - 1):
                        for i in range(self.n_hidden_states):
                            assert np.allclose(gamma[i, t], np.sum(E[i, :, t]))

                # M STEP accumulators
                pi_acc += gamma[:, 0]
                A_acc += np.sum(E, axis=-1)
                for t in range(n_obs):
                    B_acc[:, seq[t]] += gamma[:, t]

            # M STEP (mostly done in the accumulators already)
            self.pi = _normalize(pi_acc)
            self.A = _normalize(A_acc, axis=1)
            self.B = _normalize(B_acc, axis=1)

            if getattr(self, "_enable_sanity_checks", False):
                self._check_matrices_conditioning()

    def _check_matrices_conditioning(self):

        _check_array_sums_to_1(self.pi, "init_probas")
        for s in range(self.n_hidden_states):
            _check_array_sums_to_1(self.A[s], f"Row {s} of A")
            _check_array_sums_to_1(self.B[s], f"Row {s} of B")


def _check_array_sums_to_1(a, name="array"):
    a_sum = a.sum()
    if not (1 - 1e-5 < a_sum < 1 + 1e-5):
        err_msg = f"{name} must sum to 1. Got \n{a}.sum() = {a_sum}"
        raise ValueError(err_msg)


def _normalize(a, axis=None):
    return a / a.sum(axis, keepdims=True)
