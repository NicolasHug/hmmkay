import numpy as np

class HMM:

    def __init__(self, init_probas=None, transitions=None, emissions=None):

        # bad names, see Rabiner's paper.
        self.pi = np.array(init_probas)
        self.A = np.array(transitions)  # A[i, j] = P(st+1 = j / st = i)
        self.B = np.array(emissions)  # B[i, o] = P(ot = o / st = i)

        self.n_hidden_states = self.A.shape[0]
        self.n_observable_states = self.B.shape[1]

        if not (self.A.shape[0] == self.A.shape[1] == self.pi.shape[0] ==
                    self.B.shape[0]):
            raise ValueError("inconsistent number of hidden states.")

        self._check_matrices_sum_to_1()

    def _check_matrices_sum_to_1(self):

        _check_array_sums_to_1(self.pi, 'init_probas')
        for s in range(self.n_hidden_states):
            _check_array_sums_to_1(self.A[s], f"Row {s} of A")
            _check_array_sums_to_1(self.B[s], f"Row {s} of B")

    def likelihood(self, sequences):
        sequences = np.array(sequences)
        n_obs = sequences.shape[1]
        alpha = np.empty(shape=(self.n_hidden_states, n_obs))
        out = 1
        for seq in sequences:
            self._forward(seq, alpha)
            out *= alpha[:, -1].sum()
        return out

    def decode(self, seq):
        seq = np.array(seq)
        n_obs = seq.shape[0]
        V = np.empty(shape=(self.n_hidden_states, n_obs))
        back_path  = np.empty(shape=(self.n_hidden_states, n_obs),
                              dtype=np.int32)

        self._viterbi(seq, V, back_path)

        best_path = []
        s = np.argmax(V[:, -1])
        for t in range(n_obs - 1, -1, -1):
            best_path.append(s)
            s = back_path[s, t]
        return np.array(list(reversed(best_path)))

    def _viterbi(self, seq, V, back_path):
        V[:, 0] = self.pi * self.B[:, seq[0]]
        n_obs = V.shape[1]
        for t in range(1, n_obs):
            for s in range(self.n_hidden_states):
                zob = V[:, t -1] * self.A[:, s]
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

    def EM(self, seq, n_iter=100):
        seq = np.array(seq)
        n_obs = seq.shape[0]
        alpha = np.empty(shape=(self.n_hidden_states, n_obs))
        beta = np.empty(shape=(self.n_hidden_states, n_obs))

        E = np.empty(shape=(self.n_hidden_states, self.n_hidden_states, n_obs - 1))
        gamma = np.empty(shape=(self.n_hidden_states, n_obs))

        for _ in range(100):
            # E STEP
            likelihood = self.likelihood([seq])  # TODO: don't waste forward pass
            self._forward(seq, alpha)
            self._backward(seq, beta)
            for t in range(n_obs - 1):
                for i in range(self.n_hidden_states):
                    for j in range(self.n_hidden_states):
                        E[i, j, t] = \
                            alpha[i, t] * self.A[i, j] * self.B[j, seq[t + 1]] * beta[j, t + 1]
            E /= likelihood

            for t in range(E.shape[-1]):
                _check_array_sums_to_1(E[:, :, t], f'E at t={t}')

            gamma = alpha * beta / likelihood

            for t in range(n_obs - 1):
                for i in range(self.n_hidden_states):
                    assert np.allclose(gamma[i, t], np.sum(E[i, :, t]))

            for t in range(gamma.shape[1]):
                _check_array_sums_to_1(gamma[:, t], f'Col {t} of gamma')

            # M STEP
            self.pi = gamma[:, 0]

            self.A = np.sum(E, axis=-1)
            for i in range(self.n_hidden_states):
                self.A[i] /= np.sum(gamma[i, :-1])

            for o in range(self.n_observable_states):
                for i in range(self.n_hidden_states):
                    self.B[i, o] = np.sum(gamma[i, :] * (seq == o), axis=0) / np.sum(gamma[i, :], axis=0)

            self._check_matrices_sum_to_1()

    def print_matrices(self):
        print(f"pi:\n{self.pi}")
        print(f"A:\n{self.A}")
        print(f"B:\n{self.B}")

def _check_array_sums_to_1(a, name='array'):
    a_sum = a.sum()
    if not (1 - 1e-5 < a_sum < 1 + 1e-5):
        err_msg = f"{name} must sum to 1. Got \n{a}.sum() = {a_sum}"
        raise ValueError(err_msg)