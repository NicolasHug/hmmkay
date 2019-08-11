import numpy as np
import pytest

from hmmidunnomaybe import HMM


def test_likelihoods():
    # Make sure computing likelihood with just alpha is the same as computing
    # it with alpha and beta

    def likelihood2(hmm, seq, t):
        sequences = np.array(seq)
        n_obs = sequences.shape[0]
        alpha = np.empty(shape=(hmm.n_hidden_states, n_obs))
        beta = np.empty(shape=(hmm.n_hidden_states, n_obs))
        hmm._forward(seq, alpha)
        hmm._backward(seq, beta)
        return np.sum(alpha[:, t] * beta[:, t])


    pi = np.array([0.6, 0.4])
    A = np.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])
    B = np.array([
        [0.1, 0.4, 0.5],
        [0.6, 0.3, 0.1]
    ])

    hmm = HMM(pi, A, B)
    X = np.array([[0, 1, 2, 0, 1, 2, 0, 1]])
    likelihood = hmm.likelihood(X)
    for t in range(X.shape[1]):
        assert likelihood == pytest.approx(likelihood2(hmm, X[0], t))


def test_sample():

    pi = [.9, .1]
    A = [[.9, .1],
         [.2, .8]]
    B = [[1, 0],
         [0, 1]]
    hmm = HMM(pi, A, B)

    n_seq, n_obs = 1000, 1
    samples = hmm.sample(n_seq=n_seq, n_obs=n_obs)
    assert samples.shape == (n_seq, n_obs)

    assert np.mean(samples) == pytest.approx(.1, abs=1e-1)
