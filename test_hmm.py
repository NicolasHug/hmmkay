import numpy as np
import pytest
import hmmlearn.hmm
from scipy.special import logsumexp

from hmmidunnomaybe import HMM


def get_hmm_learn_model(hmm):
    hmm_learn_model = hmmlearn.hmm.MultinomialHMM(
        n_components=hmm.A.shape[0], init_params="", tol=0, n_iter=hmm.n_iter
    )
    hmm_learn_model.startprob_ = hmm.pi
    hmm_learn_model.transmat_ = hmm.A
    hmm_learn_model.emissionprob_ = hmm.B

    return hmm_learn_model


def to_weird_format(sequences):
    # Please don't ask
    return {
        "X": np.array(sequences).ravel().reshape(-1, 1),
        "lengths": [sequences.shape[1]] * sequences.shape[0],
    }


@pytest.fixture
def toy_params():
    # 2 hidden states, 3 observable states
    pi = np.array([0.6, 0.4])
    A = np.array([[0.7, 0.3], [0.4, 0.6]])
    B = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])

    return pi, A, B


def test_likelihood_alpha_beta(toy_params):
    # Make sure computing likelihood with just alpha is the same as computing
    # it with alpha and beta
    # For all t, the likelihood is equal to sum (alpha[:, t] * beta[:, t])

    def log_likelihood2(hmm, seq):
        sequences = np.array(seq)
        n_obs = sequences.shape[0]
        log_alpha = np.empty(shape=(hmm.n_hidden_states, n_obs))
        log_beta = np.empty(shape=(hmm.n_hidden_states, n_obs))
        hmm._forward(seq, log_alpha)
        hmm._backward(seq, log_beta)
        # return likelihoods computed at all ts
        return logsumexp(log_alpha + log_beta, axis=0)

    pi, A, B = toy_params

    hmm = HMM(pi, A, B)
    X = np.array([[0, 1, 2, 0, 1, 2, 0, 1]])
    expected_likelihood = hmm.log_likelihood(X)

    np.testing.assert_allclose(expected_likelihood, log_likelihood2(hmm, X[0]))


def test_loglikelihood(toy_params):
    # Basic test making sure hmmlearn has the same results

    pi, A, B = toy_params

    hmm = HMM(pi, A, B)
    hmm_learn_model = get_hmm_learn_model(hmm)

    rng = np.random.RandomState(0)
    n_seq, n_obs = 10, 100
    sequences = rng.randint(B.shape[1], size=(n_seq, n_obs))

    expected = hmm_learn_model.score(**to_weird_format(sequences))
    assert hmm.log_likelihood(sequences) == pytest.approx(expected)


def test_decode(toy_params):
    # Basic test making sure hmmlearn has the same results

    pi, A, B = toy_params

    hmm = HMM(pi, A, B)
    hmm_learn_model = get_hmm_learn_model(hmm)

    rng = np.random.RandomState(0)
    n_seq, n_obs = 10, 100
    sequences = rng.randint(B.shape[1], size=(n_seq, n_obs))

    expected = hmm_learn_model.decode(**to_weird_format(sequences))[1].reshape(
        sequences.shape
    )
    assert np.all(hmm.decode(sequences) == expected)


def test_EM(toy_params):
    # Basic test making sure hmmlearn has the same results

    pi, A, B = toy_params
    n_iter = 10

    hmm = HMM(pi, A, B, n_iter=n_iter)
    hmm_learn_model = get_hmm_learn_model(hmm)
    hmm._enable_sanity_checks = True

    rng = np.random.RandomState(0)
    n_seq, n_obs = 10, 100
    sequences = rng.randint(B.shape[1], size=(n_seq, n_obs))

    hmm.EM(sequences)
    hmm_learn_model.fit(**to_weird_format(sequences))

    np.testing.assert_allclose(hmm.pi, hmm_learn_model.startprob_)
    np.testing.assert_allclose(hmm.A, hmm_learn_model.transmat_)
    np.testing.assert_allclose(hmm.B, hmm_learn_model.emissionprob_)


def test_sample(toy_params):
    # Make sure shapes are  as expected
    # Also make sure seed behaves properly

    pi, A, B = toy_params
    hmm = HMM(pi, A, B)
    n_obs, n_seq = 10, 20

    obs_sequences, hidden_state_sequences = hmm.sample(n_seq=n_seq, n_obs=n_obs, seed=0)
    assert obs_sequences.shape == hidden_state_sequences.shape == (n_seq, n_obs)

    obs_sequences_same, hidden_state_sequences_same = hmm.sample(
        n_seq=n_seq, n_obs=n_obs, seed=0
    )
    assert np.all(obs_sequences == obs_sequences_same)
    assert np.all(hidden_state_sequences == hidden_state_sequences_same)

    obs_sequences_diff, hidden_state_sequences_diff = hmm.sample(
        n_seq=n_seq, n_obs=n_obs, seed=1
    )
    assert not np.all(obs_sequences == obs_sequences_diff)
    assert not np.all(hidden_state_sequences == hidden_state_sequences_diff)


class TestAgainstWikipedia(object):
    # Stolen from hmmlearn
    # http://en.wikipedia.org/wiki/Hidden_Markov_model
    # http://en.wikipedia.org/wiki/Viterbi_algorithm
    def setup_method(self, method):
        pi = np.array([0.6, 0.4])
        A = np.array([[0.7, 0.3], [0.4, 0.6]])
        B = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
        self.hmm = HMM(pi, A, B)

    def test_decode(self):
        # From http://en.wikipedia.org/wiki/Viterbi_algorithm:
        X = [[0, 1, 2]]
        hidden_states_seq = self.hmm.decode(X)
        # TODO : test logprob when/if we return it
        # assert round(np.exp(logprob), 5) == 0.01344
        assert np.all(hidden_states_seq == [1, 0, 0])


class TestAgainstWikipediaAgain(object):
    # Stolen from hmmlearn
    # http://en.wikipedia.org/wiki/Forward-backward_algorithm
    def setup_method(self, method):
        pi = [0.5, 0.5]
        A = [[0.7, 0.3], [0.3, 0.7]]
        B = [[0.9, 0.1], [0.2, 0.8]]
        self.hmm = HMM(pi, A, B)

    def test_forward_pass(self):
        seq = np.array([0, 0, 1, 0, 0])
        n_obs = seq.shape[0]
        log_alpha = np.empty(shape=(self.hmm.n_hidden_states, n_obs))
        log_likelihood = self.hmm._forward(seq, log_alpha)

        expected_log_likelihood = -3.3725
        assert log_likelihood == pytest.approx(expected_log_likelihood)
        expected_alpha = np.array(
            [
                [0.4500, 0.1000],
                [0.3105, 0.0410],
                [0.0230, 0.0975],
                [0.0408, 0.0150],
                [0.0298, 0.0046],
            ]
        )
        assert np.allclose(np.exp(log_alpha), expected_alpha.T, atol=1e-4)

    def test_backward_pass(self):
        seq = np.array([0, 0, 1, 0, 0])
        n_obs = seq.shape[0]
        log_beta = np.empty(shape=(self.hmm.n_hidden_states, n_obs))
        self.hmm._backward(seq, log_beta)

        expected_beta = np.array(
            [
                [0.0661, 0.0455],
                [0.0906, 0.1503],
                [0.4593, 0.2437],
                [0.6900, 0.4100],
                [1.0000, 1.0000],
            ]
        )
        assert np.allclose(np.exp(log_beta), expected_beta.T, atol=1e-4)

    def test_decode(self):
        seq = [[0, 0, 1, 0, 0]]
        hidden_states_seq = self.hmm.decode(seq)

        expected_hidden_states_seq = [0, 0, 1, 0, 0]
        assert np.all(hidden_states_seq == expected_hidden_states_seq)

        # TODO: test that when supported
        # reflogprob = -4.4590
        # assert round(logprob, 4) == reflogprob

    def test_gamma(self):
        seq = np.array([0, 0, 1, 0, 0])
        n_obs = seq.shape[0]
        log_alpha = np.empty(shape=(self.hmm.n_hidden_states, n_obs))
        log_beta = np.empty(shape=(self.hmm.n_hidden_states, n_obs))
        log_likelihood = self.hmm._forward(seq, log_alpha)
        self.hmm._backward(seq, log_beta)

        expected_log_likelihood = -3.3725
        assert log_likelihood == pytest.approx(expected_log_likelihood)

        log_gamma = log_alpha + log_beta - log_likelihood
        gamma = np.exp(log_gamma)
        assert np.allclose(gamma.sum(axis=0), 1)

        expected_gamma = np.array(
            [
                [0.8673, 0.1327],
                [0.8204, 0.1796],
                [0.3075, 0.6925],
                [0.8204, 0.1796],
                [0.8673, 0.1327],
            ]
        )
        assert np.allclose(gamma, expected_gamma.T, atol=1e-4)
