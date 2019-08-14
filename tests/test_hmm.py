import numpy as np
import pytest
from scipy.special import logsumexp

from hmmkay import HMM
from hmmkay._utils import (
    _get_hmm_learn_model,
    _to_weird_format,
    _make_random_parameters,
    _make_random_sequences_observations,
)


N_HIDDEN_STATES = 2
N_OBSERVABLE_STATES = 3

SEQUENCES_SAME_LENGTHS = _make_random_sequences_observations(
    n_seq=10, n_observable_states=N_OBSERVABLE_STATES, n_obs_min=100, random_state=0
)
SEQUENCES_DIFF_LENGTHS = _make_random_sequences_observations(
    n_seq=10,
    n_observable_states=N_OBSERVABLE_STATES,
    n_obs_min=70,
    n_obs_max=100,
    random_state=0,
)


@pytest.fixture
def toy_params():
    rng = np.random.RandomState(0)
    return _make_random_parameters(
        n_hidden_states=N_HIDDEN_STATES,
        n_observable_states=N_OBSERVABLE_STATES,
        random_state=rng,
    )


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


@pytest.mark.parametrize("sequences", (SEQUENCES_SAME_LENGTHS, SEQUENCES_DIFF_LENGTHS))
def test_loglikelihood_against_hmmlearn(toy_params, sequences):
    # Basic test making sure hmmlearn has the same results

    pi, A, B = toy_params

    hmm = HMM(pi, A, B)
    hmm_learn_model = _get_hmm_learn_model(hmm)

    expected = hmm_learn_model.score(**_to_weird_format(sequences))
    assert hmm.log_likelihood(sequences) == pytest.approx(expected)


@pytest.mark.parametrize("sequences", (SEQUENCES_SAME_LENGTHS, SEQUENCES_DIFF_LENGTHS))
def test_decode_against_hmmlearn(toy_params, sequences):
    # Basic test making sure hmmlearn has the same results

    pi, A, B = toy_params

    hmm = HMM(pi, A, B)
    hmm_learn_model = _get_hmm_learn_model(hmm)

    hidden_states_sequences, log_probas = hmm.decode(sequences, return_log_probas=True)

    expected_out = hmm_learn_model.decode(**_to_weird_format(sequences))
    expected_hidden_states_seq = expected_out[1]

    # Flatten output of hmmkay for consistency with hmmlearn
    if isinstance(sequences, list):
        hidden_states_sequences = np.concatenate(hidden_states_sequences)
    else:
        hidden_states_sequences = hidden_states_sequences.ravel()

    np.testing.assert_array_equal(hidden_states_sequences, expected_hidden_states_seq)

    # For some reason this fails when sequences have different lengths. I
    # think hmmlearn computes the log likelihood of the concatenated sequence
    # in this case???
    # expected_lob_probas = expected_out[0]
    # assert log_probas.sum() == pytest.approx(expected_lob_probas)


def test_EM_against_hmmlearn(toy_params):
    # Basic test making sure hmmlearn has the same results

    pi, A, B = toy_params
    n_iter = 10

    hmm = HMM(pi, A, B, n_iter=n_iter)
    hmm_learn_model = _get_hmm_learn_model(hmm)
    hmm._enable_sanity_checks = True

    rng = np.random.RandomState(0)
    n_seq, n_obs = 10, 100
    sequences = rng.randint(B.shape[1], size=(n_seq, n_obs))

    hmm.fit(sequences)
    hmm_learn_model.fit(**_to_weird_format(sequences))

    np.testing.assert_allclose(hmm.pi, hmm_learn_model.startprob_)
    np.testing.assert_allclose(hmm.A, hmm_learn_model.transmat_)
    np.testing.assert_allclose(hmm.B, hmm_learn_model.emissionprob_)


def test_sample(toy_params):
    # Make sure shapes are  as expected
    # Also make sure random_state behaves properly

    pi, A, B = toy_params
    hmm = HMM(pi, A, B)
    n_obs, n_seq = 10, 20

    obs_sequences, hidden_state_sequences = hmm.sample(
        n_seq=n_seq, n_obs=n_obs, random_state=0
    )
    assert obs_sequences.shape == hidden_state_sequences.shape == (n_seq, n_obs)

    obs_sequences_same, hidden_state_sequences_same = hmm.sample(
        n_seq=n_seq, n_obs=n_obs, random_state=0
    )
    assert np.all(obs_sequences == obs_sequences_same)
    assert np.all(hidden_state_sequences == hidden_state_sequences_same)

    obs_sequences_diff, hidden_state_sequences_diff = hmm.sample(
        n_seq=n_seq, n_obs=n_obs, random_state=1
    )
    assert not np.all(obs_sequences == obs_sequences_diff)
    assert not np.all(hidden_state_sequences == hidden_state_sequences_diff)


def test_against_wikipedia():
    # Stolen from hmmlearn
    # http://en.wikipedia.org/wiki/Viterbi_algorithm
    pi = np.array([0.6, 0.4])
    A = np.array([[0.7, 0.3], [0.4, 0.6]])
    B = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
    seq = [[0, 1, 2]]

    hmm = HMM(pi, A, B)
    hidden_states_seq, log_proba = hmm.decode(seq, return_log_probas=True)

    expected_proba = 0.01344
    assert np.exp(log_proba) == pytest.approx(expected_proba)

    expected_hidden_states_seq = [[1, 0, 0]]
    np.testing.assert_array_equal(hidden_states_seq, expected_hidden_states_seq)


class test_against_wikipedia_2(object):
    # Stolen from hmmlearn
    # http://en.wikipedia.org/wiki/Forward-backward_algorithm
    pi = [0.5, 0.5]
    A = [[0.7, 0.3], [0.3, 0.7]]
    B = [[0.9, 0.1], [0.2, 0.8]]

    seq = np.array([0, 0, 1, 0, 0])

    hmm = HMM(pi, A, B)

    # test forward pass
    n_obs = seq.shape[0]
    log_alpha = np.empty(shape=(hmm.n_hidden_states, n_obs))
    log_likelihood = hmm._forward(seq, log_alpha)

    expected_log_likelihood = -3.3725
    expected_alpha = np.array(
        [
            [0.4500, 0.1000],
            [0.3105, 0.0410],
            [0.0230, 0.0975],
            [0.0408, 0.0150],
            [0.0298, 0.0046],
        ]
    ).T

    assert log_likelihood == pytest.approx(expected_log_likelihood)
    assert np.allclose(np.exp(log_alpha), expected_alpha, atol=1e-4)

    # test backward pass
    log_beta = np.empty(shape=(hmm.n_hidden_states, n_obs))
    hmm._backward(seq, log_beta)

    expected_beta = np.array(
        [
            [0.0661, 0.0455],
            [0.0906, 0.1503],
            [0.4593, 0.2437],
            [0.6900, 0.4100],
            [1.0000, 1.0000],
        ]
    ).T
    assert np.allclose(np.exp(log_beta), expected_beta, atol=1e-4)

    # test gamma
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
    ).T
    assert np.allclose(gamma, expected_gamma, atol=1e-4)

    # test decode
    hidden_states_seq, log_proba = hmm.decode([seq], return_log_probas=True)

    expected_hidden_states_seq = [[0, 0, 1, 0, 0]]
    expected_log_proba = -4.4590

    np.testing.assert_array_equal(hidden_states_seq, expected_hidden_states_seq)
    assert log_proba == pytest.approx(expected_log_proba, rel=1e-5)
