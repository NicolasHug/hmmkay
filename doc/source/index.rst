.. hmmkay documentation master file, created by
   sphinx-quickstart on Tue Aug 13 18:22:58 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to hmmkay's documentation!
==================================

Hmmkay is a basic library for discrete Hidden Markov Models that relies on
numba's just-in-time compilation. It supports decoding, likelihood scoring,
and fitting (parameter estimation).

Hmmkay accepts sequences of arbitrary length, e.g. 2d numpy arrays or lists
of iterables. Hmmkay internally converts lists of iterables into Numba typed
lists of numpy arrays (you might want to do that yourself to avoid repeated
convertions using :func:`hmmkay.utils.check_sequences`)

Scoring and decoding example::

    >>> from hmmkay.utils import make_proba_matrices
    >>> from hmmkay import HMM

    >>> init_probas, transition_probas, emission_probas = make_proba_matrices(
    ...     n_hidden_states=2,
    ...     n_observable_states=4,
    ...     random_state=0
    ... )
    >>> hmm = HMM(init_probas, transition_probas, emission_probas)

    >>> sequences = [[0, 1, 2, 3], [0, 2]]
    >>> hmm.log_likelihood(sequences)
    -8.336
    >>> hmm.decode(sequences)  # most likely sequences of hidden states
    [array([1, 0, 0, 1], dtype=int32), array([1, 0], dtype=int32)]

Fitting example::

    >>> from hmmkay.utils import make_observation_sequences
    >>> sequences = make_observation_sequences(n_seq=100, n_observable_states=4, random_state=0)
    >>> hmm.fit(sequences)

Sampling example::

    >>> hmm.sample(n_obs=2, n_seq=3)  # return sequences of hidden and observable states
    ... (array([[0, 1],
    ...         [1, 1],
    ...         [0, 0]]), array([[0, 2],
    ...         [2, 3],
    ...         [0, 0]]))

API Reference
=============

HMM class
---------

.. automodule:: hmmkay
   :members:

Utils
-----

.. automodule:: hmmkay.utils
   :members:


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
