from time import time

import numpy as np
import matplotlib.pyplot as plt

from hmmkay import HMM
from hmmkay._utils import _get_hmm_learn_model, _to_weird_format


n_hidden_states, n_observable_states = 10, 20
n_seq, n_obs = 1000, 100


def make_parameters(n_hidden_states, n_observable_states):

    pi = np.random.rand(n_hidden_states)
    pi /= pi.sum()

    A = np.random.rand(n_hidden_states, n_hidden_states)
    A /= A.sum(axis=1, keepdims=True)

    B = np.random.rand(n_hidden_states, n_observable_states)
    B /= B.sum(axis=1, keepdims=True)

    return pi, A, B


def _compile_code(sequences):
    """Run methods with with small data for JIT compilation"""
    print("Compiling numba code...")
    sequences = sequences[:2]
    hmm.log_likelihood(sequences)
    hmm.decode(sequences, return_log_probas=True)
    hmm.fit(sequences)
    hmm.sample(2, 2)
    print("done")


init_probas, transitions, emissions = make_parameters(
    n_hidden_states, n_observable_states
)

hmm = HMM(init_probas, transitions, emissions, n_iter=1)
hmmlearn_model = _get_hmm_learn_model(hmm)

sequences = np.random.randint(n_observable_states, size=(n_seq, n_obs))

_compile_code(sequences)

tic = time()
log_likelihood = hmm.log_likelihood(sequences)
log_likelihood_time = time() - tic
print(f"log_likelihood computed in {log_likelihood_time:3f} sec")

tic = time()
log_likelihood = hmmlearn_model.score(sequences)
log_likelihood_time_learn = time() - tic
toc = time()
print(f"log_likelihood computed in {log_likelihood_time_learn:3f} sec")

tic = time()
hmm.decode(sequences, return_log_probas=True)
decode_time = time() - tic
print(f"decode run in {decode_time:3f} sec")

tic = time()
hmmlearn_model.decode(**_to_weird_format(sequences), algorithm="viterbi")
decode_time_learn = time() - tic
print(f"decode run in {decode_time_learn:3f} sec")

tic = time()
hmm.sample(n_seq, n_obs)
sample_time = time() - tic
print(f"sample run in {sample_time:3f} sec")

tic = time()
for _ in range(n_seq):
    hmmlearn_model.sample(n_obs)
sample_time_learn = time() - tic
print(f"sample run in {sample_time_learn:3f} sec")

tic = time()
hmm.fit(sequences)
fit_time = time() - tic
print(f"fit run in {fit_time:3f} sec")

tic = time()
hmmlearn_model.fit(**_to_weird_format(sequences))
fit_time_learn = time() - tic
print(f"fit run in {fit_time_learn:3f} sec")


def plot_times():
    times = (log_likelihood_time, decode_time, sample_time, fit_time)
    times_learn = (
        log_likelihood_time_learn,
        decode_time_learn,
        sample_time_learn,
        fit_time_learn,
    )

    x_pos = np.arange(len(times))  # the x locations for the groups
    width = 0.35  # width of the bars

    _, ax = plt.subplots()
    rects1 = ax.bar(x_pos, times, width)

    rects2 = ax.bar(x_pos + width, times_learn, width)

    # add some text for labels, title and axes ticks
    ax.set_ylabel("Time in sec")
    title = (
        f"Comparison with hmmlearn.\nn_hidden_states={n_hidden_states},  "
        f"n_observable_states={n_observable_states}  "
        f"n_sequences={n_seq},  n_observations={n_obs}"
    )
    ax.set_title(title)
    ax.set_xticks(x_pos + width / 2)

    improvements = [
        f"{hmmlearn_time / hmmkay_time:.1f}x faster"
        for (hmmkay_time, hmmlearn_time) in zip(times, times_learn)
    ]
    xticklabels = [
        fun_name + "\n" + improvement
        for (fun_name, improvement) in zip(
            ("log_likelihood", "decode", "sample", "fit"), improvements
        )
    ]
    ax.set_xticklabels(xticklabels)

    ax.legend((rects1[0], rects2[0]), ("hmmkay", "hmmlearn"))

    plt.show()


plot_times()
