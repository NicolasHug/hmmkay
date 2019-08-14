from time import time

import numpy as np
import matplotlib.pyplot as plt

from hmmkay import HMM
from hmmkay._utils import (
    _get_hmm_learn_model,
    _to_weird_format,
    _make_random_parameters,
    _make_random_sequences_observations,
)


n_hidden_states, n_observable_states = 10, 20
n_seq, n_obs_min, n_obs_max = 1000, 100, 150

sequences = _make_random_sequences_observations(
    n_seq, n_observable_states, n_obs_min, n_obs_max
)


def _compile_code(sequences):
    """Run methods with with small data for JIT compilation"""
    print("Compiling numba code...")
    sequences = sequences[:2]
    hmm.log_likelihood(sequences)
    hmm.decode(sequences, return_log_probas=True)
    hmm.fit(sequences)
    hmm.sample(2, 2)
    print("done")


init_probas, transitions, emissions = _make_random_parameters(
    n_hidden_states, n_observable_states
)

hmm = HMM(init_probas, transitions, emissions, n_iter=1)
hmmlearn_model = _get_hmm_learn_model(hmm)

_compile_code(sequences)

hmmkay_times = {}
hmmlearn_times = {}

tic = time()
log_likelihood = hmm.log_likelihood(sequences)
hmmkay_times["log_likelihood"] = time() - tic
print(f"log_likelihood computed in {hmmkay_times['log_likelihood']:3f} sec")

tic = time()
log_likelihood = hmmlearn_model.score(**_to_weird_format(sequences))
hmmlearn_times["log_likelihood"] = time() - tic
print(f"log_likelihood computed in {hmmlearn_times['log_likelihood']:3f} sec")

tic = time()
hmm.decode(sequences, return_log_probas=True)
hmmkay_times["decode"] = time() - tic
print(f"decode computed in {hmmkay_times['decode']:3f} sec")

tic = time()
hmmlearn_model.decode(**_to_weird_format(sequences), algorithm="viterbi")
hmmlearn_times["decode"] = time() - tic
print(f"decode computed in {hmmlearn_times['decode']:3f} sec")

tic = time()
hmm.sample(n_seq, n_obs_min)
hmmkay_times["sample"] = time() - tic
print(f"sample computed in {hmmkay_times['sample']:3f} sec")

tic = time()
for _ in range(n_seq):
    hmmlearn_model.sample(n_obs_min)
hmmlearn_times["sample"] = time() - tic
print(f"sample computed in {hmmlearn_times['sample']:3f} sec")

tic = time()
hmm.fit(sequences)
hmmkay_times["fit"] = time() - tic
print(f"fit computed in {hmmkay_times['fit']:3f} sec")

tic = time()
hmmlearn_model.fit(**_to_weird_format(sequences))
hmmlearn_times["fit"] = time() - tic
print(f"fit computed in {hmmlearn_times['fit']:3f} sec")


def plot_times():

    x_pos = np.arange(len(hmmkay_times))  # the x locations for the groups
    width = 0.35  # width of the bars

    _, ax = plt.subplots()
    rects1 = ax.bar(x_pos, hmmkay_times.values(), width)

    rects2 = ax.bar(x_pos + width, hmmlearn_times.values(), width)

    # add some text for labels, title and axes ticks
    ax.set_ylabel("Time in sec")
    title = (
        f"Comparison with hmmlearn.\nn_hidden_states={n_hidden_states},  "
        f"n_observable_states={n_observable_states}  "
        f"n_sequences={n_seq},  n_observations>={n_obs_min}"
    )
    ax.set_title(title)
    ax.set_xticks(x_pos + width / 2)

    improvements = [
        f"{hmmlearn_time / hmmkay_time:.1f}x faster"
        for (hmmkay_time, hmmlearn_time) in zip(
            hmmkay_times.values(), hmmlearn_times.values()
        )
    ]
    xticklabels = [
        fun_name + "\n" + improvement
        for (fun_name, improvement) in zip(hmmkay_times.keys(), improvements)
    ]
    ax.set_xticklabels(xticklabels)

    ax.legend((rects1[0], rects2[0]), ("hmmkay", "hmmlearn"))

    plt.show()


plot_times()
