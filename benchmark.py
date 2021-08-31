from time import time
from warnings import simplefilter

import numpy as np
import matplotlib.pyplot as plt

from hmmkay import HMM
from hmmkay.utils import (
    get_hmm_learn_model,
    to_weird_format,
    make_proba_matrices,
    make_observation_sequences,
)


n_hidden_states, n_observable_states = 10, 20
n_seq, n_obs_min, n_obs_max = 1000, 100, 150

sequences = make_observation_sequences(n_seq, n_observable_states, n_obs_min, n_obs_max)
sequences_hmmlearn = to_weird_format(sequences)


def _compile_code(sequences):
    """Run methods with with small data for JIT compilation"""
    print("Compiling numba code...")
    sequences = sequences[:2]
    hmm.log_likelihood(sequences)
    hmm.decode(sequences)
    hmm.fit(sequences)
    hmm.sample(2, 2)
    print("done")


init_probas, transitions, emissions = make_proba_matrices(
    n_hidden_states, n_observable_states
)

hmm = HMM(init_probas, transitions, emissions, n_iter=1)
hmmlearn_model = get_hmm_learn_model(hmm)

# filter out deprecation warnings from sklearn
simplefilter("ignore", category=DeprecationWarning)

_compile_code(sequences)

hmmkay_times = {}
hmmlearn_times = {}

tic = time()
log_likelihood = hmm.log_likelihood(sequences)
hmmkay_times["log_likelihood"] = time() - tic
print(f"log_likelihood computed in {hmmkay_times['log_likelihood']:3f} sec")

tic = time()
log_likelihood = hmmlearn_model.score(**sequences_hmmlearn)
hmmlearn_times["log_likelihood"] = time() - tic
print(f"log_likelihood computed in {hmmlearn_times['log_likelihood']:3f} sec")

tic = time()
hmm.decode(sequences)
hmmkay_times["decode"] = time() - tic
print(f"decode computed in {hmmkay_times['decode']:3f} sec")

tic = time()
hmmlearn_model.decode(**sequences_hmmlearn, algorithm="viterbi")
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
hmmlearn_model.fit(**sequences_hmmlearn)
hmmlearn_times["fit"] = time() - tic
print(f"fit computed in {hmmlearn_times['fit']:3f} sec")


def plot_times():

    x_pos = np.arange(len(hmmkay_times))  # the x locations for the groups
    width = 0.35  # width of the bars

    _, ax = plt.subplots()
    rects1 = ax.bar(x_pos, hmmkay_times.values(), width)

    rects2 = ax.bar(x_pos + width, hmmlearn_times.values(), width)

    # add some text for labels, title and axes ticks
    ax.set_ylabel("Time in sec", fontsize=15)
    title = (
        f"Comparison with hmmlearn.\nn_hidden_states={n_hidden_states},  "
        f"n_observable_states={n_observable_states}  "
        f"n_sequences={n_seq},  n_observations>={n_obs_min}"
    )
    ax.set_title(title, fontsize=15)
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
    ax.set_xticklabels(xticklabels, fontsize=15)

    ax.legend((rects1[0], rects2[0]), ("hmmkay", "hmmlearn"))

    plt.show()


plot_times()
