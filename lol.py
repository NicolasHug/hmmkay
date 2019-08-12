import hmmlearn.hmm
import numpy as np

from hmmidunnomaybe import HMM

np.set_printoptions(formatter={"float": lambda x: f"{x:.3f}"})


def print_matrices(hmm):
    if isinstance(hmm, hmmlearn.hmm.MultinomialHMM):
        print(f"pi:\n{hmm.startprob_}")
        print(f"A:\n{hmm.transmat_}")
        print(f"B:\n{hmm.emissionprob_}")
    else:
        print(f"pi:\n{hmm.pi}")
        print(f"A:\n{hmm.A}")
        print(f"B:\n{hmm.B}")


pi = np.array([0.6, 0.4])
A = np.array([[0.7, 0.3], [0.4, 0.6]])
B = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])

n_iter = 1

model = hmmlearn.hmm.MultinomialHMM(
    n_components=A.shape[0], init_params="", tol=0, n_iter=n_iter
)
model.startprob_ = pi
model.transmat_ = A
model.emissionprob_ = B

hmm = HMM(pi, A, B, n_iter=n_iter)

X = [[0, 1, 2, 0, 1, 2, 0, 1], [0, 1, 2, 0, 1, 2, 0, 1]]
# print(model.score(X)),
# print(np.log(hmm.likelihood(X)))

# X = np.array([[0, 0, 1, 1, 1, 2, 0, 1]])
# print(model.decode(X.T)[1])
# print(hmm.decode(X[0]))


# X = hmm.sample(n_seq=1, n_obs=5, seed=1)
X = [[0, 1, 2], [2, 1, 0]]


hmm.EM(X)
model.fit(np.array(X).ravel().reshape(-1, 1), lengths=[3, 3])
print()
print()
print("AFTER")
print_matrices(hmm)
print()
print_matrices(model)
