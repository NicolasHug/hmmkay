from hmmlearn.hmm import GaussianHMM
from hmmlearn import hmm
import numpy as np

from hmmidunnomaybe import HMM


pi = np.array([0.6, 0.4])
A = np.array([
  [0.7, 0.3],
  [0.4, 0.6]
])
B = np.array([
  [0.1, 0.4, 0.5],
  [0.6, 0.3, 0.1]
])

model = hmm.MultinomialHMM(n_components=A.shape[0], init_params="")
model.startprob_ = pi
model.transmat_ = A
model.emissionprob_ = B

hmm = HMM(pi, A, B)

X = [[0, 1, 2, 0, 1, 2, 0, 1],
     [0, 1, 2, 0, 1, 2, 0, 1]]
# print(model.score(X)),
# print(np.log(hmm.likelihood(X)))

# X = np.array([[0, 0, 1, 1, 1, 2, 0, 1]])
# print(model.decode(X.T)[1])
# print(hmm.decode(X[0]))



# X = [0, 1, 2, 0, 1, 2, 0, 1]
# hmm.EM(X)

X = hmm.sample(n_seq=1, n_obs=5, seed=1)
print(X)
hmm.print_matrices()
hmm.EM(X[0])
print()
hmm.print_matrices()

# for _ in range(10):
#      print(hmm.sample(n_seq=1, n_obs=5, seed=None))