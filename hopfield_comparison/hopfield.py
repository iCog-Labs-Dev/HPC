import numpy as np


class Hopfield:
    def __init__(self, n):
        self.n = n
        self.W = np.zeros((n, n))

    def train(self, patterns):
        # A train() call starts a fresh run; do not rescale old weights twice.
        self.W.fill(0)
        for p in patterns:
            self.W += np.outer(p, p)
        np.fill_diagonal(self.W, 0)
        self.W /= self.n

    def energy(self, x):
        return -0.5 * x @ self.W @ x

    def recall(self, x, sweeps=10):
        x = x.copy()
        for _ in range(sweeps):
            for i in np.random.permutation(self.n):
                x[i] = 1 if self.W[i] @ x >= 0 else -1
        return x
