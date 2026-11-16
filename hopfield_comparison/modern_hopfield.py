import numpy as np


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class ModernHopfield:
    def __init__(self, beta=1.0):
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.beta = beta
        self.P = None

    def store(self, patterns):
        self.P = np.array(patterns, dtype=float)

    def energy(self, x):
        scores = self.beta * self.P @ x
        m = scores.max()
        lse = m + np.log(np.exp(scores - m).sum())
        return -lse / self.beta + 0.5 * x @ x

    def recall(self, x, steps=1):
        x = x.astype(float)
        for _ in range(steps):
            w = softmax(self.beta * self.P @ x)
            x = w @ self.P
        return x
