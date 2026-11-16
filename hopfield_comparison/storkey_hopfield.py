"""Storkey learning with the classical Hopfield energy and recall methods."""
import numpy as np
from hopfield import Hopfield


class StorkeyHopfield(Hopfield):
    def train(self, patterns):
        patterns = np.asarray(patterns, dtype=float)
        if patterns.ndim != 2 or patterns.shape[1] != self.n:
            raise ValueError("patterns must have shape (number_of_patterns, n)")
        if not np.all((patterns == -1) | (patterns == 1)):
            raise ValueError("Storkey patterns must contain only -1 and +1")

        self.W.fill(0)
        for p in patterns:
            # All quantities below use W BEFORE this pattern's update.
            field = self.W @ p

            # h[i, j]: input to neuron i excluding neuron j.
            # Multiplication broadcasts p across columns; it is not a dot product.
            h = field[:, None] - self.W * p[None, :]

            # correction[i, j] = h[i, j]*p[j] + p[i]*h[j, i]
            correction = h * p[None, :] + h.T * p[:, None]
            delta_W = (np.outer(p, p) - correction) / self.n
            np.fill_diagonal(delta_W, 0)
            self.W += delta_W
