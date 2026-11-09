"""Network initialization and forward prediction."""

import numpy as np


class XavierInitializer:
    """NumPy Xavier/Glorot initializer with normal or uniform weights.

    Uses the same variance formulas as FabricPC's XavierInitializer, while
    keeping this package independent of JAX. Returns (fan_out, fan_in).
    """

    def __init__(self, distribution="normal", gain=1.0):
        if distribution not in ("normal", "uniform"):
            raise ValueError("distribution must be 'normal' or 'uniform'.")
        if not np.isscalar(gain) or not np.isfinite(gain) or gain <= 0:
            raise ValueError("gain must be finite and positive.")
        self.distribution = distribution
        self.gain = gain

    def __call__(self, fan_in, fan_out, rng):
        shape = (fan_out, fan_in)
        if self.distribution == "uniform":
            limit = self.gain * np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=shape)
        std = self.gain * np.sqrt(2.0 / (fan_in + fan_out))
        return rng.standard_normal(shape) * std


def initialize(n_x, h1, h2, n_y, seed=0, weight_init=None):
    """Return float32 parameters for a sigmoid/sigmoid/softmax network."""
    rng = np.random.default_rng(seed)
    weight_init = XavierInitializer() if weight_init is None else weight_init
    parameters = {}
    for i, (fan_in, fan_out) in enumerate(zip((n_x, h1, h2), (h1, h2, n_y)), 1):
        weights = np.asarray(weight_init(fan_in, fan_out, rng), dtype=np.float32)
        parameters["W" + str(i)] = weights
        parameters["b" + str(i)] = np.zeros((fan_out, 1), dtype=np.float32)
    return parameters


def sigmoid(Z):
    """Elementwise sigmoid, stable at extreme finite inputs."""
    # Stable for large positive/negative inputs and preserves float32 arrays.
    exp_negative_abs = np.exp(-np.abs(Z))
    return np.where(Z >= 0, 1 / (1 + exp_negative_abs),
                    exp_negative_abs / (1 + exp_negative_abs))


def softmax(Z):
    """Normalize logits over classes (axis 0) independently per sample."""
    shifted = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(shifted)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


def forward_propagation(X, parameters):
    """Return logits and activations without changing inputs or parameters."""
    Z1 = np.dot(parameters["W1"], X) + parameters["b1"]
    A1 = sigmoid(Z1)
    Z2 = np.dot(parameters["W2"], A1) + parameters["b2"]
    A2 = sigmoid(Z2)
    Z3 = np.dot(parameters["W3"], A2) + parameters["b3"]
    A3 = softmax(Z3)
    return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}


def predict(X, parameters):
    """Return class indices from forward probabilities; labels are not needed."""
    return np.argmax(forward_propagation(X, parameters)["A3"], axis=0)
