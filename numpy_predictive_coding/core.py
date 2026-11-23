"""Hidden-state inference and local predictive-coding learning signals."""

import numpy as np

from .network import sigmoid, softmax


def output_logit_gradient(targets, probabilities):
    """Softmax cross-entropy logit derivative for one-hot target columns."""
    # Standard softmax cross-entropy gradient for one-hot training labels.
    return probabilities - targets


def compute_errors(X, Y, parameters, H1, H2):
    """Compute activity-minus-prediction hidden errors and output signal p - Y."""
    mu1 = sigmoid(np.dot(parameters["W1"], X) + parameters["b1"])
    mu2 = sigmoid(np.dot(parameters["W2"], H1) + parameters["b2"])
    Z3 = np.dot(parameters["W3"], H2) + parameters["b3"]
    p = softmax(Z3)
    return {"mu1": mu1, "mu2": mu2, "Z3": Z3, "p": p,
            "e1": H1 - mu1, "e2": H2 - mu2,
            "dZ3": output_logit_gradient(Y, p)}


def compute_cost(Y, cache):
    """Mean cross-entropy with probabilities clipped to [1e-7, 1]."""
    # Match FabricPC CrossEntropyEnergy(eps=1e-7), including clipping.
    if "A3" in cache:
        probabilities = cache["A3"]
    elif "p" in cache:
        probabilities = cache["p"]
    else:
        probabilities = softmax(cache["Z3"])
    return np.mean(-np.sum(Y * np.log(np.clip(probabilities, 1e-7, 1.0)), axis=0))


def compute_energy(Y, errors):
    """Mean squared hidden error / 2 plus clipped output cross-entropy."""
    hidden_loss = 0.5 * (np.sum(errors["e1"]**2) + np.sum(errors["e2"]**2))
    return hidden_loss / Y.shape[1] + compute_cost(Y, errors)


def inference(X, Y, parameters, cache, steps=20, activity_rate=0.05):
    """Relax copied forward activities with simultaneous, unnormalized updates.

    Weights stay fixed. Zero steps returns independent copies of the initial
    activities. Errors are always recomputed at the returned final state.
    """
    H1, H2 = cache["A1"].copy(), cache["A2"].copy()

    for _ in range(steps):
        errors = compute_errors(X, Y, parameters, H1, H2)

        dH2 = errors["e2"] + np.dot(parameters["W3"].T, errors["dZ3"])
        dH1 = errors["e1"] - np.dot(parameters["W2"].T, errors["e2"] * (errors["mu2"] * (1 - errors["mu2"])))

        H1 = H1 - activity_rate * dH1
        H2 = H2 - activity_rate * dH2

    return H1, H2, compute_errors(X, Y, parameters, H1, H2)


def parameter_gradients(X, Y, parameters, H1, H2, errors):
    """Differentiate mean energy with inferred activities held fixed.

    Pass errors recomputed at H1/H2 with the same parameters. The output
    derivative intentionally ignores clipping in the reported loss metric.
    """
    m = X.shape[1]
    dZ1 = -errors["e1"] * (errors["mu1"] * (1 - errors["mu1"]))
    dZ2 = -errors["e2"] * (errors["mu2"] * (1 - errors["mu2"]))
    dZ3 = errors["dZ3"]
    return {"dW1": (np.dot(dZ1, X.T)) / m, "db1": np.sum(dZ1, axis=1, keepdims=True) / m,
            "dW2": (np.dot(dZ2, H1.T)) / m, "db2": np.sum(dZ2, axis=1, keepdims=True) / m,
            "dW3": (np.dot(dZ3, H2.T)) / m, "db3": np.sum(dZ3, axis=1, keepdims=True) / m}
