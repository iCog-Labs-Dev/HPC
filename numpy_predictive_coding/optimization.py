"""AdamW updates with persistent moment state."""

import numpy as np


def initialize_optimizer_state(parameters):
    """Create independent zero moment arrays and a batch-update counter."""
    def zero_moments():
        return {
            "W1": np.zeros_like(parameters["W1"]),
            "b1": np.zeros_like(parameters["b1"]),
            "W2": np.zeros_like(parameters["W2"]),
            "b2": np.zeros_like(parameters["b2"]),
            "W3": np.zeros_like(parameters["W3"]),
            "b3": np.zeros_like(parameters["b3"]),
        }

    return {"step": 0, "m": zero_moments(), "v": zero_moments()}


def weight_update(parameters, grads, optimizer_state, weight_rate=0.001,
                  weight_decay=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """Update parameters and optimizer state in place; return parameters.

    Update W1, b1, W2, b2, W3, b3 explicitly, with one shared batch step.
    Apply decoupled weight decay to every parameter, including biases.
    """
    # AdamW, matching optax.adamw defaults. Decay applies to biases too
    # because the FabricPC script supplies no optimizer mask.
    optimizer_state["step"] += 1
    step = optimizer_state["step"]

    def update_parameter(name):
        value = parameters[name]
        gradient = grads["d" + name]
        moment = beta1 * optimizer_state["m"][name] + (1 - beta1) * gradient
        variance = beta2 * optimizer_state["v"][name] + (1 - beta2) * gradient**2
        optimizer_state["m"][name] = moment
        optimizer_state["v"][name] = variance
        # Optax calculates bias corrections in the moment's dtype.
        correction1 = 1 - np.power(np.asarray(beta1, dtype=moment.dtype), step, dtype=moment.dtype)
        correction2 = 1 - np.power(np.asarray(beta2, dtype=variance.dtype), step, dtype=variance.dtype)
        corrected_m = moment / correction1
        corrected_v = variance / correction2
        update = corrected_m / (np.sqrt(corrected_v) + epsilon) + weight_decay * value
        parameters[name] = value - weight_rate * update

    update_parameter("W1")
    update_parameter("b1")
    update_parameter("W2")
    update_parameter("b2")
    update_parameter("W3")
    update_parameter("b3")
    return parameters
