"""Dataset-independent mini-batch training and forward evaluation."""

import time

import numpy as np

from .network import initialize, forward_propagation
from .core import compute_cost, inference, parameter_gradients
from .optimization import initialize_optimizer_state, weight_update


def mini_batches(X, Y, batch_size, rng):
    """Yield aligned shuffled columns, including the final partial batch."""
    if (X.ndim != 2 or Y.ndim != 2 or X.shape[1] == 0 or X.shape[1] != Y.shape[1]
            or not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
        raise ValueError("Use a positive integer batch size and nonempty matching X/Y columns.")
    order = rng.permutation(X.shape[1])
    for start in range(0, X.shape[1], batch_size):
        indices = order[start:start + batch_size]
        yield X[:, indices], Y[:, indices]


def evaluate(X, Y, parameters, batch_size=200):
    """Return sample-weighted forward (cross-entropy, accuracy)."""
    if (X.ndim != 2 or Y.ndim != 2 or X.shape[1] == 0 or X.shape[1] != Y.shape[1]
            or not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
        raise ValueError("Use a positive integer batch size and nonempty matching X/Y columns.")
    loss, correct = 0.0, 0
    for start in range(0, X.shape[1], batch_size):
        xb, yb = X[:, start:start + batch_size], Y[:, start:start + batch_size]
        cache = forward_propagation(xb, parameters)
        loss += float(compute_cost(yb, cache)) * xb.shape[1]
        correct += int(np.sum(np.argmax(cache["A3"], axis=0) == np.argmax(yb, axis=0)))
    return loss / X.shape[1], correct / X.shape[1]


def pc_model(X, Y, h1=256, h2=64, epochs=20, batch_size=200,
             steps=20, activity_rate=0.05, weight_rate=0.001, seed=0,
             weight_init=None, weight_decay=0.1, shuffle_seed=42,
             initial_parameters=None, verbose=True):
    """Train on prepared X/Y columns and return (parameters, history).

    History contains (epoch, forward_cross_entropy, forward_accuracy), starting
    at epoch zero. Supplied initial parameters are copied; their shapes define
    the architecture instead of h1/h2. Test data never enters this function.
    """
    if (not isinstance(epochs, (int, np.integer)) or epochs < 0
            or not isinstance(steps, (int, np.integer)) or steps < 1
            or not isinstance(batch_size, (int, np.integer)) or batch_size < 1
            or not np.isfinite(activity_rate) or activity_rate <= 0
            or not np.isfinite(weight_rate) or weight_rate <= 0
            or not np.isfinite(weight_decay) or weight_decay < 0
            or X.ndim != 2 or Y.ndim != 2 or 0 in X.shape or 0 in Y.shape
            or X.shape[1] != Y.shape[1]):
        raise ValueError("Use matching nonempty X/Y, valid epochs, steps, batch size, rates, and decay.")
    if initial_parameters is None and (h1 < 1 or h2 < 1):
        raise ValueError("Hidden layer widths must be positive.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        raise ValueError("Training X and Y must contain finite values.")
    n_x, n_y = X.shape[0], Y.shape[0]
    if initial_parameters is None:
        parameters = initialize(n_x, h1, h2, n_y, seed, weight_init=weight_init)
    else:
        parameters = {name: value.astype(np.float32).copy() for name, value in initial_parameters.items()}
    if not all(np.all(np.isfinite(v)) for v in parameters.values()):
        raise FloatingPointError("Nonfinite initial parameters.")
    optimizer_state = initialize_optimizer_state(parameters)
    rng = np.random.default_rng(shuffle_seed)
    history = []
    start_time = time.perf_counter()
    for epoch in range(epochs + 1):
        if epoch > 0:
            for xb, yb in mini_batches(X, Y, batch_size, rng):
                cache = forward_propagation(xb, parameters)
                H1, H2, errors = inference(xb, yb, parameters, cache, steps, activity_rate)
                grads = parameter_gradients(xb, yb, parameters, H1, H2, errors)
                parameters = weight_update(parameters, grads, optimizer_state, weight_rate, weight_decay)
                if not all(np.all(np.isfinite(v)) for v in parameters.values()):
                    raise FloatingPointError("Nonfinite parameters; reduce activity_rate or weight_rate.")
        # Forward metrics avoid conflating relaxed training energy with prediction quality.
        loss, accuracy = evaluate(X, Y, parameters, batch_size=batch_size)
        if not np.isfinite(loss):
            raise FloatingPointError("Nonfinite forward loss; reduce activity_rate or weight_rate.")
        history.append((epoch, loss, accuracy))
        if verbose:
            print(f"Epoch {epoch:2d}: forward CE = {loss:.5f}, accuracy = {accuracy:.2%}, elapsed = {time.perf_counter() - start_time:.1f}s", flush=True)
    return parameters, history
