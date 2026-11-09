import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from numpy_predictive_coding import (
    forward_propagation, output_logit_gradient, compute_errors, compute_cost,
    compute_energy, inference, parameter_gradients,
)


def central_difference(array, objective, epsilon=1e-5):
    """Differentiate a scalar objective while restoring each perturbed entry."""
    result = np.empty_like(array)
    for index in np.ndindex(array.shape):
        original = array[index]
        try:
            array[index] = original + epsilon
            plus = objective()
            array[index] = original - epsilon
            minus = objective()
            result[index] = (plus - minus) / (2 * epsilon)
        finally:
            array[index] = original
    return result


def test_forward_initialization_has_zero_hidden_errors(problem):
    """Set hidden activities to the activations from the forward pass.

    Assert both hidden errors are exactly zero, output probabilities match the
    forward pass, p - Y is nonzero, and energy equals classification cost.
    """
    X, Y, parameters = problem
    cache = forward_propagation(X, parameters)
    errors = compute_errors(X, Y, parameters, cache["A1"], cache["A2"])
    assert_array_equal(errors["e1"], np.zeros_like(cache["A1"]))
    assert_array_equal(errors["e2"], np.zeros_like(cache["A2"]))
    assert_array_equal(errors["p"], cache["A3"])
    assert_array_equal(errors["dZ3"], cache["A3"] - Y)
    assert np.linalg.norm(errors["dZ3"]) > 0
    assert_allclose(compute_energy(Y, errors), compute_cost(Y, cache))


def test_errors_use_current_hidden_states(problem):
    """Perturb both hidden activities away from their forward values.

    Assert e1 reflects its perturbation, mu2 is recomputed from current H1,
    e2 equals H2 - mu2, and output logits use current H2.
    """
    X, Y, parameters = problem
    cache = forward_propagation(X, parameters)
    H1, H2 = cache["A1"] + 0.2, cache["A2"] - 0.1
    errors = compute_errors(X, Y, parameters, H1, H2)
    mu2 = 1 / (1 + np.exp(-(parameters["W2"] @ H1 + parameters["b2"])))
    assert_allclose(errors["e1"], 0.2)
    assert_allclose(errors["mu2"], mu2)
    assert_allclose(errors["e2"], H2 - mu2)
    assert not np.allclose(mu2, cache["A2"])
    assert_allclose(errors["Z3"], parameters["W3"] @ H2 + parameters["b3"])


def test_output_gradient_matches_unclipped_cross_entropy_derivative(problem):
    """Use moderate logits whose probabilities avoid the clipping boundary.

    Assert the mean-loss derivative (p - Y) / batch_size agrees with central
    finite differences of ordinary, unclipped cross-entropy.
    """
    _, Y, _ = problem
    logits = np.array([[0.3, -0.4, 0.5], [-0.2, 0.6, -0.1]])
    def probabilities():
        exps = np.exp(logits)
        return exps / exps.sum(axis=0)
    analytic = output_logit_gradient(Y, probabilities()) / Y.shape[1]
    numeric = central_difference(logits, lambda: -np.sum(Y * np.log(probabilities())) / Y.shape[1])
    assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("cache_key", ["A3", "p", "Z3"])
def test_cost_supported_cache_forms(cache_key):
    """Supply the same predictions as A3, p, or logarithmic logits Z3.

    Assert each supported cache form gives the hand-calculated mean
    cross-entropy for the two target classes.
    """
    probabilities = np.array([[0.8, 0.25], [0.2, 0.75]])
    Y = np.eye(2)
    values = np.log(probabilities) if cache_key == "Z3" else probabilities
    assert_allclose(compute_cost(Y, {cache_key: values}), -(np.log(0.8) + np.log(0.75)) / 2)


def test_cost_clipping_and_energy():
    """Include a zero target probability and known nonzero hidden errors.

    Assert cross-entropy clips zero to 1e-7, and total energy adds half the
    squared hidden errors divided by the number of samples.
    """
    Y = np.eye(2)
    errors = {"p": np.array([[0., 0.], [1., 1.]]),
              "e1": np.array([[1., 2.]]), "e2": np.array([[3., 4.]])}
    expected_ce = -np.log(1e-7) / 2
    assert_allclose(compute_cost(Y, errors), expected_ce)
    assert_allclose(compute_energy(Y, errors), expected_ce + 0.5 * 30 / 2)


def test_inference_zero_steps_copies_forward_states(problem):
    """Run inference with zero relaxation steps from a forward cache.

    Assert returned activities equal the initial activations without sharing
    their storage, and both returned hidden errors remain exactly zero.
    """
    X, Y, parameters = problem
    cache = forward_propagation(X, parameters)
    H1, H2, errors = inference(X, Y, parameters, cache, steps=0)
    for activity, name in ((H1, "A1"), (H2, "A2")):
        assert_array_equal(activity, cache[name])
        assert not np.shares_memory(activity, cache[name])
    assert_array_equal(errors["e1"], 0)
    assert_array_equal(errors["e2"], 0)


def test_inference_first_step_only_moves_second_hidden_layer(problem):
    """Take one simultaneous inference step from zero hidden errors.

    Assert H1 stays exactly fixed while H2 changes by the learning rate times
    W3.T @ (p - Y), using the initial forward output signal.
    """
    X, Y, parameters = problem
    cache = forward_propagation(X, parameters)
    H1, H2, _ = inference(X, Y, parameters, cache, steps=1, activity_rate=0.05)
    assert_array_equal(H1, cache["A1"])
    expected_H2 = cache["A2"] - 0.05 * (parameters["W3"].T @ (cache["A3"] - Y))
    assert_allclose(H2, expected_H2)
    assert not np.array_equal(H2, cache["A2"])


def test_inference_simultaneous_updates_and_final_errors(problem):
    """Perturb both hidden layers and manually compute three relaxation steps.

    Assert inference matches updates computed from each step's old states,
    returns errors for the final states, lowers energy in this case, and
    leaves its input cache and parameters unchanged.
    """
    X, Y, parameters = problem
    cache = forward_propagation(X, parameters)
    # Perturb both activities to make both gradients nonzero on the first step.
    cache["A1"] += 0.1
    cache["A2"] -= 0.2
    initial_cache = {k: v.copy() for k, v in cache.items()}
    initial_parameters = {k: v.copy() for k, v in parameters.items()}
    H1, H2 = cache["A1"].copy(), cache["A2"].copy()
    initial_energy = compute_energy(Y, compute_errors(X, Y, parameters, H1, H2))
    for _ in range(3):
        mu1 = 1 / (1 + np.exp(-(parameters["W1"] @ X + parameters["b1"])))
        mu2 = 1 / (1 + np.exp(-(parameters["W2"] @ H1 + parameters["b2"])))
        exps = np.exp(parameters["W3"] @ H2 + parameters["b3"])
        p = exps / exps.sum(axis=0)
        g1 = H1 - mu1 - parameters["W2"].T @ ((H2 - mu2) * mu2 * (1 - mu2))
        g2 = H2 - mu2 + parameters["W3"].T @ (p - Y)
        H1, H2 = H1 - 0.01 * g1, H2 - 0.01 * g2
    actual1, actual2, errors = inference(X, Y, parameters, cache, steps=3, activity_rate=0.01)
    assert_allclose(actual1, H1)
    assert_allclose(actual2, H2)
    for key, expected in compute_errors(X, Y, parameters, actual1, actual2).items():
        assert_allclose(errors[key], expected)
    assert compute_energy(Y, errors) < initial_energy
    for key in parameters:
        assert_array_equal(parameters[key], initial_parameters[key])
    for key in cache:
        assert_array_equal(cache[key], initial_cache[key])


def test_inference_does_not_clip_activities(problem):
    """Start H1 below zero and H2 above one, then take a small inference step.

    Assert activities remain outside [0, 1], so inference has not clipped them
    or applied an additional sigmoid.
    """
    X, Y, parameters = problem
    cache = forward_propagation(X, parameters)
    cache["A1"][:] = -0.5
    cache["A2"][:] = 1.5
    H1, H2, _ = inference(X, Y, parameters, cache, steps=1, activity_rate=0.001)
    assert np.all(H1 < 0)
    assert np.all(H2 > 1)


def test_hidden_state_updates_match_energy_derivatives(problem):
    """Compare one inference step with finite differences at perturbed states.

    Assert both hidden-state update directions match energy derivatives
    multiplied by batch size, since inference uses per-example derivatives.
    """
    X, Y, parameters = problem
    cache = forward_propagation(X, parameters)
    cache["A1"] += 0.12
    cache["A2"] -= 0.17
    H1, H2 = cache["A1"], cache["A2"]
    objective = lambda: compute_energy(Y, compute_errors(X, Y, parameters, H1, H2))
    # Inference uses per-example derivatives, without the mean's 1 / batch_size.
    expected1 = central_difference(H1, objective) * X.shape[1]
    expected2 = central_difference(H2, objective) * X.shape[1]
    actual1, actual2, _ = inference(X, Y, parameters, cache, steps=1, activity_rate=0.01)
    assert_allclose((H1 - actual1) / 0.01, expected1, rtol=1e-5, atol=1e-8)
    assert_allclose((H2 - actual2) / 0.01, expected2, rtol=1e-5, atol=1e-8)


def test_parameter_gradients_match_energy_derivatives(problem):
    """Hold perturbed hidden states fixed while varying each weight and bias.

    Assert every local gradient has its parameter's shape and matches central
    finite differences of mean energy away from probability clipping.
    """
    X, Y, parameters = problem
    cache = forward_propagation(X, parameters)
    H1, H2 = cache["A1"] + 0.12, cache["A2"] - 0.17
    objective = lambda: compute_energy(Y, compute_errors(X, Y, parameters, H1, H2))
    gradients = parameter_gradients(X, Y, parameters, H1, H2, compute_errors(X, Y, parameters, H1, H2))
    for name, value in parameters.items():
        numeric = central_difference(value, objective)
        assert gradients["d" + name].shape == value.shape
        assert_allclose(gradients["d" + name], numeric, rtol=1e-5, atol=1e-8)


def test_gradient_batch_averaging_and_zero_initial_hidden_gradients(problem):
    """Compute gradients at forward-initialized states for a batch and its samples.

    Assert hidden-layer weight/bias gradients start at zero, and every batch
    gradient equals the average of separately calculated single-sample gradients.
    """
    X, Y, parameters = problem
    cache = forward_propagation(X, parameters)
    H1, H2 = cache["A1"], cache["A2"]
    errors = compute_errors(X, Y, parameters, H1, H2)
    gradients = parameter_gradients(X, Y, parameters, H1, H2, errors)
    for name in ("dW1", "db1", "dW2", "db2"):
        assert_array_equal(gradients[name], 0)
    # Mean of per-sample gradients equals the full-batch result; covers size 1 too.
    per_sample = []
    for j in range(X.shape[1]):
        x, y, a, b = X[:, j:j+1], Y[:, j:j+1], H1[:, j:j+1], H2[:, j:j+1]
        per_sample.append(parameter_gradients(x, y, parameters, a, b, compute_errors(x, y, parameters, a, b)))
    for name in gradients:
        assert_allclose(gradients[name], np.mean([g[name] for g in per_sample], axis=0), atol=1e-15)
