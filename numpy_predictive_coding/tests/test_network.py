import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from numpy_predictive_coding import (
    XavierInitializer, initialize, sigmoid, softmax, forward_propagation, predict,
)


@pytest.mark.parametrize("distribution", ["normal", "uniform"])
def test_xavier_distribution_and_gain(distribution):
    """Sample Xavier normal and uniform weights with a fixed seed and gain.

    Assert output shape, near-zero mean, theoretical variance, linear gain
    scaling, and the uniform distribution's bounds when applicable.
    """
    # Large fixed sample: verify the advertised moments rather than exact RNG draws.
    gain, fan_in, fan_out = 1.7, 500, 400
    weights = XavierInitializer(distribution, gain)(fan_in, fan_out, np.random.default_rng(3))
    variance = 2 * gain**2 / (fan_in + fan_out)
    assert weights.shape == (fan_out, fan_in)
    assert abs(weights.mean()) < np.sqrt(variance) * 0.01
    assert_allclose(weights.var(), variance, rtol=0.015)
    base = XavierInitializer(distribution)(fan_in, fan_out, np.random.default_rng(3))
    assert_allclose(weights, gain * base)
    if distribution == "uniform":
        assert np.max(np.abs(weights)) <= gain * np.sqrt(6 / (fan_in + fan_out))


@pytest.mark.parametrize("kwargs", [
    {"distribution": "other"}, {"gain": 0}, {"gain": -1},
    {"gain": np.inf}, {"gain": np.nan},
])
def test_invalid_initializer(kwargs):
    """Pass unsupported distributions or nonpositive/nonfinite gains.

    Assert each invalid configuration raises ValueError.
    """
    with pytest.raises(ValueError):
        XavierInitializer(**kwargs)


def test_initialize_shapes_biases_dtype_and_seed():
    """Initialize the same network twice with the same seed, then a new seed.

    Assert layer shapes, zero biases, float32 arrays, identical parameters for
    repeated seeds, and different first-layer weights for a different seed.
    """
    first, second = initialize(3, 4, 2, 2, seed=8), initialize(3, 4, 2, 2, seed=8)
    for i, shape in enumerate(((4, 3), (2, 4), (2, 2)), 1):
        assert first[f"W{i}"].shape == shape
        assert first[f"b{i}"].shape == (shape[0], 1)
        assert_array_equal(first[f"b{i}"], 0)
    for key in first:
        assert first[key].dtype == np.float32
        assert_array_equal(first[key], second[key])
    assert not np.array_equal(first["W1"], initialize(3, 4, 2, 2, seed=9)["W1"])


def test_initialize_custom_initializer():
    """Supply an initializer that always returns weights of 0.25.

    Assert the first weight matrix uses those values instead of Xavier draws.
    """
    parameters = initialize(3, 4, 2, 2, weight_init=lambda a, b, rng: np.full((b, a), 0.25))
    assert_array_equal(parameters["W1"], 0.25)


def test_sigmoid_extremes_and_dtype():
    """Evaluate float32 sigmoid inputs ranging from -1000 to 1000.

    Assert known probabilities, monotonicity, preserved dtype, and no overflow,
    division-by-zero, or invalid-operation errors.
    """
    values = np.array([-1000, -1, 0, 1, 1000], dtype=np.float32)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = sigmoid(values)
    assert actual.dtype == np.float32
    assert_allclose(actual, [0, 1 / (1 + np.e), 0.5, np.e / (1 + np.e), 1], atol=1e-7)
    assert np.all(np.diff(actual) >= 0)


def test_softmax_normalizes_columns_and_is_shift_invariant():
    """Evaluate extreme logits and add a different constant to each column.

    Assert nonnegative float32 probabilities sum to one per sample, stay
    unchanged by those shifts, and produce no numerical errors.
    """
    logits = np.array([[1000, -1000], [1001, -1002], [999, -999]], dtype=np.float32)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        probabilities = softmax(logits)
    assert probabilities.dtype == np.float32
    assert_allclose(probabilities.sum(axis=0), 1, atol=1e-7)
    assert np.all(probabilities >= 0)
    assert_allclose(probabilities, softmax(logits + np.array([[200, -300]])), atol=1e-7)


def test_forward_matches_independent_calculation_and_predict(problem):
    """Calculate each layer independently for a small deterministic network.

    Assert cached logits and activations match, predictions select the largest
    output probability, and neither inputs nor parameters are modified.
    """
    X, _, parameters = problem
    originals = {k: v.copy() for k, v in parameters.items()}
    X_before = X.copy()
    cache = forward_propagation(X, parameters)
    activity = X
    for i in (1, 2, 3):
        logits = parameters[f"W{i}"] @ activity + parameters[f"b{i}"]
        if i < 3:
            activity = 1 / (1 + np.exp(-logits))
        else:
            exponentials = np.exp(logits)
            activity = exponentials / exponentials.sum(axis=0)
        assert_allclose(cache[f"Z{i}"], logits)
        assert_allclose(cache[f"A{i}"], activity)
    assert_array_equal(predict(X, parameters), np.argmax(activity, axis=0))
    assert_array_equal(X, X_before)
    for key in parameters:
        assert_array_equal(parameters[key], originals[key])
