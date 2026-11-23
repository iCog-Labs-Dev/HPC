import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from numpy_predictive_coding import initialize_optimizer_state, weight_update


def test_optimizer_state_is_independent_and_zero(problem):
    """Create AdamW bookkeeping for an initialized network.

    Assert the step starts at zero and each moment is a zero array with the
    parameter's shape/dtype, sharing storage with neither parameters nor
    the other moment. These zeros are optimizer memory, not network weights.
    """
    _, _, parameters = problem
    state = initialize_optimizer_state(parameters)
    assert state["step"] == 0
    for name, value in parameters.items():
        for moment in ("m", "v"):
            assert state[moment][name].shape == value.shape
            assert state[moment][name].dtype == value.dtype
            assert_array_equal(state[moment][name], 0)
            assert not np.shares_memory(state[moment][name], value)
        assert not np.shares_memory(state["m"][name], state["v"][name])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_adamw_two_updates_match_hand_calculation(dtype):
    """Apply two distinct gradient sets to all six float32/float64 parameters.

    Assert parameters and moments match an independent AdamW calculation with
    bias correction and decay, dtype is preserved, the same parameter dictionary
    is returned, and the shared step advances once per update.
    """
    parameters = {
        "W1": np.array([[1., -2.]], dtype=dtype), "b1": np.array([[0.5]], dtype=dtype),
        "W2": np.array([[-1.]], dtype=dtype), "b2": np.array([[-0.5]], dtype=dtype),
        "W3": np.array([[0.75]], dtype=dtype), "b3": np.array([[1.5]], dtype=dtype),
    }
    expected = {k: v.astype(np.float64) for k, v in parameters.items()}
    first = {
        "dW1": np.array([[0.2, -0.4]], dtype=dtype), "db1": np.array([[0.3]], dtype=dtype),
        "dW2": np.array([[-0.1]], dtype=dtype), "db2": np.array([[0.5]], dtype=dtype),
        "dW3": np.array([[0.7]], dtype=dtype), "db3": np.array([[-0.6]], dtype=dtype),
    }
    second = {
        "dW1": np.array([[-0.1, 0.6]], dtype=dtype), "db1": np.array([[-0.2]], dtype=dtype),
        "dW2": np.array([[0.4]], dtype=dtype), "db2": np.array([[-0.3]], dtype=dtype),
        "dW3": np.array([[-0.5]], dtype=dtype), "db3": np.array([[0.2]], dtype=dtype),
    }
    state = initialize_optimizer_state(parameters)
    # Representable beta values make this calculation independent of float32
    # bias-correction rounding while still checking correction on both steps.
    beta1, beta2, lr, decay, eps = 0.5, 0.75, 0.01, 0.2, 1e-8
    m = {k: np.zeros_like(v) for k, v in expected.items()}
    v = {k: np.zeros_like(a) for k, a in expected.items()}
    for step, grads in enumerate((first, second), 1):
        for name in expected:
            g = grads["d" + name].astype(np.float64)
            m[name] = beta1 * m[name] + (1 - beta1) * g
            v[name] = beta2 * v[name] + (1 - beta2) * g**2
            corrected_m, corrected_v = m[name] / (1 - beta1**step), v[name] / (1 - beta2**step)
            expected[name] -= lr * (corrected_m / (np.sqrt(corrected_v) + eps) + decay * expected[name])
        returned = weight_update(parameters, grads, state, lr, decay, beta1, beta2, eps)
        assert returned is parameters
        assert state["step"] == step
        for name in parameters:
            assert parameters[name].dtype == dtype
            assert_allclose(parameters[name], expected[name], rtol=1e-6)
            assert_allclose(state["m"][name], m[name], rtol=1e-6)
            assert_allclose(state["v"][name], v[name], rtol=1e-6)


def test_zero_gradient_decays_weights_and_biases(problem):
    """Update nonzero weights and biases with zero gradients and zero moments.

    Assert decoupled weight decay alone changes every weight from 2 to 1.96
    and every bias from 3 to 2.94, confirming biases are decayed too.
    """
    _, _, original = problem
    parameters = {name: np.full_like(value, 2 if name.startswith("W") else 3)
                  for name, value in original.items()}
    state = initialize_optimizer_state(parameters)
    grads = {"d" + name: np.zeros_like(value) for name, value in parameters.items()}
    weight_update(parameters, grads, state,
                  weight_rate=0.1, weight_decay=0.2)
    for name, value in parameters.items():
        assert_allclose(value, 1.96 if name.startswith("W") else 2.94)
