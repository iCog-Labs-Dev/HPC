import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from numpy_predictive_coding import (
    mini_batches, evaluate, forward_propagation, predict, pc_model,
    initialize, inference, parameter_gradients, initialize_optimizer_state, weight_update,
)
from numpy_predictive_coding import training


def test_mini_batches_align_shuffle_and_keep_remainder():
    """Shuffle seven paired samples into batches of three using a fixed seed.

    Assert batch sizes are 3, 3, 1, each sample appears once, labels stay aligned,
    shuffling is reproducible, inputs stay unchanged, and an oversized batch
    includes all samples in one batch.
    """
    X = np.arange(14).reshape(2, 7)
    Y = X + 100
    originals = X.copy(), Y.copy()
    first = list(mini_batches(X, Y, 3, np.random.default_rng(42)))
    second = list(mini_batches(X, Y, 3, np.random.default_rng(42)))
    assert [x.shape[1] for x, _ in first] == [3, 3, 1]
    joined_X = np.concatenate([x for x, _ in first], axis=1)
    joined_Y = np.concatenate([y for _, y in first], axis=1)
    assert_array_equal(joined_Y, joined_X + 100)
    assert_array_equal(np.sort(joined_X[0]), X[0])
    assert not np.array_equal(joined_X, X)
    for (a, b), (c, d) in zip(first, second):
        assert_array_equal(a, c)
        assert_array_equal(b, d)
    assert_array_equal(X, originals[0])
    assert_array_equal(Y, originals[1])
    assert len(list(mini_batches(X, Y, 100, np.random.default_rng(0)))) == 1


@pytest.mark.parametrize("batch_size", [1, 2, 3, 20])
def test_evaluate_sample_weighted_and_forward_only(problem, monkeypatch, batch_size):
    """Evaluate the same samples with singleton, uneven, and oversized batches.

    Assert loss and accuracy equal full-dataset forward calculations, inference
    is never called, and evaluating different labels does not change predictions.
    """
    X, Y, parameters = problem
    probabilities = forward_propagation(X, parameters)["A3"]
    expected_loss = -np.sum(Y * np.log(probabilities)) / X.shape[1]
    expected_accuracy = np.mean(probabilities.argmax(axis=0) == Y.argmax(axis=0))
    def forbidden_inference(*args, **kwargs):
        pytest.fail("Evaluation must use forward prediction only")
    monkeypatch.setattr(training, "inference", forbidden_inference)
    actual = evaluate(X, Y, parameters, batch_size)
    assert_allclose(actual, [expected_loss, expected_accuracy])
    before = predict(X, parameters)
    evaluate(X, 1 - Y, parameters, batch_size)
    assert_array_equal(predict(X, parameters), before)


@pytest.mark.parametrize("X,Y,batch_size", [
    (np.zeros((2, 0)), np.zeros((2, 0)), 2),
    (np.zeros((2, 3)), np.zeros((2, 2)), 2),
    (np.zeros(3), np.zeros((2, 3)), 2),
    (np.zeros((2, 3)), np.zeros((2, 3)), 0),
    (np.zeros((2, 3)), np.zeros((2, 3)), 1.5),
])
def test_invalid_batch_and_evaluation_inputs(X, Y, batch_size):
    """Pass empty/mismatched sample columns, a 1D input, or invalid batch sizes.

    Assert both batch generation and evaluation reject these cases with ValueError.
    """
    with pytest.raises(ValueError):
        list(mini_batches(X, Y, batch_size, np.random.default_rng(0)))
    with pytest.raises(ValueError):
        evaluate(X, Y, initialize(2, 2, 2, 2), batch_size)


def test_zero_epochs_copies_supplied_parameters_and_records_baseline(problem, capsys):
    """Request zero training epochs with supplied parameters and quiet output.

    Assert history contains only epoch-zero evaluation, supplied arrays stay
    unchanged, returned parameters are independent float32 copies, and nothing
    is printed.
    """
    X, Y, parameters = problem
    originals = {k: v.copy() for k, v in parameters.items()}
    trained, history = pc_model(X, Y, initial_parameters=parameters, epochs=0, verbose=False)
    assert len(history) == 1
    assert history[0][0] == 0
    assert_allclose(history[0][1:], evaluate(X, Y, trained))
    for key in parameters:
        assert_array_equal(parameters[key], originals[key])
        assert_array_equal(trained[key], parameters[key].astype(np.float32))
        assert not np.shares_memory(trained[key], parameters[key])
    assert capsys.readouterr().out == ""


def test_training_matches_manual_batches_with_persistent_optimizer(problem):
    """Train two epochs with a partial final batch and repeat the seeded run.

    Assert results match manual batch updates with persistent AdamW state,
    four batches advance the step to four, history includes epochs 0 through 2,
    supplied parameters stay unchanged, and repeated results are identical.
    """
    X, Y, parameters = problem
    original = {k: v.copy() for k, v in parameters.items()}
    kwargs = dict(initial_parameters=parameters, epochs=2, batch_size=2, steps=3,
                  shuffle_seed=9, verbose=False)
    actual, history = pc_model(X, Y, **kwargs)
    expected = {k: v.astype(np.float32).copy() for k, v in parameters.items()}
    state = initialize_optimizer_state(expected)
    rng = np.random.default_rng(9)
    for _ in range(2):
        for xb, yb in mini_batches(X, Y, 2, rng):
            cache = forward_propagation(xb, expected)
            H1, H2, errors = inference(xb, yb, expected, cache, steps=3)
            gradients = parameter_gradients(xb, yb, expected, H1, H2, errors)
            weight_update(expected, gradients, state)
    assert state["step"] == 4
    assert [epoch for epoch, _, _ in history] == [0, 1, 2]
    for name in expected:
        assert_array_equal(actual[name], expected[name])
        assert_array_equal(parameters[name], original[name])
    repeated, repeated_history = pc_model(X, Y, **kwargs)
    assert history == repeated_history
    for name in actual:
        assert_array_equal(actual[name], repeated[name])


def test_small_synthetic_problem_learns():
    """Train a small network on two easily separated synthetic classes.

    Assert final loss is below 70% of initial loss, accuracy reaches at least
    95%, and all recorded metrics remain finite.
    """
    rng = np.random.default_rng(12)
    X = np.concatenate([rng.normal(-1, 0.1, (2, 12)), rng.normal(1, 0.1, (2, 12))], axis=1).astype(np.float32)
    Y = np.eye(2, dtype=np.float32)[[0] * 12 + [1] * 12].T
    _, history = pc_model(X, Y, h1=8, h2=4, epochs=30, batch_size=8, steps=10,
                          weight_rate=0.02, weight_decay=0, verbose=False)
    assert history[-1][1] < history[0][1] * 0.7
    assert history[-1][2] >= 0.95
    assert np.all(np.isfinite(history))


@pytest.mark.parametrize("kwargs", [
    {"epochs": -1}, {"epochs": 1.5}, {"steps": 0}, {"batch_size": 0},
    {"h1": 0}, {"activity_rate": 0}, {"weight_rate": np.nan},
    {"weight_decay": -1}, {"weight_decay": np.inf},
])
def test_invalid_training_settings(problem, kwargs):
    """Supply invalid counts, hidden width, rates, or weight decay one at a time.

    Assert the training entry point rejects each setting with ValueError.
    """
    X, Y, _ = problem
    with pytest.raises(ValueError):
        pc_model(X, Y, verbose=False, **kwargs)


def test_training_rejects_nonfinite_data_and_parameters(problem):
    """Insert NaN into training inputs and infinity into supplied initial weights.

    Assert nonfinite data raises ValueError and nonfinite initial parameters
    raise FloatingPointError, with messages identifying the problem.
    """
    X, Y, parameters = problem
    invalid_X = X.copy()
    invalid_X[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        pc_model(invalid_X, Y)
    parameters["W1"][0, 0] = np.inf
    with pytest.raises(FloatingPointError, match="initial"):
        pc_model(X, Y, initial_parameters=parameters)


def test_training_detects_nonfinite_updates(problem, monkeypatch):
    """Replace the weight update with one that writes NaN into a parameter.

    Assert training stops with FloatingPointError reporting nonfinite parameters.
    """
    X, Y, parameters = problem
    def unstable_update(parameters, *args):
        parameters["W1"][0, 0] = np.nan
        return parameters
    monkeypatch.setattr(training, "weight_update", unstable_update)
    with pytest.raises(FloatingPointError, match="Nonfinite parameters"):
        pc_model(X, Y, initial_parameters=parameters, epochs=1, verbose=False)


def test_training_detects_nonfinite_loss(problem, monkeypatch):
    """Replace evaluation with a result containing NaN loss at epoch zero.

    Assert training raises FloatingPointError identifying the nonfinite forward
    loss rather than recording it as a valid history entry.
    """
    X, Y, parameters = problem
    monkeypatch.setattr(training, "evaluate", lambda *args, **kwargs: (np.nan, 0))
    with pytest.raises(FloatingPointError, match="forward loss"):
        pc_model(X, Y, initial_parameters=parameters, epochs=0, verbose=False)


def test_training_progress_output(problem, capsys):
    """Run the default verbose mode for zero training epochs.

    Assert stdout includes the epoch-zero forward cross-entropy progress line.
    """
    X, Y, parameters = problem
    pc_model(X, Y, initial_parameters=parameters, epochs=0)
    assert "Epoch  0: forward CE" in capsys.readouterr().out
