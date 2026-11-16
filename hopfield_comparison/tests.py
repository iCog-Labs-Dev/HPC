from unittest.mock import patch

import numpy as np
import pytest

import compare
from hopfield import Hopfield
from modern_hopfield import ModernHopfield, softmax
from storkey_hopfield import StorkeyHopfield


def test_initial_weights_are_zero():
    network = Hopfield(4)

    np.testing.assert_array_equal(network.W, np.zeros((4, 4)))


def test_training_uses_normalized_hebbian_weights():
    network = Hopfield(4)
    network.train(np.array([[1, 1, -1, -1], [1, -1, 1, -1]]))

    expected = np.array([
        [0, 0, 0, -0.5],
        [0, 0, -0.5, 0],
        [0, -0.5, 0, 0],
        [-0.5, 0, 0, 0],
    ])
    np.testing.assert_allclose(network.W, expected)


def test_retraining_replaces_previous_memories():
    network = Hopfield(4)
    network.train(np.array([[1, 1, 1, 1]]))
    replacement = np.array([[1, -1, 1, -1]])
    network.train(replacement)

    fresh = Hopfield(4)
    fresh.train(replacement)
    np.testing.assert_array_equal(network.W, fresh.W)

    network.train([])
    np.testing.assert_array_equal(network.W, np.zeros((4, 4)))


def test_energy_matches_known_states():
    network = Hopfield(4)
    pattern = np.array([1, 1, -1, -1])
    network.train([pattern])

    assert network.energy(pattern) == -1.5
    assert network.energy(np.array([-1, 1, -1, -1])) == 0


def test_recall_recovers_corrupted_memory_without_mutating_input():
    network = Hopfield(4)
    pattern = np.array([1, 1, -1, -1])
    network.train([pattern])
    noisy = np.array([-1, 1, -1, -1])
    original = noisy.copy()

    with patch("numpy.random.permutation", return_value=np.arange(4)):
        recalled = network.recall(noisy, sweeps=1)

    np.testing.assert_array_equal(recalled, pattern)
    np.testing.assert_array_equal(noisy, original)


def test_recall_resolves_zero_input_to_positive_one():
    network = Hopfield(4)

    np.testing.assert_array_equal(
        network.recall(-np.ones(4, dtype=int), sweeps=1),
        np.ones(4, dtype=int),
    )


def test_zero_sweeps_returns_unchanged_copy():
    network = Hopfield(4)
    query = np.array([1, -1, 1, -1])

    recalled = network.recall(query, sweeps=0)

    np.testing.assert_array_equal(recalled, query)
    assert not np.shares_memory(recalled, query)


def test_storkey_weights_match_connection_by_connection_reference():
    patterns = np.array([
        [1, 1, -1, -1, 1, -1],
        [1, -1, 1, -1, -1, 1],
        [-1, 1, 1, -1, 1, 1],
    ])
    n = patterns.shape[1]
    expected = np.zeros((n, n))
    for pattern in patterns:
        previous = expected.copy()
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                h_ij = sum(previous[i, k] * pattern[k]
                           for k in range(n) if k not in (i, j))
                h_ji = sum(previous[j, k] * pattern[k]
                           for k in range(n) if k not in (i, j))
                expected[i, j] += (
                    pattern[i] * pattern[j]
                    - h_ij * pattern[j] - pattern[i] * h_ji
                ) / n

    network = StorkeyHopfield(n)
    network.train(patterns)

    np.testing.assert_allclose(network.W, expected, atol=1e-14)
    np.testing.assert_allclose(network.W, network.W.T, atol=1e-14)
    np.testing.assert_array_equal(np.diag(network.W), np.zeros(n))


def test_storkey_retraining_replaces_previous_memories():
    network = StorkeyHopfield(4)
    network.train([[1, 1, 1, 1]])
    replacement = [[1, -1, 1, -1]]
    network.train(replacement)
    fresh = StorkeyHopfield(4)
    fresh.train(replacement)

    np.testing.assert_array_equal(network.W, fresh.W)
    network.train(np.empty((0, 4)))
    np.testing.assert_array_equal(network.W, np.zeros((4, 4)))


@pytest.mark.parametrize("patterns", [
    [1, -1, 1, -1],
    [[1, -1, 1]],
    [[1, 0, -1, 1]],
])
def test_storkey_rejects_invalid_patterns(patterns):
    with pytest.raises(ValueError):
        StorkeyHopfield(4).train(patterns)


def test_storkey_inherited_recall_recovers_memory_and_lowers_energy():
    network = StorkeyHopfield(4)
    pattern = np.array([1, 1, -1, -1])
    network.train([pattern])
    noisy = np.array([-1, 1, -1, -1])

    with patch("numpy.random.permutation", return_value=np.arange(4)):
        recalled = network.recall(noisy, sweeps=1)

    np.testing.assert_array_equal(recalled, pattern)
    assert network.energy(recalled) < network.energy(noisy)


def test_softmax_is_normalized_and_stable_for_large_scores():
    scores = np.array([1000.0, 1001.0, 1002.0])
    expected = np.array([1.0, np.e, np.e ** 2])
    expected /= expected.sum()

    np.testing.assert_allclose(softmax(scores), expected)
    np.testing.assert_allclose(softmax(scores - 2000), expected)
    np.testing.assert_array_equal(scores, [1000, 1001, 1002])


@pytest.mark.parametrize("beta", [0, -1])
def test_modern_rejects_nonpositive_beta(beta):
    with pytest.raises(ValueError, match="beta must be positive"):
        ModernHopfield(beta=beta)


def test_modern_store_copies_and_replaces_memories():
    network = ModernHopfield()
    patterns = np.array([[1, -1], [-1, 1]])
    network.store(patterns)
    patterns[0, 0] = -1

    np.testing.assert_array_equal(network.P, [[1, -1], [-1, 1]])
    assert np.issubdtype(network.P.dtype, np.floating)

    network.store([[1, 1]])
    np.testing.assert_array_equal(network.P, [[1, 1]])


def test_modern_recall_matches_analytical_two_memory_updates():
    network = ModernHopfield(beta=0.5)
    network.store([[1, -1], [-1, 1]])
    query = np.array([1, -1])

    # Opposite memories yield [tanh(2 * beta * a), -tanh(2 * beta * a)].
    for steps in (1, 2, 3):
        amplitude = 1.0
        for _ in range(steps):
            amplitude = np.tanh(amplitude)
        recalled = network.recall(query, steps=steps)
        np.testing.assert_allclose(recalled, [amplitude, -amplitude])

    np.testing.assert_array_equal(query, [1, -1])
    unchanged = network.recall(query, steps=0)
    np.testing.assert_array_equal(unchanged, query)
    assert not np.shares_memory(unchanged, query)


def test_modern_recovers_binary_memory_from_noisy_query():
    network = ModernHopfield(beta=4)
    pattern = np.array([1, 1, -1, -1])
    network.store([pattern, -pattern])

    recalled = network.recall(np.array([-1, 1, -1, -1]))

    np.testing.assert_array_equal(np.sign(recalled), pattern)


def test_modern_energy_matches_known_value_and_handles_large_scores():
    network = ModernHopfield(beta=2)
    network.store([[1, -1], [-1, 1]])

    assert network.energy(np.zeros(2)) == pytest.approx(-np.log(2) / 2)
    query = np.array([1000.0, -1000.0])
    assert network.energy(query) == pytest.approx(998000.0)
    assert network.energy(network.recall(query)) <= network.energy(query)


def test_comparison_runner_shares_memories_queries_and_update_seeds(capsys):
    # Exercise the full table with one trial per count and lightweight models.
    # Numerical behavior of the actual models is covered above.
    with (
        patch.object(compare, "range", return_value=range(1), create=True),
        patch.object(compare, "Hopfield", autospec=True) as classical,
        patch.object(compare, "StorkeyHopfield", autospec=True) as storkey,
        patch.object(compare, "ModernHopfield", autospec=True) as modern,
        patch.object(compare.np.random, "seed") as seed,
    ):
        for model in (classical, storkey, modern):
            model.return_value.recall.side_effect = lambda query: query.copy()
        compare.main()

    lines = capsys.readouterr().out.splitlines()
    assert lines[1] == "patterns | classical | storkey | modern"
    counts = [int(line.split("|")[0]) for line in lines[2:]]
    assert counts == [5, 10, 14, 20, 50, 100, 200, 500, 1000, 10000]
    for line in lines[2:]:
        # Returning the still-corrupted query must never count as success.
        assert [float(value) for value in line.split("|")[1:]] == [0, 0, 0]
    for index, count in enumerate(counts):
        memories = classical.return_value.train.call_args_list[index].args[0]
        assert memories.shape == (count, 100)
        assert storkey.return_value.train.call_args_list[index].args[0] is memories
        assert modern.return_value.store.call_args_list[index].args[0] is memories
        query = classical.return_value.recall.call_args_list[index].args[0]
        assert storkey.return_value.recall.call_args_list[index].args[0] is query
        assert modern.return_value.recall.call_args_list[index].args[0] is query
        assert seed.call_args_list[2 * index] == seed.call_args_list[2 * index + 1]
    assert seed.call_count == 2 * len(counts)
