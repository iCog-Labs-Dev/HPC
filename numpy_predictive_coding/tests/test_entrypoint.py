import subprocess
import sys
from pathlib import Path

import numpy as np

import train_mnist


ROOT = Path(__file__).resolve().parents[1]


def test_cli_help_does_not_load_data():
    """Run the training script with --help in a separate Python process.

    Assert it exits successfully and lists cache, inference-step, and shuffle-seed
    options; the help path returns before dataset loading or training.
    """
    result = subprocess.run([sys.executable, str(ROOT / "train_mnist.py"), "--help"],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    assert "--cache-path" in result.stdout
    assert "--steps" in result.stdout
    assert "--shuffle-seed" in result.stdout


def test_imports_do_not_download_train_or_print():
    """Import the package and script with downloading and training forbidden.

    Assert a fresh process imports successfully without invoking either action
    and produces no stdout.
    """
    script = '''
from unittest.mock import patch
with patch("urllib.request.urlopen", side_effect=AssertionError("unexpected download")):
    import predictive_coding
    with patch.object(predictive_coding, "pc_model", side_effect=AssertionError("unexpected training")):
        import train_mnist
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    assert result.stdout == ""


def test_main_routes_options_and_evaluates_held_out_data(monkeypatch, capsys):
    """Call the CLI with custom options and mocked loading/training/evaluation.

    Assert options and the Xavier configuration reach training, only training
    arrays are used to learn, held-out arrays and learned parameters reach
    evaluation, and the expected test metrics are printed.
    """
    X_train, Y_train = np.zeros((784, 2)), np.eye(10)[:, :2]
    X_test, Y_test = np.ones((784, 1)), np.eye(10)[:, :1]
    parameters = {"sentinel": object()}
    def load(path):
        assert path == "local.npz"
        return X_train, Y_train, X_test, Y_test
    def train(X, Y, **kwargs):
        assert X is X_train and Y is Y_train
        assert kwargs["epochs"] == 2
        assert kwargs["h1"] == 5 and kwargs["h2"] == 3
        assert kwargs["batch_size"] == 7 and kwargs["steps"] == 4
        assert kwargs["seed"] == 8 and kwargs["shuffle_seed"] == 9
        assert kwargs["activity_rate"] == 0.03 and kwargs["weight_rate"] == 0.02
        assert kwargs["weight_decay"] == 0.2
        assert kwargs["weight_init"].distribution == "uniform"
        assert kwargs["weight_init"].gain == 1.5
        return parameters, []
    def evaluate(X, Y, actual, batch_size):
        assert X is X_test and Y is Y_test and actual is parameters
        assert batch_size == 7
        return 0.25, 0.75
    monkeypatch.setattr(train_mnist, "load_mnist", load)
    monkeypatch.setattr(train_mnist, "pc_model", train)
    monkeypatch.setattr(train_mnist, "evaluate", evaluate)
    train_mnist.main(["--cache-path", "local.npz", "--epochs", "2", "--h1", "5", "--h2", "3",
                      "--batch-size", "7", "--steps", "4", "--seed", "8", "--shuffle-seed", "9",
                      "--activity-rate", "0.03", "--weight-rate", "0.02", "--weight-decay", "0.2",
                      "--distribution", "uniform", "--gain", "1.5"])
    assert "Test: CE = 0.25000, accuracy = 75.00%" in capsys.readouterr().out
