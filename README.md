# HPC: NumPy predictive coding

A reusable extraction of `mnist_pc_simple.ipynb`. The original notebook remains
unchanged as a reference. Training and tests run directly from Python.

## Setup

Python 3.10 or newer is required. NumPy is the only runtime dependency; pytest
is included in the development extra.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

Tests use tiny synthetic arrays and mocked downloads, so they do not need
network access or the MNIST dataset.

## Train MNIST

```sh
python train_mnist.py
python train_mnist.py --epochs 2 --h1 64 --h2 32
python train_mnist.py --cache-path /path/to/mnist.npz
python train_mnist.py --help
```

The default run uses the notebook's full MNIST settings: `784 → 256 → 64 → 10`,
20 epochs, batches of 200, 20 inference steps at activity rate 0.05, and AdamW
with learning rate 0.001 and weight decay 0.1. Initialization uses Xavier normal
with seed 0; shuffling uses seed 42. The script reports training metrics at epoch
zero and after every epoch, then evaluates the separate test set once.

The first data load downloads the Keras MNIST archive to
`<system temporary directory>/mnist_pc_simple/mnist.npz`; subsequent runs reuse
it. `--cache-path` selects an existing archive or a download destination.
Pixels are scaled by 255 and standardized using mean 0.1307 and standard
deviation 0.3081. Importing the package or script does not load data or train.

## Use another classification dataset

Prepare floating-point NumPy arrays with **samples as columns**: `X` has shape
`(number_of_features, number_of_samples)` and one-hot `Y` has shape
`(number_of_classes, number_of_samples)`. Apply any dataset-specific scaling
before calling the training API, fitting preprocessing on training data only.

```python
import numpy as np
from predictive_coding import pc_model, evaluate, predict

# Stand-in for your prepared training and held-out arrays.
X_train = np.array([[-1, -0.8, 0.8, 1], [-0.9, -1, 1, 0.9]], dtype=np.float32)
Y_train = np.eye(2, dtype=np.float32)[[0, 0, 1, 1]].T
X_test = np.array([[-0.95, 0.95], [-0.95, 0.95]], dtype=np.float32)
Y_test = np.eye(2, dtype=np.float32)

parameters, history = pc_model(
    X_train, Y_train, h1=16, h2=8, epochs=10,
    batch_size=2, steps=20, verbose=False,
)
loss, accuracy = evaluate(X_test, Y_test, parameters)
classes = predict(X_test, parameters)
```

The input and output widths come from the arrays. There are exactly two
configurable sigmoid hidden layers and a softmax output. History is a list of
`(epoch, forward_cross_entropy, forward_accuracy)` tuples, including epoch zero.
`initial_parameters` accepts a parameter dictionary that is copied before
training; its shapes define the network instead of `h1`/`h2`.

## Modules and contracts

| Module | Responsibility |
| --- | --- |
| `predictive_coding.data` | MNIST preprocessing, loading, and caching |
| `predictive_coding.network` | Xavier initialization, activations, forward pass, prediction |
| `predictive_coding.core` | Errors, loss, energy, inference, local gradients |
| `predictive_coding.optimization` | AdamW state and updates |
| `predictive_coding.training` | Mini-batches, evaluation, training orchestration |
| `train_mnist.py` | Dataset-specific executable |

Public functions are also exported from `predictive_coding` for convenient
imports. Parameters use keys `W1`, `b1`, `W2`, `b2`, `W3`, `b3`; weights have
shape `(next_width, previous_width)` and biases `(next_width, 1)`. The forward
cache contains `Z1`/`A1` through `Z3`/`A3`. The extracted API spells
`forward_propagation` correctly; the unchanged notebook retains its original
`forwared_propagation` spelling.

For each batch, training performs a forward pass, copies its hidden activities,
infers activities with fixed weights, recomputes errors, calculates local
gradients, and makes one AdamW update. Hidden activities update simultaneously
and remain unconstrained. Gradients average over the actual batch size,
including the final partial batch. AdamW moments persist across batches and
epochs; decay applies to biases as well as weights.

`inference` returns `(H1, H2, errors)` and leaves parameters and its forward cache
unchanged. `parameter_gradients` expects errors from those same states and
parameters, and returns `dW1`/`db1` through `dW3`/`db3`. `weight_update` mutates
both the parameter dictionary and optimizer state and returns the dictionary.
Prediction and evaluation use only forward probabilities.

The math functions assume arrays follow the shapes described above; there is
no separate validation module or repeated layer-shape checking. Basic checks
remain at the data/training entry points, along with finite-value checks that
stop unstable training. AdamW initializes and updates the six named parameters
explicitly, using a shared formula and a single step increment per batch.

## What the tests verify

- Forward-initialized hidden errors `e1` and `e2` are exactly zero. The output
  signal `p - Y` is generally nonzero. On the first simultaneous inference step,
  `H1` stays fixed and `H2` moves.
- Central finite differences check hidden-state derivatives and every weight
  and bias derivative with inferred activities held fixed.
- Hand calculations verify cross-entropy, hidden energy, forward prediction,
  simultaneous inference, and two AdamW updates with bias correction.
- Batches preserve feature/label alignment and partial batches; evaluation
  weights each example equally. Repeated seeded training is deterministic.
- Synthetic training lowers loss and learns a simple classification problem.
  Local archives, mocked downloads, CLI wiring, input validation, and unstable
  training failures are covered too.

The reported cross-entropy clips probabilities to `[1e-7, 1]`, matching the
notebook. The training output derivative is the usual `p - Y`, which does not
differentiate that clipping. Numerical derivative tests use float64 values away
from the clipping boundary. Full MNIST training is a separate manual run, not
part of the fast test suite.
