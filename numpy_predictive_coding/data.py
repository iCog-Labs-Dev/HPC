"""MNIST I/O and preprocessing, independent of the training core."""

from pathlib import Path
import tempfile
from urllib.request import urlopen

import numpy as np


MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def prepare_mnist(images, labels):
    """Flatten uint8 images into standardized float32 columns and one-hot labels.

    Accept (samples, 28, 28) images and integer digit IDs. Return X (784, m)
    and Y (10, m), using the notebook's fixed MNIST mean/std: 0.1307/0.3081.
    """
    if (not isinstance(images, np.ndarray) or images.ndim != 3
            or images.shape[1:] != (28, 28) or images.shape[0] == 0):
        raise ValueError("images must be a nonempty array with shape (samples, 28, 28).")
    if (not isinstance(labels, np.ndarray) or labels.shape != (images.shape[0],)
            or not np.issubdtype(labels.dtype, np.integer)
            or np.any((labels < 0) | (labels > 9))):
        raise ValueError("labels must contain one integer digit ID (0-9) per image.")
    if not np.all(np.isfinite(images)) or np.any((images < 0) | (images > 255)):
        raise ValueError("MNIST pixels must be finite values between 0 and 255.")
    X = images.reshape(-1, 784).T.astype(np.float32) / 255.0
    X = (X - np.float32(0.1307)) / np.float32(0.3081)
    Y = np.eye(10, dtype=np.float32)[labels].T
    return X, Y


def load_mnist(path=None):
    """Return X_train, Y_train, X_test, Y_test; download only if cache is absent.

    An explicit path may point to an existing Keras-format MNIST .npz archive
    or a desired cache location. Network and archive errors propagate to callers.
    """
    path = (Path(path) if path is not None else
            Path(tempfile.gettempdir()) / "mnist_pc_simple" / "mnist.npz")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(MNIST_URL, timeout=60) as response:
            content = response.read()
        # Do not leave a partially written cache if a write is interrupted.
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
                temporary_path = Path(handle.name)
                handle.write(content)
            temporary_path.replace(path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    with np.load(path, allow_pickle=False) as data:
        train = prepare_mnist(data["x_train"], data["y_train"])
        test = prepare_mnist(data["x_test"], data["y_test"])
    return (*train, *test)
