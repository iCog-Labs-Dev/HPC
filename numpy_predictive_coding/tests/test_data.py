import io
from urllib.error import URLError

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from numpy_predictive_coding import load_mnist, prepare_mnist
from numpy_predictive_coding import data


def archive_bytes():
    """Build an in-memory MNIST-format archive with known pixels and digit labels."""
    buffer = io.BytesIO()
    np.savez(buffer, x_train=np.zeros((2, 28, 28), dtype=np.uint8),
             y_train=np.array([0, 9]), x_test=np.full((1, 28, 28), 255, dtype=np.uint8),
             y_test=np.array([3]))
    return buffer.getvalue()


def test_prepare_mnist_layout_standardization_and_labels():
    """Prepare black/white images with one known intermediate pixel and labels.

    Assert column-oriented shapes, float32 arrays, pixel scaling/standardization,
    flattening order, and one-hot columns encoding the original digit IDs.
    """
    images = np.stack([np.zeros((28, 28)), np.full((28, 28), 255)]).astype(np.uint8)
    images[0, 2, 3] = 128
    X, Y = prepare_mnist(images, np.array([2, 9]))
    assert X.shape == (784, 2)
    assert Y.shape == (10, 2)
    assert X.dtype == Y.dtype == np.float32
    assert_allclose(X[0], [(0 - 0.1307) / 0.3081, (1 - 0.1307) / 0.3081], rtol=1e-6)
    assert_allclose(X[2 * 28 + 3, 0], (128 / 255 - 0.1307) / 0.3081, rtol=1e-6)
    assert_array_equal(Y.argmax(axis=0), [2, 9])
    assert_array_equal(Y.sum(axis=0), 1)


@pytest.mark.parametrize("images,labels", [
    (np.zeros((0, 28, 28)), np.array([], dtype=int)),
    (np.zeros((1, 784)), np.array([0])),
    (np.zeros((1, 28, 28)), np.array([10])),
    (np.zeros((1, 28, 28)), np.array([-1])),
    (np.zeros((1, 28, 28)), np.array([1.5])),
    (np.zeros((2, 28, 28)), np.array([0])),
    (np.full((1, 28, 28), np.nan), np.array([0])),
    (np.full((1, 28, 28), 256), np.array([0])),
])
def test_invalid_mnist_inputs(images, labels):
    """Pass empty/malformed images, invalid labels, or nonfinite/out-of-range pixels.

    Assert preprocessing rejects each invalid image/label pair with ValueError.
    """
    with pytest.raises(ValueError):
        prepare_mnist(images, labels)


def test_load_local_archive_never_downloads(tmp_path, monkeypatch):
    """Load a local MNIST archive while making any network call fail the test.

    Assert loading uses the existing file, returns the expected train/test
    feature shapes, and preserves the known labels in each split.
    """
    path = tmp_path / "mnist.npz"
    path.write_bytes(archive_bytes())
    def unexpected_download(*args, **kwargs):
        pytest.fail("Existing archive should not trigger a download")
    monkeypatch.setattr(data, "urlopen", unexpected_download)
    X_train, Y_train, X_test, Y_test = load_mnist(path)
    assert X_train.shape == (784, 2)
    assert X_test.shape == (784, 1)
    assert_array_equal(Y_train.argmax(axis=0), [0, 9])
    assert_array_equal(Y_test.argmax(axis=0), [3])


def test_download_then_reuse_default_cache(tmp_path, monkeypatch):
    """Mock a first download, then load again from a temporary default cache.

    Assert exactly one request uses the expected URL/timeout, cached bytes match
    the archive, no temporary files remain, and both loads return identical arrays.
    """
    monkeypatch.setattr(data.tempfile, "gettempdir", lambda: str(tmp_path))
    calls = []
    def download(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(archive_bytes())
    monkeypatch.setattr(data, "urlopen", download)
    first, second = load_mnist(), load_mnist()
    assert calls == [(data.MNIST_URL, 60)]
    cache = tmp_path / "mnist_pc_simple" / "mnist.npz"
    assert cache.read_bytes() == archive_bytes()
    assert list(cache.parent.iterdir()) == [cache]
    for a, b in zip(first, second):
        assert_array_equal(a, b)


def test_download_failure_leaves_no_cache(tmp_path, monkeypatch):
    """Make downloading a missing archive fail with an offline network error.

    Assert URLError reaches the caller and no archive is left at the cache path.
    """
    path = tmp_path / "cache" / "mnist.npz"
    def fail(*args, **kwargs):
        raise URLError("offline")
    monkeypatch.setattr(data, "urlopen", fail)
    with pytest.raises(URLError):
        load_mnist(path)
    assert not path.exists()


def test_invalid_archive_error_propagates(tmp_path):
    """Attempt to load an existing file containing invalid NumPy archive bytes.

    Assert the archive reader's ValueError reaches the caller.
    """
    path = tmp_path / "bad.npz"
    path.write_bytes(b"not a numpy archive")
    with pytest.raises(ValueError):
        load_mnist(path)
