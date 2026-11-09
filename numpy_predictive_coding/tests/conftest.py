"""Small deterministic fixtures; no dataset downloads are needed."""

import numpy as np
import pytest

from numpy_predictive_coding import initialize


@pytest.fixture
def problem():
    """Provide three feature columns, one-hot targets, and seeded network weights.

    Convert the small network's Xavier-initialized parameters to float64 for
    accurate finite-difference checks; each test receives fresh arrays.
    """
    X = np.array([[0.2, -0.3, 0.7], [0.4, 0.1, -0.2], [-0.1, 0.6, 0.3]])
    Y = np.eye(2)[[0, 1, 0]].T
    parameters = {k: v.astype(np.float64) for k, v in initialize(3, 4, 2, 2).items()}
    return X, Y, parameters
