"""NumPy predictive coding for column-oriented classification datasets."""

from .data import load_mnist, prepare_mnist
from .network import XavierInitializer, initialize, sigmoid, softmax, forward_propagation, predict
from .core import (
    output_logit_gradient, compute_errors, compute_cost, compute_energy,
    inference, parameter_gradients,
)
from .optimization import initialize_optimizer_state, weight_update
from .training import mini_batches, evaluate, pc_model

__all__ = [
    "load_mnist", "prepare_mnist", "XavierInitializer", "initialize", "sigmoid",
    "softmax", "forward_propagation", "predict", "output_logit_gradient",
    "compute_errors", "compute_cost", "compute_energy", "inference",
    "parameter_gradients", "initialize_optimizer_state", "weight_update",
    "mini_batches", "evaluate", "pc_model",
]
