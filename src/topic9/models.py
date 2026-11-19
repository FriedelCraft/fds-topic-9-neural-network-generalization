"""Binary logistic regression and a one-hidden-layer network in NumPy."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Parameters = dict[str, Array]


def initialize(n_features: int, width: int, seed: int) -> Parameters:
    """Use Glorot uniform weights and zero biases. Width zero is the linear model."""
    if n_features < 1 or width < 0:
        raise ValueError("Use at least one feature and a nonnegative network width.")

    rng = np.random.default_rng(seed)
    if width == 0:
        limit = np.sqrt(6.0 / (n_features + 1))
        return {"w": rng.uniform(-limit, limit, n_features), "b": np.zeros(())}

    hidden_limit = np.sqrt(6.0 / (n_features + width))
    output_limit = np.sqrt(6.0 / (width + 1))
    return {
        "w1": rng.uniform(-hidden_limit, hidden_limit, (n_features, width)),
        "b1": np.zeros(width),
        "w2": rng.uniform(-output_limit, output_limit, width),
        "b2": np.zeros(()),
    }


def _forward(params: Parameters, x: Array) -> tuple[Array, Array | None]:
    if "w" in params:
        return x @ params["w"] + params["b"], None

    hidden = np.tanh(x @ params["w1"] + params["b1"])
    return hidden @ params["w2"] + params["b2"], hidden


def _binary_scores(logits: Array, y: Array) -> tuple[Array, Array]:
    sign = 1.0 - 2.0 * y
    return np.logaddexp(0.0, sign * logits), sign


def metrics(params: Parameters, x: Array, y: Array) -> dict[str, float]:
    """Return mean Bernoulli negative log-likelihood and threshold-0.5 accuracy."""
    logits, _ = _forward(params, x)
    losses, _ = _binary_scores(logits, y)
    return {"loss": float(losses.mean()), "accuracy": float(np.mean((logits >= 0) == y))}


def _loss_gradient_metrics(
    params: Parameters, x: Array, y: Array, weight_decay: float
) -> tuple[float, Parameters, dict[str, float]]:
    logits, hidden = _forward(params, x)
    losses, sign = _binary_scores(logits, y)
    # -expm1(-softplus(s*z)) is sigmoid(s*z), including for extreme logits.
    residual = sign * (-np.expm1(-losses)) / len(y)
    if hidden is None:
        grads = {"w": x.T @ residual, "b": np.asarray(residual.sum())}
        weight_keys = ("w",)
    else:
        hidden_gradient = (residual[:, None] * params["w2"]) * (1.0 - hidden**2)
        grads = {
            "w1": x.T @ hidden_gradient,
            "b1": hidden_gradient.sum(axis=0),
            "w2": hidden.T @ residual,
            "b2": np.asarray(residual.sum()),
        }
        weight_keys = ("w1", "w2")

    data_loss = float(losses.mean())
    objective = data_loss
    if weight_decay:
        for key in weight_keys:
            objective += 0.5 * weight_decay * float(np.sum(params[key] ** 2))
            grads[key] += weight_decay * params[key]

    return objective, grads, {
        "loss": data_loss,
        "accuracy": float(np.mean((logits >= 0) == y)),
    }


def loss_and_grad(
    params: Parameters, x: Array, y: Array, weight_decay: float = 0.0
) -> tuple[float, Parameters]:
    """Return the mean training objective and its manual gradient."""
    loss, gradient, _ = _loss_gradient_metrics(params, x, y, weight_decay)
    return loss, gradient


def _objective(params: Parameters, x: Array, y: Array, weight_decay: float) -> float:
    data_loss = metrics(params, x, y)["loss"]
    if weight_decay == 0:
        return data_loss
    penalty = sum(
        float(np.sum(value**2)) for key, value in params.items() if key.startswith("w")
    )
    return data_loss + 0.5 * weight_decay * penalty


def gradient_check(
    params: Parameters,
    x: Array,
    y: Array,
    weight_decay: float = 0.0,
    epsilon: float = 1e-5,
) -> dict[str, float | int]:
    """Perturb every coordinate and compare central differences with backpropagation."""
    if epsilon <= 0:
        raise ValueError("The finite-difference step must be positive.")
    _, analytic = loss_and_grad(params, x, y, weight_decay)
    perturbed = {key: value.copy() for key, value in params.items()}
    errors: list[float] = []
    relative_errors: list[float] = []
    for key, param in perturbed.items():
        for index in np.ndindex(param.shape):
            original = param[index]
            param[index] = original + epsilon
            upper = _objective(perturbed, x, y, weight_decay)
            param[index] = original - epsilon
            lower = _objective(perturbed, x, y, weight_decay)
            param[index] = original
            numeric = (upper - lower) / (2.0 * epsilon)
            exact = float(analytic[key][index])
            error = abs(numeric - exact)
            errors.append(error)
            relative_errors.append(error / max(1e-8, abs(numeric) + abs(exact)))

    return {
        "parameters_checked": len(errors),
        "max_abs_error": max(errors),
        "max_relative_error": max(relative_errors),
    }
