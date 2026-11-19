"""Full-batch gradient descent and validation checkpointing."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np

from .models import Array, Parameters, _loss_gradient_metrics, metrics


@dataclass
class TrainingResult:
    model: Parameters
    history: list[dict[str, float | int]]
    steps: int
    stop_reason: str
    selected_step: int
    final_gradient_norm: float
    validation_evaluations: int


def _validate_data(x: Array, y: Array) -> None:
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != len(y) or not len(y):
        raise ValueError("Expected a nonempty feature matrix and one label per row.")
    if not (np.isfinite(x).all() and np.isin(y, (0, 1)).all()):
        raise ValueError("Expected finite features and binary labels.")


def train(
    params: Parameters,
    x_train: Array,
    y_train: Array,
    x_val: Array,
    y_val: Array,
    *,
    learning_rate: float,
    max_steps: int,
    weight_decay: float = 0.0,
    patience: int | None = None,
    min_delta: float = 1e-5,
    gradient_tol: float | None = None,
    log_every: int = 1,
) -> TrainingResult:
    """Train with up to max_steps updates, evaluating validation at step 0 and after each.

    With patience, a validation drop greater than min_delta resets its counter.
    Regardless of min_delta, the earliest checkpoint with the lowest validation
    loss is returned. Without patience the terminal iterate is returned.
    The optional gradient criterion uses the penalized training objective at
    the current iterate, before deciding whether to make another update.
    """
    _validate_data(x_train, y_train)
    _validate_data(x_val, y_val)
    if not isfinite(learning_rate) or learning_rate <= 0 or max_steps < 0 or log_every < 1:
        raise ValueError("Use a positive learning rate and log interval, and nonnegative step budget.")
    tolerances = (min_delta, weight_decay, 0.0 if gradient_tol is None else gradient_tol)
    if any(not isfinite(value) or value < 0 for value in tolerances):
        raise ValueError("Loss improvement, weight decay, and gradient tolerance must be nonnegative.")
    if patience is not None and patience < 1:
        raise ValueError("Patience must be at least one evaluation.")

    current = {name: value.copy() for name, value in params.items()}
    best_model = current
    best_loss = float("inf")
    patience_reference = float("inf")
    best_step = 0
    stale_steps = 0
    history: list[dict[str, float | int]] = []
    stop_reason = "budget"

    for step in range(max_steps + 1):
        objective, gradients, train_scores = _loss_gradient_metrics(
            current, x_train, y_train, weight_decay
        )
        grad_norm = sqrt(sum(float(np.vdot(grad, grad)) for grad in gradients.values()))
        val_scores = metrics(current, x_val, y_val)
        val_loss = val_scores["loss"]
        if not all(isfinite(value) for value in (objective, grad_norm, val_loss)):
            raise FloatingPointError(f"Nonfinite objective, gradient, or validation loss at step {step}.")

        if patience is not None:
            if val_loss < best_loss:
                best_model, best_loss, best_step = current, val_loss, step
            if val_loss < patience_reference - min_delta:
                patience_reference, stale_steps = val_loss, 0
            else:
                stale_steps += 1

        terminal = True
        if gradient_tol is not None and grad_norm <= gradient_tol:
            stop_reason = "converged"
        elif patience is not None and stale_steps >= patience:
            stop_reason = "early_stopping"
        else:
            terminal = step == max_steps

        if step % log_every == 0 or terminal:
            history.append({
                "step": step,
                "train_loss": train_scores["loss"],
                "train_accuracy": train_scores["accuracy"],
                "val_loss": val_loss,
                "val_accuracy": val_scores["accuracy"],
                "gradient_norm": grad_norm,
            })
        if terminal:
            break
        current = {
            name: np.asarray(value - learning_rate * gradients[name])
            for name, value in current.items()
        }

    model, selected_step = (best_model, best_step) if patience is not None else (current, step)
    return TrainingResult(model, history, step, stop_reason, selected_step, grad_norm, step + 1)
