"""Convergence sensitivity on noisy XOR: vary the update cap and the learning rate separately.

All runs use the unregularized network, 256 training examples, the baseline gradient-norm
threshold, and the paired data and initialization seeds of the main experiments. The cap
sweep holds the learning rate at its baseline value. The learning-rate sweep holds the
cap at its baseline value. Each row records losses, the gradient norm, and the reason the
run stopped.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

from threadpoolctl import threadpool_limits

from .data import make_dataset, prepare_split
from .experiments import (
    DATA_SEEDS, GRADIENT_TOL, INIT_SEEDS, LEARNING_RATE, MAX_STEPS, MIN_DELTA, write_csv,
)
from .models import initialize, metrics
from .training import train

DATASET = "synthetic"
TRAIN_SIZE = 256
WIDTHS = (8, 32)
CAPS = (100, 200, 400, 800, 1600, 3200)
LEARNING_RATES = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
LOOKBACK = 100


def _trajectory_summary(history):
    """Describe the training-loss path from the per-update history."""
    losses = [row["train_loss"] for row in history]
    increases = [later - earlier for earlier, later in zip(losses, losses[1:]) if later > earlier]
    lookback = history[max(len(history) - 1 - LOOKBACK, 0)]["train_loss"]
    return dict(
        loss_increases=len(increases),
        largest_loss_increase=max(increases, default=0.0),
        train_loss_drop_last_100=lookback - losses[-1] if len(losses) > LOOKBACK else "",
    )


def run(output: Path, quick: bool = False) -> None:
    output.mkdir(parents=True, exist_ok=True)
    dataset = make_dataset(DATASET)
    data_seeds = DATA_SEEDS[:1] if quick else DATA_SEEDS
    init_seeds = INIT_SEEDS[:1] if quick else INIT_SEEDS

    @lru_cache(maxsize=None)
    def execute(width, data_seed, init_seed, learning_rate, max_steps):
        split = prepare_split(dataset, TRAIN_SIZE, data_seed)
        params = initialize(split.x_train.shape[1], width, init_seed)
        try:
            result = train(
                params, split.x_train, split.y_train, split.x_val, split.y_val,
                learning_rate=learning_rate, max_steps=max_steps, min_delta=MIN_DELTA,
                gradient_tol=GRADIENT_TOL, log_every=1,
            )
        except FloatingPointError:
            return dict(steps="", stop_reason="nonfinite")
        score = {name: metrics(result.model, getattr(split, f"x_{name}"), getattr(split, f"y_{name}"))
                 for name in ("train", "val", "test")}
        return dict(
            steps=result.steps, stop_reason=result.stop_reason,
            train_loss=score["train"]["loss"], val_loss=score["val"]["loss"],
            test_loss=score["test"]["loss"], train_accuracy=score["train"]["accuracy"],
            test_accuracy=score["test"]["accuracy"],
            final_gradient_norm=result.final_gradient_norm,
            initial_gradient_norm=result.history[0]["gradient_norm"],
            **_trajectory_summary(result.history),
        )

    settings = [("cap", LEARNING_RATE, cap) for cap in CAPS]
    settings += [("learning_rate", rate, MAX_STEPS) for rate in LEARNING_RATES]
    rows = []
    for sweep, rate, cap in settings:
        print(f"{sweep:14} lr={rate:<5g} cap={cap}", flush=True)
        for width in WIDTHS:
            for data_seed in data_seeds:
                for init_seed in init_seeds:
                    outcome = execute(width, data_seed, init_seed, rate, cap)
                    rows.append(dict(
                        sweep=sweep, dataset=DATASET, train_size=TRAIN_SIZE, width=width,
                        learning_rate=rate, max_steps=cap, data_seed=data_seed,
                        init_seed=init_seed, **outcome,
                    ))
    fields = {name: "" for row in rows for name in row}
    write_csv(output / "convergence.csv", [dict(fields, **row) for row in rows])
    metadata = dict(
        quick=quick, dataset=DATASET, train_size=TRAIN_SIZE, widths=WIDTHS, caps=CAPS,
        learning_rates=LEARNING_RATES, baseline_learning_rate=LEARNING_RATE,
        baseline_cap=MAX_STEPS, gradient_tolerance=GRADIENT_TOL, data_seeds=data_seeds,
        init_seeds=init_seeds, n_runs=len(rows),
    )
    (output / "convergence.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} convergence runs.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--quick", action="store_true", help="One seed pair.")
    args = parser.parse_args()
    with threadpool_limits(limits=1, user_api="blas"):
        run(args.output, quick=args.quick)


if __name__ == "__main__":
    main()
