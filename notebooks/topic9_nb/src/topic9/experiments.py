"""Run the paired capacity, regularization, label and compute experiments."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .data import make_dataset, prepare_split
from .models import gradient_check, initialize, metrics
from .training import train


DATASETS = ("synthetic", "wdbc")
WIDTHS = (0, 2, 8, 32)
TRAIN_SIZES = (32, 96, 256)
DATA_SEEDS = (301, 302, 303, 304, 305)
INIT_SEEDS = (11, 29, 47)
LEARNING_RATE = 0.2
MAX_STEPS = 800
GRADIENT_TOL = 0.005
DECAYS = (0.0005, 0.005, 0.05)
PATIENCE = 35
MIN_DELTA = 0.0002
EXTENDED_STEPS = 3000
REFERENCE_WIDTH = 8
REFERENCE_STEPS = 180
CURVE_INTERVAL = 5
RANDOM_LABEL_SEED_OFFSET = 10_000


def random_label_seed(data_seed: int) -> int:
    """Label stream paired with one data seed; shared by every width and setting."""
    return RANDOM_LABEL_SEED_OFFSET + data_seed


@dataclass(frozen=True)
class Case:
    experiment: str
    dataset: str
    width: int
    train_size: int
    regularization: str = "none"
    labels: str = "clean"
    budget: str = "convergence"
    condition: str = ""


def cases() -> list[Case]:
    design = []
    for dataset in DATASETS:
        for size in TRAIN_SIZES:
            for width in WIDTHS:
                design.append(Case("capacity", dataset, width, size))
        for size in (32, 256):
            for regularization in ("none", "weight_decay", "early_stopping"):
                design.append(Case("regularization", dataset, 32, size, regularization))
        for size in (32, 96):
            for width in WIDTHS:
                for labels in ("clean", "random"):
                    design.append(Case("random_labels", dataset, width, size, labels=labels))
        for width in WIDTHS:
            for budget in ("fixed", "convergence"):
                design.append(Case("budget", dataset, width, 96, budget=budget))

    wide = Case("diagnostic", "synthetic", 32, 256)
    overfit = Case("diagnostic", "wdbc", 32, 32)
    design.extend(
        (
            replace(wide, condition="optimization_short", budget="fixed"),
            replace(wide, condition="optimization_long"),
            replace(wide, condition="optimization_extended"),
            replace(wide, condition="representation_linear", width=0),
            replace(wide, condition="representation_wide"),
            replace(overfit, condition="overfit_unregularized"),
            replace(overfit, condition="overfit_early_stopping", regularization="early_stopping"),
        )
    )
    return design


def affine_work_per_checkpoint(width: int, features: int, n_train: int, n_val: int) -> int:
    """Estimate affine work for one full training/validation checkpoint."""
    if width == 0:
        return n_train * 2 * features + n_val * features
    return (2 * n_train + n_val) * width * (features + 1)


def limits(case: Case, features: int, n_val: int) -> tuple[int, float | None, int | None]:
    if case.condition == "optimization_short":
        return 8, None, None
    if case.condition == "optimization_extended":
        return EXTENDED_STEPS, GRADIENT_TOL, None
    if case.budget == "convergence":
        return MAX_STEPS, GRADIENT_TOL, None
    ceiling = (REFERENCE_STEPS + 1) * affine_work_per_checkpoint(
        REFERENCE_WIDTH, features, case.train_size, n_val
    )
    per_checkpoint = affine_work_per_checkpoint(case.width, features, case.train_size, n_val)
    return ceiling // per_checkpoint - 1, None, ceiling


def verify_gradients() -> dict[str, dict]:
    rng = np.random.default_rng(815)
    x = rng.normal(size=(7, 3))
    y = rng.integers(0, 2, size=7).astype(float)
    checks = {}
    for width in (0, 4):
        for decay in (0.0, 0.05):
            name = f"width={width}, decay={decay:g}"
            checks[name] = gradient_check(initialize(3, width, 83), x, y, weight_decay=decay)
            if checks[name]["max_abs_error"] > 1e-7:
                raise RuntimeError(f"Finite-difference check failed: {name}: {checks[name]}")
    return checks


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(output: Path, quick: bool = False) -> None:
    output.mkdir(parents=True, exist_ok=True)
    checks = verify_gradients()
    datasets = {name: make_dataset(name) for name in DATASETS}
    data_seeds = DATA_SEEDS[:1] if quick else DATA_SEEDS
    init_seeds = INIT_SEEDS[:1] if quick else INIT_SEEDS
    runs, curves, tuning = [], [], []

    @lru_cache(maxsize=None)
    def split_for(dataset, size, data_seed, labels):
        return prepare_split(
            datasets[dataset], size, data_seed,
            label_seed=random_label_seed(data_seed) if labels == "random" else None,
        )

    @lru_cache(maxsize=None)
    def execute(dataset, size, data_seed, labels, width, init_seed, max_steps, gradient_tol,
                patience, decay):
        split = split_for(dataset, size, data_seed, labels)
        params = initialize(split.x_train.shape[1], width, init_seed)
        return train(
            params, split.x_train, split.y_train, split.x_val, split.y_val,
            learning_rate=LEARNING_RATE, max_steps=max_steps, weight_decay=decay,
            patience=patience, min_delta=MIN_DELTA, gradient_tol=gradient_tol,
            log_every=CURVE_INTERVAL,
        )

    for case in cases():
        print(f"{case.experiment:14} {case.dataset:9} n={case.train_size:3} w={case.width:2} "
              f"{case.regularization:14} {case.labels:6} {case.budget:11} {case.condition}",
              flush=True)
        for data_seed in data_seeds:
            for init_seed in init_seeds:
                split = split_for(case.dataset, case.train_size, data_seed, case.labels)
                features = split.x_train.shape[1]
                n_val = len(split.y_val)
                max_steps, tol, ceiling = limits(case, features, n_val)
                patience = PATIENCE if case.regularization == "early_stopping" else None
                if patience is not None:
                    tol = None
                if quick:
                    max_steps = min(max_steps, 40)

                identity = dict(asdict(case), data_seed=data_seed, init_seed=init_seed)
                candidates = DECAYS if case.regularization == "weight_decay" else (0.0,)
                finalists = []
                for decay in candidates:
                    result = execute(case.dataset, case.train_size, data_seed, case.labels,
                                     case.width, init_seed, max_steps, tol, patience, decay)
                    val_loss = metrics(result.model, split.x_val, split.y_val)["loss"]
                    finalists.append((val_loss, decay, result))
                _, decay, result = min(finalists, key=lambda item: (item[0], item[1]))
                if len(candidates) > 1:
                    for val_loss, candidate, fit in finalists:
                        tuning.append(dict(identity, decay=candidate, val_loss=val_loss,
                                           steps=fit.steps, selected=int(candidate == decay)))

                work_per_checkpoint = affine_work_per_checkpoint(
                    case.width, features, case.train_size, n_val
                )
                score_train = metrics(result.model, split.x_train, split.y_train)
                score_val = metrics(result.model, split.x_val, split.y_val)
                score_test = metrics(result.model, split.x_test, split.y_test)
                runs.append(dict(
                    identity, decay=decay, train_accuracy=score_train["accuracy"],
                    val_accuracy=score_val["accuracy"], test_accuracy=score_test["accuracy"],
                    train_loss=score_train["loss"], val_loss=score_val["loss"],
                    test_loss=score_test["loss"], steps=result.steps,
                    selected_step=result.selected_step,
                    compute=result.validation_evaluations * work_per_checkpoint,
                    compute_ceiling=ceiling, stop_reason=result.stop_reason,
                    final_gradient_norm=result.final_gradient_norm,
                ))
                if case.experiment == "regularization":
                    curves.extend(dict(identity, **row) for row in result.history)

    write_csv(output / "runs.csv", runs)
    write_csv(output / "curves.csv", curves)
    write_csv(output / "tuning.csv", tuning)
    metadata = dict(
        quick=quick, data_seeds=data_seeds, init_seeds=init_seeds, widths=WIDTHS,
        train_sizes=TRAIN_SIZES, learning_rate=LEARNING_RATE, max_steps=MAX_STEPS,
        gradient_tolerance=GRADIENT_TOL, decays=DECAYS, patience=PATIENCE,
        min_delta=MIN_DELTA, extended_steps=EXTENDED_STEPS, reference_width=REFERENCE_WIDTH, reference_steps=REFERENCE_STEPS,
        curve_interval=CURVE_INTERVAL,
        n_runs=len(runs), gradient_checks=checks,
    )
    (output / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(runs)} runs, {len(curves)} curve points and {len(tuning)} tuning rows.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--quick", action="store_true", help="One seed pair and at most 40 updates.")
    args = parser.parse_args()
    with threadpool_limits(limits=1, user_api="blas"):
        run(args.output, quick=args.quick)


if __name__ == "__main__":
    main()
