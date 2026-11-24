"""Time matched training runs and compare measured runtime with estimated affine work.

Every run in the baseline compute comparison (training size 96, both datasets, logistic
regression and widths 2, 8 and 32, under the common-compute rule and the gradient-norm
rule with an update cap) is timed again with the same split, initialization and update
rule. Each matched run is timed several times, and an untimed pass comes first.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path
from statistics import mean, median, pstdev

import numpy as np
from threadpoolctl import threadpool_limits

from .data import make_dataset, prepare_split
from .experiments import (
    CURVE_INTERVAL, DATA_SEEDS, GRADIENT_TOL, INIT_SEEDS, LEARNING_RATE, MIN_DELTA,
    affine_work_per_checkpoint, cases, limits, write_csv,
)
from .models import initialize
from .training import train

REPEATS = 5


def _time_run(params, split, max_steps, tol):
    """Return the training wall time in seconds and the result of one run."""
    gc.collect()
    gc.disable()
    try:
        start = time.perf_counter()
        result = train(
            params, split.x_train, split.y_train, split.x_val, split.y_val,
            learning_rate=LEARNING_RATE, max_steps=max_steps, weight_decay=0.0,
            patience=None, min_delta=MIN_DELTA, gradient_tol=tol, log_every=CURVE_INTERVAL,
        )
        seconds = time.perf_counter() - start
    finally:
        gc.enable()
    return seconds, result


def measure(data_seeds, init_seeds, repeats):
    """Return one row per timed repeat."""
    design = [case for case in cases() if case.experiment == "budget"]
    datasets = {name: make_dataset(name) for name in {case.dataset for case in design}}
    rows = []
    for data_seed in data_seeds:
        for init_seed in init_seeds:
            # Repeat 0 is an untimed warm-up. Cases are interleaved within each repeat so
            # that slow drift in machine speed does not favor one condition.
            reference = {}
            for repeat in range(repeats + 1):
                for case in design:
                    split = prepare_split(datasets[case.dataset], case.train_size, data_seed)
                    features = split.x_train.shape[1]
                    n_val = len(split.y_val)
                    max_steps, tol, ceiling = limits(case, features, n_val)
                    params = initialize(features, case.width, init_seed)
                    seconds, result = _time_run(params, split, max_steps, tol)
                    work = result.validation_evaluations * affine_work_per_checkpoint(
                        case.width, features, case.train_size, n_val
                    )
                    key = (case.dataset, case.width, case.budget)
                    outcome = (result.steps, result.stop_reason)
                    if reference.setdefault(key, outcome) != outcome:
                        raise RuntimeError(f"Runs are not reproducible for {key}")
                    if repeat == 0:
                        continue
                    rows.append(dict(
                        dataset=case.dataset, width=case.width, train_size=case.train_size,
                        budget=case.budget, data_seed=data_seed, init_seed=init_seed,
                        repeat=repeat, seconds=f"{seconds:.6f}", steps=result.steps,
                        stop_reason=result.stop_reason, compute=work,
                    ))
    return rows


def summarize(rows):
    """Collapse repeats to one time per matched run, then summarize each condition."""
    runs = {}
    for row in rows:
        key = (row["dataset"], row["width"], row["budget"], row["data_seed"], row["init_seed"])
        runs.setdefault(key, []).append(row)
    conditions = {}
    for key, group in runs.items():
        times = [float(row["seconds"]) for row in group]
        entry = dict(
            seconds=median(times), cv=pstdev(times) / mean(times),
            steps=group[0]["steps"], compute=group[0]["compute"],
        )
        conditions.setdefault(key[:3], []).append(entry)
    summary = []
    for (dataset, width, budget), entries in sorted(conditions.items()):
        seconds = median(entry["seconds"] for entry in entries)
        work = mean(entry["compute"] for entry in entries)
        summary.append(dict(
            dataset=dataset, width=width, budget=budget, matched_runs=len(entries),
            median_seconds=f"{seconds:.5f}", mean_updates=f"{mean(e['steps'] for e in entries):.1f}",
            mean_work=f"{work:.0f}", ns_per_affine_term=f"{seconds / work * 1e9:.3f}",
            microseconds_per_checkpoint=f"{seconds / (mean(e['steps'] for e in entries) + 1) * 1e6:.1f}",
            median_repeat_cv=f"{median(entry['cv'] for entry in entries):.4f}",
        ))
    return summary


def log_log_fit(summary):
    """Fit log10 seconds on log10 work across conditions, pooled and by dataset."""
    def fit(items):
        x = np.log10([float(item["mean_work"]) for item in items])
        y = np.log10([float(item["median_seconds"]) for item in items])
        slope, intercept = np.polyfit(x, y, 1)
        r2 = 1 - np.sum((y - (slope * x + intercept)) ** 2) / np.sum((y - y.mean()) ** 2)
        return dict(slope=float(slope), r_squared=float(r2), conditions=len(items))

    fits = {"pooled": fit(summary)}
    for dataset in sorted({item["dataset"] for item in summary}):
        fits[dataset] = fit([item for item in summary if item["dataset"] == dataset])
    return fits


def cost_models(summary):
    """Compare runtime models by median relative error across conditions.

    Each model is fitted by least squares on relative error, so a 0.01 s condition counts
    as much as a 0.2 s one. Checkpoints are updates plus one.
    """
    seconds = np.array([float(item["median_seconds"]) for item in summary])
    checkpoints = np.array([float(item["mean_updates"]) + 1 for item in summary])
    work = np.array([float(item["mean_work"]) for item in summary])
    designs = {
        "work_only": np.column_stack([work]),
        "updates_only": np.column_stack([checkpoints]),
        "updates_plus_work": np.column_stack([checkpoints, work]),
    }
    models = {}
    for name, design in designs.items():
        coefficients, *_ = np.linalg.lstsq(design / seconds[:, None], np.ones(len(seconds)), rcond=None)
        error = np.abs(design @ coefficients - seconds) / seconds
        models[name] = dict(
            coefficients=[float(value) for value in coefficients],
            median_relative_error=float(np.median(error)), max_relative_error=float(error.max()),
        )
    return models


def run(output: Path, quick: bool = False) -> None:
    output.mkdir(parents=True, exist_ok=True)
    data_seeds = DATA_SEEDS[:1] if quick else DATA_SEEDS
    init_seeds = INIT_SEEDS[:1] if quick else INIT_SEEDS
    repeats = 2 if quick else REPEATS
    rows = measure(data_seeds, init_seeds, repeats)
    summary = summarize(rows)
    write_csv(output / "runtime.csv", rows)
    write_csv(output / "runtime_summary.csv", summary)
    metadata = dict(
        quick=quick, data_seeds=data_seeds, init_seeds=init_seeds, repeats=repeats,
        timed_region="train() call only: updates, per-update training pass and validation scoring",
        timer="time.perf_counter, garbage collection paused, one BLAS thread",
        gradient_tolerance=GRADIENT_TOL, learning_rate=LEARNING_RATE,
        fits=log_log_fit(summary), cost_models=cost_models(summary),
        machine=dict(
            platform=platform.platform(), processor=platform.processor(),
            cpu_count=os.cpu_count(), python=sys.version.split()[0], numpy=np.__version__,
        ),
    )
    (output / "runtime.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} timings for {len(summary)} conditions.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--quick", action="store_true", help="One seed pair and two repeats.")
    args = parser.parse_args()
    with threadpool_limits(limits=1, user_api="blas"):
        run(args.output, quick=args.quick)


if __name__ == "__main__":
    main()
