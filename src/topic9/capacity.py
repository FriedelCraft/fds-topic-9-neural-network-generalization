"""Summarize the paired neural-network and logistic-regression capacity runs."""

import argparse
import csv
import json
from collections import defaultdict
from itertools import product
from math import isclose
from pathlib import Path
from statistics import mean, stdev


DATASETS = ("synthetic", "wdbc")
METRICS = ("test_accuracy", "test_loss")


def read_capacity_runs(path):
    runs = {}
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row["experiment"] != "capacity":
                continue
            key = (row["dataset"], int(row["train_size"]), int(row["width"]),
                   int(row["data_seed"]), int(row["init_seed"]))
            if key in runs:
                raise ValueError(f"Duplicate capacity run: {key}")
            runs[key] = tuple(float(row[metric]) for metric in METRICS)
    return runs


def summarize(results_dir):
    """Pair runs by dataset, training size, data seed and initialization seed."""
    protocol = json.loads((results_dir / "protocol.json").read_text(encoding="utf-8"))
    sizes = protocol["train_sizes"]
    widths = protocol["widths"]
    data_seeds = protocol["data_seeds"]
    init_seeds = protocol["init_seeds"]
    runs = read_capacity_runs(results_dir / "runs.csv")
    expected = set(product(DATASETS, sizes, widths, data_seeds, init_seeds))
    if missing := expected - runs.keys():
        raise ValueError(f"Missing {len(missing)} capacity runs, e.g. {min(missing)}")
    if unexpected := runs.keys() - expected:
        raise ValueError(f"Found {len(unexpected)} unexpected capacity runs, e.g. {min(unexpected)}")

    for dataset, size, width in product(DATASETS, sizes, widths):
        if width == 0:
            continue
        accuracy_deltas = defaultdict(list)
        loss_deltas = defaultdict(list)
        for data_seed, init_seed in product(data_seeds, init_seeds):
            accuracy, loss = runs[dataset, size, width, data_seed, init_seed]
            baseline_accuracy, baseline_loss = runs[dataset, size, 0, data_seed, init_seed]
            accuracy_deltas[data_seed].append(100 * (accuracy - baseline_accuracy))
            loss_deltas[data_seed].append(loss - baseline_loss)

        accuracy_by_seed = [mean(accuracy_deltas[seed]) for seed in data_seeds]
        loss_by_seed = [mean(loss_deltas[seed]) for seed in data_seeds]
        summary = dict(dataset=dataset, train_size=size, width=width,
                       n_pairs=len(data_seeds) * len(init_seeds), n_data_seeds=len(data_seeds))
        for name, values in (("test_accuracy_delta_pp", accuracy_by_seed),
                             ("test_loss_delta", loss_by_seed)):
            summary[f"{name}_mean"] = f"{mean(values):.9f}"
            summary[f"{name}_sd"] = f"{stdev(values):.9f}" if len(values) > 1 else ""
        summary["data_seeds_with_higher_accuracy"] = sum(
            delta > 0 and not isclose(delta, 0, abs_tol=1e-12) for delta in accuracy_by_seed
        )
        summary["data_seeds_with_lower_loss"] = sum(
            delta < 0 and not isclose(delta, 0, abs_tol=1e-12) for delta in loss_by_seed
        )
        yield summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="?", default=Path("results"))
    args = parser.parse_args()
    rows = list(summarize(args.results))
    output = args.results / "capacity_comparison.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} paired capacity comparisons to {output}.")


if __name__ == "__main__":
    main()
