"""Audit the random-label streams and compare training fit with chance on held-out data.

Under the random-label control every validation and test label is an independent
Bernoulli(0.5) draw that the model never sees. A model's predictions depend only on the
features and the training labels, so for any fitted model the number of correct test
predictions is exactly Binomial(n_test, 0.5). The summary checks each run against that
null distribution and sets it beside the training accuracy the same run achieved.
"""

import argparse
import csv
import json
from collections import defaultdict
from itertools import product
from math import exp, lgamma, log, sqrt
from pathlib import Path
from statistics import NormalDist, mean, stdev

import numpy as np

from .data import make_dataset, prepare_split
from .experiments import DATASETS, random_label_seed

LEVEL = 0.95


def binomial_interval(n: int, p: float = 0.5, level: float = LEVEL) -> tuple[float, float]:
    """Central interval [lo, hi] of Binomial(n, p)/n with coverage at least ``level``."""
    log_pmf = [lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + k * log(p) + (n - k) * log(1 - p)
               for k in range(n + 1)]
    tail = (1 - level) / 2
    cdf, lo = 0.0, 0
    for k, value in enumerate(log_pmf):
        if cdf + exp(value) > tail:
            lo = k
            break
        cdf += exp(value)
    cdf, hi = 0.0, n
    for k in range(n, -1, -1):
        if cdf + exp(log_pmf[k]) > tail:
            hi = k
            break
        cdf += exp(log_pmf[k])
    return lo / n, hi / n


def max_feature_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Largest absolute Pearson correlation between the labels and any feature column."""
    x = x - x.mean(axis=0)
    y = y - y.mean()
    scale = np.sqrt((x * x).sum(axis=0) * (y * y).sum())
    scale = np.where(scale > 0, scale, np.inf)
    return float(np.max(np.abs(x.T @ y) / scale))


def audit(data_seeds, train_size: int):
    """One row per dataset, data seed and split describing the random labels drawn there."""
    for name in DATASETS:
        dataset = make_dataset(name)
        for data_seed in data_seeds:
            clean = prepare_split(dataset, train_size, data_seed)
            rand = prepare_split(dataset, train_size, data_seed, label_seed=random_label_seed(data_seed))
            for split in ("train", "val", "test"):
                x = getattr(rand, f"x_{split}")
                y = getattr(rand, f"y_{split}")
                n, d = x.shape
                # Two-sided Bonferroni bound over d columns; |r| * sqrt(n) is approximately N(0, 1).
                bound = NormalDist().inv_cdf(1 - (1 - LEVEL) / (2 * d)) / sqrt(n)
                yield dict(
                    dataset=name, data_seed=data_seed, label_seed=random_label_seed(data_seed),
                    split=split, n=n, positive_rate=f"{y.mean():.4f}",
                    agreement_with_true_labels=f"{np.mean(y == getattr(clean, f'y_{split}')):.4f}",
                    max_abs_feature_correlation=f"{max_feature_correlation(x, y):.4f}",
                    correlation_bound=f"{bound:.4f}",
                )


def read_runs(path):
    runs = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row["experiment"] == "random_labels":
                key = (row["dataset"], int(row["train_size"]), int(row["width"]), row["labels"])
                runs[key].append(row)
    return runs


def summarize(results_dir):
    """Per condition: training fit, held-out accuracy, and how many runs sit inside the chance interval."""
    protocol = json.loads((results_dir / "protocol.json").read_text(encoding="utf-8"))
    runs = read_runs(results_dir / "runs.csv")
    expected = len(protocol["data_seeds"]) * len(protocol["init_seeds"])
    n_test = {name: len(make_dataset(name).y_test) for name in DATASETS}
    sizes = sorted({key[1] for key in runs})
    for dataset, size, width, labels in product(DATASETS, sizes, protocol["widths"], ("clean", "random")):
        rows = runs[dataset, size, width, labels]
        if len(rows) != expected:
            raise ValueError(f"Expected {expected} runs for {(dataset, size, width, labels)}, found {len(rows)}")
        by_seed = defaultdict(list)
        for row in rows:
            by_seed[row["data_seed"]].append(row)
        seed_means = {metric: [mean(float(r[metric]) for r in group) for group in by_seed.values()]
                      for metric in ("train_accuracy", "test_accuracy")}
        test = [float(r["test_accuracy"]) for r in rows]
        summary = dict(dataset=dataset, train_size=size, width=width, labels=labels, n_runs=len(rows))
        for metric in ("train_accuracy", "test_accuracy"):
            summary[f"{metric}_mean"] = f"{100 * mean(seed_means[metric]):.2f}"
            values = seed_means[metric]
            summary[f"{metric}_sd"] = f"{100 * stdev(values):.2f}" if len(values) > 1 else ""
        summary["runs_fitting_all_train_labels"] = sum(float(r["train_accuracy"]) == 1.0 for r in rows)
        summary["generalization_gap_pp"] = f"{100 * (mean(float(r['train_accuracy']) for r in rows) - mean(test)):.2f}"
        lo, hi = binomial_interval(n_test[dataset]) if labels == "random" else (None, None)
        summary["chance_interval"] = f"[{100 * lo:.1f}, {100 * hi:.1f}]" if lo is not None else ""
        summary["runs_inside_chance_interval"] = (sum(lo <= t <= hi for t in test) if lo is not None else "")
        summary["test_accuracy_min"] = f"{100 * min(test):.2f}"
        summary["test_accuracy_max"] = f"{100 * max(test):.2f}"
        yield summary


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, nargs="?", default=Path("results"))
    args = parser.parse_args()
    protocol = json.loads((args.results / "protocol.json").read_text(encoding="utf-8"))
    sizes = sorted({int(r["train_size"]) for rows in read_runs(args.results / "runs.csv").values() for r in rows})
    audit_rows = list(audit(protocol["data_seeds"], max(sizes)))
    summary_rows = list(summarize(args.results))
    write_csv(args.results / "random_labels_audit.csv", audit_rows)
    write_csv(args.results / "random_labels_summary.csv", summary_rows)
    print(f"Wrote {len(audit_rows)} label audits and {len(summary_rows)} condition summaries to {args.results}.")


if __name__ == "__main__":
    main()
