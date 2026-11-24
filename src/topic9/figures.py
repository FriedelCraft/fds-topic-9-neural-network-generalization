"""Render the experiment CSVs as standalone figures for the project report."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator, MultipleLocator, PercentFormatter


DATASETS = {"synthetic": "Synthetic XOR", "wdbc": "Wisconsin diagnostic dataset"}
METHODS = {"none": "Unregularized", "weight_decay": "Weight decay", "early_stopping": "Early stopping"}
METHOD_COLORS = ("#64748B", "#00796B", "#8C4084")
COLORS = ("#146EAC", "#C26813", "#16836B")
INK = "#223047"
LIGHT_INK = "#4E6075"
INT_FIELDS = {"width", "train_size", "data_seed", "init_seed", "steps", "selected_step", "step", "compute"}
RC = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.edgecolor": "#A6B0BD",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.axisbelow": True,
    "grid.color": "#E2E7ED",
    "grid.linewidth": 0.8,
    "xtick.color": LIGHT_INK,
    "ytick.color": LIGHT_INK,
    "svg.fonttype": "none",
    "svg.hashsalt": "mac312-topic9",
}
SPREAD_NOTE = "Points are means across data seeds after averaging initializations. Error bars show ±1 SD across data seeds."


def _read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        for field, value in row.items():
            if value and field in INT_FIELDS:
                row[field] = int(value)
            elif value and field.endswith(("_loss", "_accuracy")):
                row[field] = float(value)
    return rows


def _select(rows, **criteria):
    return [row for row in rows if all(row.get(field) == value for field, value in criteria.items())]


def _levels(rows, field):
    return sorted({row[field] for row in rows})


def _summary(rows, metric):
    if not rows:
        raise ValueError(f"Cannot summarize {metric} for an empty condition")
    by_data_seed = defaultdict(list)
    for row in rows:
        by_data_seed[row["data_seed"]].append(row[metric])
    seed_means = [mean(values) for values in by_data_seed.values()]
    spread = stdev(seed_means) if len(seed_means) > 1 else 0
    return mean(seed_means), spread


def _series(ax, positions, groups, metric, *, color, label, marker="o", linestyle="-", log_scale=False):
    values, errors = zip(*(_summary(group, metric) for group in groups))
    # Spreads can extend beyond the metric's domain. Clip only those endpoints.
    floor = 1 if log_scale else 0
    low = [min(error, max(value - floor, 0)) for value, error in zip(values, errors)]
    high = errors
    if metric.endswith("_accuracy"):
        high = [min(error, 1 - value) for value, error in zip(values, errors)]
    ax.errorbar(
        positions, values, yerr=(low, high), color=color, label=label, marker=marker,
        linestyle=linestyle, linewidth=1.8, markersize=5, capsize=3, elinewidth=1,
    )


def _panel(ax, title, ylabel, *, accuracy=False, log_scale=False):
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11, pad=9)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")
    ax.tick_params(axis="x", length=0, pad=7)
    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(numticks=5))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        if accuracy:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))


def _wide_accuracy_axis(ax):
    ax.set_ylim(0.34, 1.04)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))


def _width_axis(ax, widths):
    labels = [str(width) if width else "Linear" for width in widths]
    ax.set_xticks(range(len(widths)), labels)
    ax.set_xlabel("Hidden units")


def _header(fig, title, subtitle, note):
    fig.text(0.055, 0.985, title, va="top", fontsize=19, fontweight="bold")
    fig.text(0.055, 0.931, subtitle, va="top", fontsize=10, color=LIGHT_INK)
    fig.text(0.055, 0.025, note, va="bottom", fontsize=9, color=LIGHT_INK)


def _legend(fig, ax, columns, *, loc=(0.055, 0.876)):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=loc, ncol=columns,
               frameon=False, borderaxespad=0, handlelength=2)


def _save(fig, path):
    metadata = {"Date": None} if path.suffix == ".svg" else None
    fig.savefig(path, dpi=180, metadata=metadata)
    if path.suffix == ".svg":
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(line.rstrip() for line in lines) + "\n",
                        encoding="utf-8", newline="\n")
    plt.close(fig)


def _capacity(rows, path):
    widths = _levels(rows, "width")
    sizes = _levels(rows, "train_size")
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(11, 8), squeeze=False)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.80, bottom=0.13, hspace=0.63, wspace=0.29)
    _header(fig, "Capacity and training size", "Held-out accuracy and cross-entropy across model widths and sample sizes.", SPREAD_NOTE)

    for row_axes, (dataset, name) in zip(axes, DATASETS.items()):
        source = _select(rows, dataset=dataset)
        for ax, (metric, label) in zip(row_axes, (("test_accuracy", "Test accuracy"), ("test_loss", "Test cross-entropy"))):
            for size, color, marker in zip(sizes, COLORS, ("o", "s", "^")):
                groups = [_select(source, train_size=size, width=width) for width in widths]
                _series(ax, range(len(widths)), groups, metric, color=color, label=f"n = {size}", marker=marker)
            _panel(ax, name, label, accuracy=metric == "test_accuracy")
            _width_axis(ax, widths)
            if metric == "test_accuracy":
                _wide_accuracy_axis(ax)
            else:
                ax.set_ylim(bottom=0)
    _legend(fig, axes[0, 0], len(sizes), loc=(0.055, 0.855))
    _save(fig, path)


def _regularization(rows, curves, path):
    sizes = _levels(rows, "train_size")
    fig, axes = plt.subplots(len(DATASETS), 3, figsize=(14, 8), squeeze=False)
    fig.subplots_adjust(left=0.078, right=0.975, top=0.81, bottom=0.13, hspace=0.60, wspace=0.37)
    _header(fig, "Regularization and validation", "Learning curves for a matched run at the smallest n. Test panels summarize all repeats.", SPREAD_NOTE)

    for row_axes, (dataset, name) in zip(axes, DATASETS.items()):
        source = _select(rows, dataset=dataset)
        history = _select(curves, dataset=dataset, train_size=sizes[0])
        curve_ax = row_axes[0]
        by_method = {
            method: {(record["data_seed"], record["init_seed"]) for record in _select(history, regularization=method)}
            for method in METHODS
        }
        data_seed, init_seed = min(set.intersection(*by_method.values()))
        pair = _select(history, data_seed=data_seed, init_seed=init_seed)
        early_run, = _select(source, data_seed=data_seed, init_seed=init_seed, regularization="early_stopping",
                             train_size=sizes[0])
        for (method, method_label), color in zip(METHODS.items(), METHOD_COLORS):
            samples = sorted(_select(pair, regularization=method), key=lambda sample: sample["step"])
            for metric, linestyle in (("val_loss", "-"), ("train_loss", "--")):
                points = [(sample["step"], sample[metric]) for sample in samples]
                steps, losses = zip(*points)
                curve_ax.plot(steps, losses, color=color, linestyle=linestyle, linewidth=1.6,
                              label=method_label if metric == "val_loss" else None)
                if method == "early_stopping":
                    curve_ax.scatter(early_run["selected_step"], early_run[metric], color=color, s=22, zorder=3)
        curve_ax.set_xlabel("Gradient updates")
        _panel(curve_ax, f"{name} | n = {sizes[0]}", "Cross-entropy")
        curve_ax.text(0.98, 0.98, f"Seeds: {data_seed} / {init_seed}\nSolid: validation\nDashed: training\nDot: selected checkpoint", ha="right", va="top",
                      transform=curve_ax.transAxes, color=LIGHT_INK, fontsize=8,
                      bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2})
        for ax, (metric, label) in zip(row_axes[1:], (("test_accuracy", "Test accuracy"), ("test_loss", "Test cross-entropy"))):
            for position, ((method, method_label), color) in enumerate(zip(METHODS.items(), METHOD_COLORS)):
                offset = (position - 1) * 0.17
                groups = [_select(source, train_size=size, regularization=method) for size in sizes]
                _series(ax, [idx + offset for idx in range(len(sizes))], groups, metric,
                        color=color, label=method_label, marker=("o", "s", "^")[position], linestyle="none")
            _panel(ax, name, label, accuracy=metric == "test_accuracy")
            ax.set_xticks(range(len(sizes)), sizes)
            ax.set_xlim(-0.45, len(sizes) - 0.55)
            ax.set_xlabel("Training examples")
            if metric == "test_accuracy":
                lower, upper = ax.get_ylim()
                ax.set_ylim(max(0, lower - 0.025), min(1.04, upper + 0.025))
            else:
                ax.set_ylim(bottom=0)
    _legend(fig, axes[0, 0], len(METHODS), loc=(0.055, 0.855))
    _save(fig, path)


def _random_labels(rows, path):
    widths = _levels(rows, "width")
    sizes = _levels(rows, "train_size")
    fig, axes = plt.subplots(len(DATASETS), len(sizes), figsize=(11, 8), sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.80, bottom=0.125, hspace=0.58, wspace=0.19)
    _header(fig, "Learning and memorization", "Training and test accuracy with population labels and independently randomized labels.", SPREAD_NOTE)

    for row_axes, (dataset, name) in zip(axes, DATASETS.items()):
        for ax, size in zip(row_axes, sizes):
            source = _select(rows, dataset=dataset, train_size=size)
            for label_mode, color in (("clean", COLORS[0]), ("random", COLORS[1])):
                groups = [_select(source, width=width, labels=label_mode) for width in widths]
                for split, linestyle, marker in (("train", "--", "s"), ("test", "-", "o")):
                    _series(ax, range(len(widths)), groups, f"{split}_accuracy", color=color,
                            label=f"{label_mode.title()} {split}", linestyle=linestyle, marker=marker)
            _panel(ax, f"{name} | n = {size}", "Accuracy" if ax is row_axes[0] else "", accuracy=True)
            _wide_accuracy_axis(ax)
            ax.axhline(0.5, color="#8290A0", linewidth=1, linestyle=":", label="50% chance")
            _width_axis(ax, widths)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    order = ("Clean train", "Clean test", "Random train", "Random test", "50% chance")
    fig.legend([by_label[label] for label in order], order, loc="lower left", bbox_to_anchor=(0.055, 0.855),
               ncol=len(order), frameon=False, borderaxespad=0)
    _save(fig, path)


def _budgets(rows, path):
    widths = _levels(rows, "width")
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(11, 8), squeeze=False)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.81, bottom=0.13, hspace=0.59, wspace=0.33)
    _header(fig, "Compute budget and convergence", "Performance and work by stopping protocol. The work axis is logarithmic.", SPREAD_NOTE)

    for row_axes, (dataset, name) in zip(axes, DATASETS.items()):
        source = _select(rows, dataset=dataset)
        for ax, (metric, label) in zip(row_axes, (("test_loss", "Test cross-entropy"), ("compute", "Estimated affine work"))):
            for budget, title, color, marker, offset in (
                ("fixed", "Common compute budget", COLORS[0], "o", -0.05),
                ("convergence", "Convergence criterion with cap", COLORS[1], "s", 0.05),
            ):
                groups = [_select(source, width=width, budget=budget) for width in widths]
                positions = [index + offset for index in range(len(widths))]
                _series(ax, positions, groups, metric, color=color, label=title, marker=marker,
                        log_scale=metric == "compute")
            _panel(ax, name, label, log_scale=metric == "compute")
            _width_axis(ax, widths)
            if metric == "test_loss":
                ax.set_ylim(bottom=0)
    _legend(fig, axes[0, 0], 2, loc=(0.055, 0.855))
    _save(fig, path)


def _diagnostics(rows, path):
    """Draw one panel per diagnostic contrast."""
    optimization = [("optimization_short", "8 updates"), ("optimization_long", "Standard cap")]
    if any(row["condition"] == "optimization_extended" for row in rows):
        optimization.append(("optimization_extended", "Extended cap"))
    contrasts = (
        ("Optimization | Synthetic XOR", optimization),
        ("Representation | Synthetic XOR", (("representation_linear", "Logistic regression"),
                                             ("representation_wide", "32 hidden units"))),
        ("Overfitting | Wisconsin", (("overfit_unregularized", "Unregularized"),
                                     ("overfit_early_stopping", "Early stopping"))),
    )
    fig, axes = plt.subplots(1, len(contrasts), figsize=(12, 5.3), sharey=True)
    fig.subplots_adjust(left=0.083, right=0.98, top=0.73, bottom=0.16, wspace=0.22)
    _header(fig, "Controlled diagnostic runs", "Training and held-out cross-entropy for each control and intervention.", SPREAD_NOTE)
    for ax, (title, conditions) in zip(axes, contrasts):
        groups = [_select(rows, condition=condition) for condition, _ in conditions]
        positions = range(len(groups))
        for offset, metric, label, color, marker in (
            (-0.17, "train_loss", "Training", COLORS[0], "o"),
            (0, "val_loss", "Validation", COLORS[2], "^"),
            (0.17, "test_loss", "Test", COLORS[1], "s"),
        ):
            _series(ax, [x + offset for x in positions], groups, metric,
                    color=color, label=label, marker=marker, linestyle="none")
        _panel(ax, title, "Cross-entropy" if ax is axes[0] else "")
        ax.set_xticks(positions, [label for _, label in conditions])
        ax.set_xlim(-0.42, len(groups) - 0.58)
    axes[0].set_ylim(bottom=0)
    _legend(fig, axes[0], 3, loc=(0.055, 0.79))
    _save(fig, path)


def generate_figures(results_dir, output_dir=None, image_format="svg"):
    """Render the project figures, returning the paths that were written."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir is not None else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = _read_csv(results_dir / "runs.csv")
    curves = _read_csv(results_dir / "curves.csv")
    paths = []
    with plt.rc_context(RC):
        for experiment, draw in (
            ("capacity", _capacity), ("regularization", _regularization),
            ("random_labels", _random_labels), ("budget", _budgets), ("diagnostic", _diagnostics),
        ):
            rows = _select(runs, experiment=experiment)
            if not rows:
                continue
            path = output_dir / f"{experiment}.{image_format}"
            if experiment == "regularization":
                draw(rows, _select(curves, experiment=experiment), path)
            else:
                draw(rows, path)
            paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", nargs="?", default="results")
    parser.add_argument("--output-dir")
    parser.add_argument("--format", choices=("svg", "png"), default="svg")
    args = parser.parse_args()
    for path in generate_figures(args.results_dir, args.output_dir, args.format):
        print(path)


if __name__ == "__main__":
    main()
