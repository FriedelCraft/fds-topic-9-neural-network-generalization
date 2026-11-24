"""Build the standalone project notebook.

The notebook embeds the source of the ``topic9`` package as ``%%writefile`` cells, so it
runs from any directory without the repository. Regenerate it after changing ``src/``:

    python notebooks/build_notebook.py

This writes a clean notebook without outputs. The committed copy was then executed once in a
fresh directory outside the repository (jupyter nbconvert --execute) so that it ships with results.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("__init__", "data", "models", "training", "experiments", "capacity", "random_labels",
           "figures", "benchmark", "sensitivity")

cells = []


def md(text: str) -> None:
    cells.append(new_markdown_cell(dedent(text).strip()))


def code(text: str) -> None:
    cells.append(new_code_cell(dedent(text).strip()))


md(r"""
# Neural-Network Capacity and Generalization

MAC F312 - Foundations of Data Science · BITS Pilani, K.K. Birla Goa Campus

| Name | ID | Contribution |
|---|---|---|
| Yash Saxena | 2024ADPS0868G | Data and protocol: noisy XOR and WDBC loaders, paired nested training subsets, training-only standardization, metrics and seeds. |
| Raghav Singhal | 2024ADPS0827G | One-hidden-layer network and loss, derivation and NumPy implementation of backpropagation, finite-difference gradient checks. |
| Devam Bhavsar | 2024ADPS0831G | Logistic-regression baseline, paired sweeps over hidden width and training size on both datasets, paired comparison with the baseline. |
| Arnav Gupta | 2024ADPS0843G | Unregularized, weight-decay and early-stopping experiments with validation-based selection, and the review of selection and of paired accuracy and loss. |
| Rishi Motwani | 2024ADPS0838G | Random-label experiments: separate label streams for train, validation and test, matched clean and random runs, label-independence audit. |
| Mehta Prem | 2024ADPS0851G | Compute-budget comparison and the three controlled diagnoses, runtime benchmark, convergence study, the report and this notebook. |

This notebook runs the project end to end and does not need the repository. It writes the project's `topic9` package
from the cells below, runs each stage and draws the figures. The project is still in progress, so the notebook
covers the work completed so far.

| Stage | What it does |
|---|---|
| 1. Data | Noisy XOR and WDBC, nested training subsets, training-only standardization |
| 2. Model and gradients | One-hidden-layer tanh network, logistic baseline, manual backpropagation, finite-difference check |
| 3. Capacity | Hidden width and training-set size varied independently over repeated, paired runs |
| 4. Regularization | No regularization, weight decay and early stopping, with validation-only selection |
| 5. Random labels | Independently randomized labels to measure memorization |
| 6. Compute budgets and diagnoses | Equal estimated compute against a capped convergence rule, plus three controlled diagnostics |
| 7. Extension | Measured runtime against estimated work, and a convergence study on noisy XOR |

You need Python 3.10 or newer with `numpy`, `scikit-learn`, `matplotlib`, `threadpoolctl` and Jupyter. The WDBC
data ship with scikit-learn, so nothing is downloaded.

Set `MODE = "full"` to reproduce the archived protocol (15 paired runs per condition), which takes roughly 10 minutes
on a laptop. `MODE = "quick"` uses one seed pair and at most 40 updates and finishes in about a minute. It is only a
smoke test, and its numbers are not results.
""")

md("## 0. Setup")

code(r"""
# Uncomment to install the dependencies in the notebook's environment.
# %pip install numpy scikit-learn matplotlib threadpoolctl
""")

code(r"""
MODE = "full"          # "full" reproduces the archived protocol; "quick" is a smoke test
WORKDIR = "topic9_nb"  # created next to this notebook; holds the package source and all results

assert MODE in ("full", "quick")
QUICK = MODE == "quick"
""")

code(r"""
import os
from pathlib import Path

for sub in ("src/topic9", "results"):
    Path(WORKDIR, sub).mkdir(parents=True, exist_ok=True)
print("Working directory:", WORKDIR)
""")

md(r"""
### Package source

The next cells write the `topic9` modules to `topic9_nb/src/topic9/`. They are the same files as in the
repository (`src/topic9/`), included so the notebook is self-contained.
""")

for name in MODULES:
    source = (ROOT / "src" / "topic9" / f"{name}.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    cells.append(new_code_cell(f"%%writefile topic9_nb/src/topic9/{name}.py\n{source.rstrip()}"))

code(r"""
import sys
import importlib

src = str(Path(WORKDIR, "src").resolve())
sys.path.insert(0, src)
for name in [m for m in sys.modules if m == "topic9" or m.startswith("topic9.")]:
    del sys.modules[name]          # make sure the package written above is the one imported
importlib.invalidate_caches()

import csv
import contextlib
import io
import json
import statistics as st

import numpy as np
from IPython.display import Image, Markdown, display
from threadpoolctl import threadpool_limits

from topic9 import benchmark, capacity, experiments, figures, random_labels, sensitivity
from topic9.data import make_dataset, prepare_split
from topic9.models import gradient_check, initialize, metrics

import shutil

RESULTS = Path(WORKDIR, "results")
PNG = RESULTS / "figures_png"
shutil.rmtree(PNG, ignore_errors=True)   # never reuse figures from an earlier run
print("topic9 loaded from:", Path(sys.modules["topic9"].__file__).parent.name, "| mode:", MODE)
""")

code(r"""
# Helpers for reading results and printing tables.
def load(name):
    rows = list(csv.DictReader(open(RESULTS / name, encoding="utf-8")))
    for row in rows:
        for key, value in row.items():
            if value == "":
                continue
            try:
                row[key] = float(value) if ("." in value or "e" in value.lower()) else int(value)
            except ValueError:
                pass
    return rows


def select(rows, **criteria):
    return [r for r in rows if all(r.get(k) == v for k, v in criteria.items())]


def mean(rows, metric):
    return st.mean(r[metric] for r in rows if r[metric] != "")


def table(headers, body):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in body]
    display(Markdown("\n".join(lines)))


def show_figures(*names):
    if not all((PNG / f"{name}.png").exists() for name in names):
        figures.generate_figures(RESULTS, PNG, "png")   # draws every figure whose CSV exists
    for name in names:
        display(Image(filename=str(PNG / f"{name}.png")))


def quiet(function, *args, **kwargs):
    # Experiment runners print one line per condition. Keep only a summary in the notebook.
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), threadpool_limits(limits=1, user_api="blas"):
        function(*args, **kwargs)
    return sink.getvalue().strip().splitlines()[-1]
""")

md(r"""
## 1. Data and protocol

Both datasets follow one protocol.

* Noisy XOR has two uniform features in $[-1,1]$. The label is 1 when the signs differ, and each label is flipped
  independently with probability 0.10, so the Bayes accuracy is 90%.
* WDBC has all 569 Wisconsin Diagnostic Breast Cancer examples with 30 numeric features (malignant = 1), taken from the
  copy bundled with scikit-learn. The data are by Wolberg, Mangasarian, Street and Street (1993), UCI Machine Learning
  Repository, [doi:10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B), CC BY 4.0.

Each dataset has a training reservoir and fixed validation and test pools. A *data seed* permutes the reservoir, and
training sizes 32, 96 and 256 are nested prefixes of that permutation. Features are standardized with the training
subset's own means and standard deviations only.
""")

code(r"""
rows = []
for name in experiments.DATASETS:
    data = make_dataset(name)
    split = prepare_split(data, 96, experiments.DATA_SEEDS[0])
    rows.append([name, len(data.y_train_pool), len(data.y_val), len(data.y_test), data.x_val.shape[1],
                 f"{split.x_train.mean():+.1e}", f"{split.x_train.std():.3f}",
                 f"{split.x_val.mean():+.3f}"])
table(["Dataset", "Train reservoir", "Validation", "Test", "Features",
       "Mean of train features (n=96)", "SD of train features", "Mean of val features"], rows)
""")

code(r"""
# Training-only scaling check: the training subset is centered and has unit spread by construction,
# while the held-out pools are transformed with the training statistics and are not.
import matplotlib.pyplot as plt

xor = make_dataset("synthetic")
fig, ax = plt.subplots(figsize=(4.6, 4.2))
sample = slice(0, 400)
ax.scatter(xor.x_train_pool[sample, 0], xor.x_train_pool[sample, 1], c=xor.y_train_pool[sample],
           cmap="coolwarm", s=14, alpha=0.8)
ax.set_title("Noisy XOR training candidates (first 400)")
ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
# topic9.figures selects the non-interactive Agg backend, so render to a buffer and display the image.
buffer = io.BytesIO()
fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
plt.close(fig)
display(Image(data=buffer.getvalue()))
""")

md(r"""
## 2. Model, loss and gradient check

With standardized features $X$, labels $y\in\{0,1\}^n$ and hidden width $m$:

$$A=XW_1+\mathbf 1b_1^\top,\quad H=\tanh A,\quad z=Hw_2+b_2\mathbf 1,\quad
J=\frac1n\sum_i\operatorname{softplus}\big((1-2y_i)z_i\big)+\frac\lambda2\sum_W\lVert W\rVert_F^2 .$$

Backpropagation with $\delta=(\sigma(z)-y)/n$ and $G=(\delta w_2^\top)\odot(1-H\odot H)$ gives

$$\nabla_{w_2}J=H^\top\delta+\lambda w_2,\quad \nabla_{b_2}J=\mathbf 1^\top\delta,\quad
\nabla_{W_1}J=X^\top G+\lambda W_1,\quad \nabla_{b_1}J=G^\top\mathbf 1 .$$

The logistic baseline is the same model without the hidden layer ($z=Xw+b\mathbf 1$). Training is full-batch gradient
descent with learning rate 0.2. The cell below compares every analytic gradient with central finite differences on a
small problem, with and without weight decay.
""")

code(r"""
checks = experiments.verify_gradients()      # raises if any parameter's error exceeds 1e-7
table(["Model", "Parameters checked", "Max abs. error", "Max rel. error"],
      [[name, c["parameters_checked"], f"{c['max_abs_error']:.2e}", f"{c['max_relative_error']:.2e}"]
       for name, c in checks.items()])

# An independent seed, as the review map suggests.
rng = np.random.default_rng(2024)
x = rng.normal(size=(9, 4)); y = rng.integers(0, 2, size=9).astype(float)
worst = max(gradient_check(initialize(4, width, seed), x, y, weight_decay=decay)["max_abs_error"]
            for width in (0, 3, 6) for decay in (0.0, 0.03) for seed in (5, 6))
print(f"Independent check (new data, widths 0/3/6, two initializations): worst absolute error {worst:.2e}")
""")

md(r"""
## 3. Run the main experiments

One call runs all 91 conditions: the capacity sweep, the three regularization methods, the random-label
controls, both budget rules and the diagnostics. Related experiments reuse the same examples and initialization
seeds. Results are written to `topic9_nb/results/`.
""")

code(r"""
summary = quiet(experiments.run, RESULTS, quick=QUICK)
print(summary)
runs = load("runs.csv")
""")

md(r"""
## 4. Capacity and training-set size

Hidden width (0 = logistic regression, 2, 8, 32) and training size (32, 96, 256) are varied independently. Each cell
is the mean over the paired runs. Error bars in the figure show one standard deviation across data seeds.
""")

code(r"""
cap = select(runs, experiment="capacity")
body = []
for dataset in experiments.DATASETS:
    for size in experiments.TRAIN_SIZES:
        cells_ = []
        for width in experiments.WIDTHS:
            group = select(cap, dataset=dataset, train_size=size, width=width)
            cells_.append(f"{100*mean(group,'test_accuracy'):.1f}% / {mean(group,'test_loss'):.3f}")
        body.append([dataset, size, *cells_])
table(["Dataset", "n", "Logistic", "Width 2", "Width 8", "Width 32"], body)
show_figures("capacity")
""")

md(r"""
### Paired comparison with logistic regression

Each network run is paired with the logistic run that has the same data seed and initialization seed. The
differences are averaged within each data seed first, so the spread below is the sample SD across the five data
seeds. The 15 runs share data seeds and are not independent.
""")

code(r"""
pairs = list(capacity.summarize(RESULTS))
sd = lambda v: f"{float(v):.2f}" if v != "" else "n/a"
table(["Dataset", "n", "Width", "Accuracy change vs logistic (pp)", "Loss change vs logistic",
       "Data seeds with higher accuracy", "Data seeds with lower loss"],
      [[p["dataset"], p["train_size"], p["width"],
        f"{float(p['test_accuracy_delta_pp_mean']):+.1f} ± {sd(p['test_accuracy_delta_pp_sd'])}",
        f"{float(p['test_loss_delta_mean']):+.3f} ± {float(p['test_loss_delta_sd'] or 0):.3f}",
        f"{p['data_seeds_with_higher_accuracy']} / {p['n_data_seeds']}",
        f"{p['data_seeds_with_lower_loss']} / {p['n_data_seeds']}"] for p in pairs])
""")

md(r"""
## 5. Regularization

At width 32 the notebook compares an unregularized fit, weight decay ($\lambda\in\{5\cdot10^{-4},5\cdot10^{-3},0.05\}$ chosen by
validation cross-entropy) and early stopping (patience 35, restoring the best validation checkpoint). Test scores are
computed only after selection.
""")

code(r"""
reg = select(runs, experiment="regularization")
body = []
for dataset in experiments.DATASETS:
    for size in (32, 256):
        row = [dataset, size]
        for method in ("none", "weight_decay", "early_stopping"):
            group = select(reg, dataset=dataset, train_size=size, regularization=method)
            row.append(f"{mean(group,'test_loss'):.3f} ({100*mean(group,'test_accuracy'):.1f}%)")
        body.append(row)
table(["Dataset", "n", "No regularization", "Weight decay", "Early stopping"], body)

tuning = load("tuning.csv")
chosen = {}
for r in select(tuning, selected=1):
    chosen.setdefault((r["dataset"], r["train_size"]), []).append(r["decay"])
table(["Dataset", "n", "Selected weight decay (all runs)"],
      [[d, n, ", ".join(f"{v:g}" for v in sorted(set(vals)))] for (d, n), vals in sorted(chosen.items())])
# Where early stopping restored its checkpoint (update 0 is the initial model).
display(Markdown("#### Early stopping on noisy XOR: restored checkpoint and accuracy change against the matched unregularized fit"))
base = {(r["dataset"], r["train_size"], r["data_seed"], r["init_seed"]): r
        for r in reg if r["regularization"] == "none"}
body = []
for size in (32, 256):
    es = [r for r in select(reg, dataset="synthetic", train_size=size, regularization="early_stopping")]
    change = lambda r: 100 * (r["test_accuracy"] - base[(r["dataset"], r["train_size"], r["data_seed"], r["init_seed"])]["test_accuracy"])
    for label, group in (("restored update <= 9", [r for r in es if r["selected_step"] <= 9]),
                         ("restored update > 9", [r for r in es if r["selected_step"] > 9])):
        body.append([size, label, len(group),
                     f"{st.mean(r['selected_step'] for r in group):.0f}" if group else "-",
                     f"{st.mean(change(r) for r in group):+.1f} pp" if group else "-"])
table(["n", "Group", "Runs", "Mean restored update", "Mean test-accuracy change"], body)
show_figures("regularization")
""")

md(r"""
## 6. Random-label control

Training, validation and test labels are redrawn independently from Bernoulli(0.5) with separate random streams,
keeping features, sample membership and initial weights fixed. Held-out accuracy must be near 50%, so any gap
between training and held-out accuracy is memorization of the training labels.
""")

code(r"""
rl = select(runs, experiment="random_labels")
body = []
for dataset in experiments.DATASETS:
    for size in (32, 96):
        for width in (0, 32):
            clean = select(rl, dataset=dataset, train_size=size, width=width, labels="clean")
            rand = select(rl, dataset=dataset, train_size=size, width=width, labels="random")
            body.append([dataset, size, "Logistic" if width == 0 else f"Width {width}",
                         f"{100*mean(clean,'train_accuracy'):.1f}% / {100*mean(clean,'test_accuracy'):.1f}%",
                         f"{100*mean(rand,'train_accuracy'):.1f}% / {100*mean(rand,'test_accuracy'):.1f}%"])
table(["Dataset", "n", "Model", "Clean labels: train / test", "Random labels: train / test"], body)
show_figures("random_labels")
""")

md(r"""
### Chance check and label audit

Because the test labels are independent of the model, the number of correct test predictions is exactly
Binomial($n_\text{test}$, 0.5). The table counts how many random-label runs fall inside the central 95% interval
and how many fit every training label. The audit checks that the drawn labels are unrelated to the features: the largest
absolute feature-label correlation of each draw should stay below a Bonferroni-style threshold.

One width-2 WDBC random-label fit (n = 96) stops almost exactly at the gradient threshold, so its result can differ slightly between platforms. The archived results in the repository have 229 of 240 runs inside the interval, and a rerun on another machine can give 230.
""")

code(r"""
chance = [r for r in random_labels.summarize(RESULTS) if r["labels"] == "random"]
table(["Dataset", "n", "Model", "Train acc.", "Test acc.", "Runs fitting all train labels",
       "95% chance interval", "Runs inside interval"],
      [[r["dataset"], r["train_size"], "Logistic" if r["width"] == 0 else f"Width {r['width']}",
        f"{r['train_accuracy_mean']}%", f"{r['test_accuracy_mean']}%",
        f"{r['runs_fitting_all_train_labels']} / {r['n_runs']}", r["chance_interval"],
        f"{r['runs_inside_chance_interval']} / {r['n_runs']}"] for r in chance])
inside = sum(r["runs_inside_chance_interval"] for r in chance)
print(f"Inside the chance interval: {inside} of {sum(r['n_runs'] for r in chance)} random-label runs")

audit = list(random_labels.audit(json.load(open(RESULTS / "protocol.json"))["data_seeds"], 96))
ratio = lambda a: float(a["max_abs_feature_correlation"]) / float(a["correlation_bound"])
below = sum(ratio(a) < 1 for a in audit)
closest = max(audit, key=ratio)
print(f"Label audit: {below} of {len(audit)} draws have every feature correlation below the threshold; closest is "
      f"{closest['max_abs_feature_correlation']} against {closest['correlation_bound']} "
      f"({closest['dataset']}, {closest['split']}, seed {closest['data_seed']})")
""")

md(r"""
## 7. Compute budgets and controlled diagnoses

One checkpoint is a training forward and gradient pass plus a validation forward pass. It costs $(2n+n_v)d$ affine
multiply-accumulates for logistic regression and $(2n+n_v)\,m\,(d+1)$ for the network.

At $n=96$ the notebook compares two budget rules.

1. Common compute: every model gets the largest whole number of updates that fits under the work of 181 width-8
   checkpoints.
2. Gradient rule with cap: train until the gradient norm of the penalized training objective is at most 0.005, or for
   800 updates.

The second rule usually stops at its cap for the networks, so the table also records how each run stopped.
""")

code(r"""
bud = select(runs, experiment="budget")
body = []
for dataset in experiments.DATASETS:
    for width in experiments.WIDTHS:
        row = [dataset, "Logistic" if width == 0 else f"Width {width}"]
        for rule in ("fixed", "convergence"):
            g = select(bud, dataset=dataset, width=width, budget=rule)
            stops = {s: sum(r["stop_reason"] == s for r in g) for s in ("converged", "budget")}
            row.append(f"{mean(g,'steps'):.0f} updates, {100*mean(g,'test_accuracy'):.1f}% / {mean(g,'test_loss'):.3f}")
            if rule == "convergence":
                row.append(f"{stops['converged']} / {len(g)}")
        body.append(row)
table(["Dataset", "Model", "Common compute", "Gradient rule with cap", "Converged runs"], body)
show_figures("budget")
""")

md(r"""
### Three controlled diagnoses

Each diagnosis changes one thing and keeps the data, seeds and learning rate fixed.

* Optimization (XOR, width 32): 8 updates against 800, then 800 against 3,000.
* Representation (XOR): the linear model against the width-32 network.
* Overfitting (WDBC, $n=32$): the unregularized terminal iterate against the validation-selected checkpoint.

Validation loss is the selection criterion in the overfitting comparison, so only the change in test loss counts as
evidence there.
""")

code(r"""
diag = select(runs, experiment="diagnostic")
pairs = [("Optimization: 8 -> 800 updates", "optimization_short", "optimization_long"),
         ("Optimization: 800 -> 3,000 updates", "optimization_long", "optimization_extended"),
         ("Representation: linear -> width 32", "representation_linear", "representation_wide"),
         ("Overfitting: terminal -> validation-selected", "overfit_unregularized", "overfit_early_stopping")]
body = []
for label, before, after in pairs:
    a, b = select(diag, condition=before), select(diag, condition=after)
    body.append([label] + [f"{mean(a,m):.3f} -> {mean(b,m):.3f}" for m in ("train_loss", "val_loss", "test_loss")]
                + [f"{sum(r['stop_reason']=='converged' for r in b)} / {len(b)}"])
table(["Change", "Train loss", "Validation loss", "Test loss", "Converged after change"], body)
show_figures("diagnostic")
""")

md(r"""
## 8. Extension: runtime against estimated work

The estimated-work count is architecture sensitive but is not elapsed time. This stage times the 16 budget-rule
conditions (both datasets, $n=96$, all four models) on the same matched seeds. Each run is timed repeatedly after an untimed
pass, conditions are interleaved, garbage collection is paused, and BLAS uses one thread. The timed region is the
whole `train` call.
""")

code(r"""
print(quiet(benchmark.run, RESULTS, quick=QUICK))
rt = load("runtime_summary.csv")
table(["Dataset", "Model", "Rule", "Updates", "Estimated work", "Median seconds", "µs / checkpoint"],
      [[r["dataset"], "Logistic" if r["width"] == 0 else f"Width {r['width']}",
        "common compute" if r["budget"] == "fixed" else "gradient rule",
        f"{r['mean_updates']:.0f}", f"{r['mean_work']:.3g}", f"{r['median_seconds']:.4f}",
        f"{r['microseconds_per_checkpoint']:.0f}"] for r in rt])
show_figures("runtime")
""")

code(r"""
meta = json.load(open(RESULTS / "runtime.json", encoding="utf-8"))
fits, models = meta["fits"], meta["cost_models"]
print(f"Log-log fit of time on work across conditions: slope {fits['pooled']['slope']:.2f}, "
      f"R^2 {fits['pooled']['r_squared']:.2f}")
table(["Runtime model", "Median relative error", "Max relative error"],
      [[name.replace("_", " "), f"{m['median_relative_error']:.0%}", f"{m['max_relative_error']:.0%}"]
       for name, m in models.items()])

# Runtime per work, compared across models at (nearly) equal estimated work.
fixed = [r for r in rt if r["budget"] == "fixed"]
for dataset in experiments.DATASETS:
    times = {r["width"]: r["median_seconds"] for r in fixed if r["dataset"] == dataset}
    print(f"{dataset}: at equal estimated work, time ranges {min(times.values()):.4f}-{max(times.values()):.4f} s "
          f"(x{max(times.values())/min(times.values()):.1f})")
""")

md(r"""
## 9. Extension: convergence sensitivity on noisy XOR

With 256 training examples and widths 8 and 32, the update cap (100 to 3,200 at learning rate 0.2) and the learning
rate (0.05 to 1.6 at cap 800) are varied separately. Every run records the train, validation and test loss, the final
gradient norm, any increase in training loss between updates, and the reason it stopped.
""")

code(r"""
print(quiet(sensitivity.run, RESULTS, quick=QUICK))
conv = load("convergence.csv")


def row_for(sweep, width, field, level):
    g = [r for r in select(conv, sweep=sweep, width=width) if r[field] == level]
    ok = sum(r["stop_reason"] == "converged" for r in g)
    return [f"{level:g}", f"{ok} / {len(g)}", f"{mean(g,'steps'):.0f}", f"{mean(g,'train_loss'):.3f}",
            f"{mean(g,'test_loss'):.3f}", f"{mean(g,'final_gradient_norm'):.4f}",
            sum(1 for r in g if r["loss_increases"] > 0)]


heads = ["Setting", "Converged", "Mean updates", "Train loss", "Test loss", "Final grad. norm", "Runs with loss increases"]
for width in sensitivity.WIDTHS:
    display(Markdown(f"#### Width {width}: update cap (learning rate 0.2)"))
    table(heads, [row_for("cap", width, "max_steps", c) for c in sensitivity.CAPS if select(conv, sweep="cap", width=width, max_steps=c)])
    display(Markdown(f"#### Width {width}: learning rate (cap 800)"))
    table(heads, [row_for("learning_rate", width, "learning_rate", lr) for lr in sensitivity.LEARNING_RATES
                  if select(conv, sweep="learning_rate", width=width, learning_rate=lr)])
show_figures("convergence")
""")

md(r"""
## 10. Summary of the results so far

The numbers below are read from this run, so they match the tables above.
""")

code(r"""
def acc_loss(rows, **c):
    g = select(rows, **c)
    return f"{100*mean(g,'test_accuracy'):.1f}% / {mean(g,'test_loss'):.3f}"

ok = lambda sweep, w, **c: sum(r["stop_reason"] == "converged" for r in select(conv, sweep=sweep, width=w, **c))

display(Markdown(f'''
* Capacity: On noisy XOR with n = 256, the width-32 network reaches {acc_loss(cap, dataset="synthetic", train_size=256, width=32)}
  (test accuracy / cross-entropy) and logistic regression {acc_loss(cap, dataset="synthetic", train_size=256, width=0)}. On WDBC at n = 256
  logistic regression reaches {acc_loss(cap, dataset="wdbc", train_size=256, width=0)} and the width-32 network {acc_loss(cap, dataset="wdbc", train_size=256, width=32)}.
* Regularization: Early-stopping test loss on WDBC at n = 32 is
  {mean(select(reg, dataset="wdbc", train_size=32, regularization="early_stopping"), "test_loss"):.3f}, against
  {mean(select(reg, dataset="wdbc", train_size=32, regularization="none"), "test_loss"):.3f} without regularization.
* Random labels: On WDBC with n = 32 the width-32 network reaches
  {100*mean(select(rl, dataset="wdbc", train_size=32, width=32, labels="random"), "train_accuracy"):.1f}% training accuracy and
  {100*mean(select(rl, dataset="wdbc", train_size=32, width=32, labels="random"), "test_accuracy"):.1f}% held-out accuracy on random labels.
* Compute: On XOR, width 32 gets {acc_loss(bud, dataset="synthetic", width=32, budget="fixed")} under the common budget and
  {acc_loss(bud, dataset="synthetic", width=32, budget="convergence")} under the capped gradient rule, which stopped at its cap in
  {sum(r["stop_reason"] == "budget" for r in select(bud, dataset="synthetic", width=32, budget="convergence"))} of
  {len(select(bud, dataset="synthetic", width=32, budget="convergence"))} runs.
* Runtime: Across the 16 conditions, estimated work alone explains R² = {fits["pooled"]["r_squared"]:.2f} of the variation
  in log runtime, so equal estimated work does not mean equal elapsed time.
* Convergence: At the baseline (learning rate 0.2, cap 800), {ok("cap", 32, max_steps=800)} of
  {len(select(conv, sweep="cap", width=32, max_steps=800))} width-32 runs meet the 0.005 gradient threshold.
'''))
""")

md(r"""
## 11. Status and open items

Done so far: the data and protocol, the manually derived and finite-difference-checked network, the capacity sweep,
the three regularization methods, the random-label control, both compute-budget rules with three controlled diagnoses,
the runtime benchmark and the convergence study.

Still open:

* Each workstream owner still has to review their part (data and splits, gradient trace, capacity matches,
  validation-only selection, random-label independence), as listed in the repository's `TEAM.md`.
* The final report is a separate LaTeX document in the repository. This notebook covers only the experiments and figures.
* The timings come from one machine, and the conclusions hold only for this protocol: two datasets, full-batch gradient
  descent at the stated learning rates, and the tested widths and sample sizes.
""")

md(r"""
## References

1. D. E. Rumelhart, G. E. Hinton and R. J. Williams. Learning representations by back-propagating errors. *Nature*, 323(6088):533-536, 1986.
2. X. Glorot and Y. Bengio. Understanding the difficulty of training deep feedforward neural networks. In *Proceedings of the 13th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 9:249-256, 2010.
3. A. Krogh and J. A. Hertz. A simple weight decay can improve generalization. In *Advances in Neural Information Processing Systems 4*, pages 950-957, 1992.
4. L. Prechelt. Early stopping, but when? In *Neural Networks: Tricks of the Trade*, Lecture Notes in Computer Science 1524, pages 55-69. Springer, 1998.
5. C. Zhang, S. Bengio, M. Hardt, B. Recht and O. Vinyals. Understanding deep learning requires rethinking generalization. In *International Conference on Learning Representations (ICLR)*, 2017.
6. W. Wolberg, O. Mangasarian, N. Street and W. Street. Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository, 1993. [doi:10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B).
7. F. Pedregosa et al. Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12:2825-2830, 2011.
8. C. R. Harris et al. Array programming with NumPy. *Nature*, 585:357-362, 2020.
9. J. D. Hunter. Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3):90-95, 2007.
""")

notebook = new_notebook(cells=cells)
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

if __name__ == "__main__":
    output = Path(__file__).with_name("topic9_project.ipynb")
    nbformat.write(notebook, output)
    print(f"Wrote {output.name} with {len(cells)} cells.")
