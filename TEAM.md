# Team review map

Each workstream has a runnable implementation and results. Start in your files, rerun the relevant checks, and use a focused pull request for improvements. Keep the original evaluation protocol reproducible if you add an experiment.

| Reviewer | Implemented workstream | Start here | Suggested review |
|---|---|---|---|
| FriedelCraft | Datasets, nested splits, preprocessing, protocol | [data.py](src/topic9/data.py), [data tests](tests/test_data.py), [dataset provenance](data/README.md) | Check the split sizes and how the scaler is fitted. |
| RaghavSinghal15 | MLP, logistic model primitives, manual backpropagation, finite differences | [models.py](src/topic9/models.py), [model tests](tests/test_models.py), [numerical checks](results/protocol.json) | Trace one gradient through the equations and verify an independent seed. |
| devam2006 | Linear comparison and width × training-size sweep | [experiment design](src/topic9/experiments.py), [capacity figure](results/figures/capacity.svg), [paired comparison](CAPACITY.md) | Check comparisons within each width and size with the data and initialization seeds matched. |
| yllow-lan | Unregularized model, weight decay, and early stopping | [training.py](src/topic9/training.py), [regularization figure](results/figures/regularization.svg), [tuning results](results/tuning.csv) | Check the selected penalties and checkpoints against validation loss, then compare test accuracy and cross-entropy in the [matched runs](results/runs.csv). |
| kumarrishimotwani-byte | Independently randomized labels and memorization | [data.py](src/topic9/data.py), [random-label figure](results/figures/random_labels.svg), [run results](results/runs.csv), [label audit and chance check](RANDOM_LABELS.md) | Trace the random-label streams, then compare train and test accuracy. |
| Prem-Mehta03 | Equal compute, convergence, diagnostics, and report | [experiments.py](src/topic9/experiments.py), [compute figure](results/figures/budget.svg), [diagnostic figure](results/figures/diagnostic.svg), [report](report/report.tex) | Done: runtime benchmark, convergence sensitivity study, and report extension (see below). |

[Download the team handoff PDF](docs/Topic-9-Team-Handoff.pdf).

## Prem-Mehta03 — Runtime and convergence extension

The project already compared estimated affine work and a capped gradient rule. This extension measured how closely the estimate tracks actual runtime and tested whether the conclusions depend on stopping settings. All three parts are complete and merged.

1. **Runtime benchmark (done).** [benchmark.py](src/topic9/benchmark.py) times logistic regression and hidden widths 2, 8, and 32 at 96 training examples on both datasets under both budget rules, using the matched data and initialization seeds. Each of the 15 matched runs is timed five times after a warm-up, on one machine with one BLAS thread, and the machine and software versions are recorded. Results are in [runtime.csv](results/runtime.csv), [runtime_summary.csv](results/runtime_summary.csv), and [runtime.json](results/runtime.json). Equal estimated work does not give comparable elapsed time across widths.
2. **Sensitivity to the optimizer and cap (done).** [sensitivity.py](src/topic9/sensitivity.py) varies the update cap (100 to 3,200 at learning rate 0.2) and, separately, the learning rate (0.05 to 1.6 at cap 800) on noisy XOR with 256 training examples at widths 8 and 32. It records train, validation, and test loss, gradient norms, and stop reasons in [convergence.csv](results/convergence.csv). The 800-update baseline gives the lowest mean test loss but is not converged.
3. **Write-up (done).** The code, results CSVs, the [runtime figure](results/figures/runtime.svg), the [convergence figure](results/figures/convergence.svg), and the report subsections were submitted in a pull request. The reproduction commands are in the [README](README.md) and at the end of the [report](report/report.tex).
