# Team review map

Each workstream has a runnable implementation and results. Start in your files, rerun the relevant checks, and use a focused pull request for improvements. Keep the original evaluation protocol reproducible if you add an experiment.

| Reviewer | Implemented workstream | Start here | Suggested review |
|---|---|---|---|
| FriedelCraft | Datasets, nested splits, preprocessing, protocol | [data.py](src/topic9/data.py), [data tests](tests/test_data.py), [dataset provenance](data/README.md) | Check the split sizes and how the scaler is fitted. |
| RaghavSinghal15 | MLP, logistic model primitives, manual backpropagation, finite differences | [models.py](src/topic9/models.py), [model tests](tests/test_models.py), [numerical checks](results/protocol.json) | Trace one gradient through the equations and verify an independent seed. |
| devam2006 | Linear comparison and width × training-size sweep | [experiment design](src/topic9/experiments.py), [capacity figure](results/figures/capacity.svg), [paired comparison](CAPACITY.md) | Check comparisons within each width and size with the data and initialization seeds matched. |
| yllow-lan | Unregularized model, weight decay, and early stopping | [training.py](src/topic9/training.py), [regularization figure](results/figures/regularization.svg), [tuning results](results/tuning.csv) | Check the selected penalties and checkpoints against validation loss, then compare test accuracy and cross-entropy in the [matched runs](results/runs.csv). |
| kumarrishimotwani-byte | Independently randomized labels and memorization | [data.py](src/topic9/data.py), [random-label figure](results/figures/random_labels.svg), [run results](results/runs.csv), [label audit and chance check](RANDOM_LABELS.md) | Trace the random-label streams, then compare train and test accuracy. |
| Prem-Mehta03 | Equal compute, convergence, diagnostics, and report | [experiments.py](src/topic9/experiments.py), [compute figure](results/figures/budget.svg), [diagnostic figure](results/figures/diagnostic.svg), [report](REPORT.md) | Own the runtime benchmark, convergence sensitivity study, and report extension below. |

[Download the team handoff PDF](docs/Topic-9-Team-Handoff.pdf).

## Prem-Mehta03 — Runtime and convergence extension

The project already compares estimated affine work and a capped gradient rule. Prem's remaining work is to measure how closely the estimate tracks actual runtime and test whether the conclusions depend on stopping settings.

1. **Build a reproducible runtime benchmark.** Compare logistic regression and hidden widths 2, 8, and 32 at 96 training examples on both datasets. Reuse matched data and initialization seeds under both budget rules. Repeat timings at least three times per configuration on one machine with one BLAS thread, after a warm-up. Time the training loop, including its validation evaluations, and record the machine, Python/BLAS versions, updates, and affine work. Summarize median elapsed time and timing spread. Check whether equal estimated work produces comparable elapsed time or changes the practical ranking.
2. **Test sensitivity to the optimizer and cap.** On noisy XOR at 256 training examples, repeat the matched width-32 runs starting from the existing settings. Compare the 800-update cap with a larger cap at the same learning rate. Separately vary the learning rate at a fixed cap. Set the comparison grid before inspecting test results. Record train and validation loss, gradient norms, stop reasons, and held-out scores. Explain which results support an optimization limit and which remain unresolved.
3. **Write up the new evidence.** Submit a focused pull request with runnable benchmark code, a results CSV, a runtime-versus-work figure, and a short report subsection interpreting the timing and sensitivity results. Include the exact reproduction command and keep the existing results available for comparison.
