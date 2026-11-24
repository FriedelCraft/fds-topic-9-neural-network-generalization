# Team review map

Each workstream has a runnable implementation and results. Start in your files, rerun the relevant checks, and use a focused pull request for improvements. Keep the original evaluation protocol reproducible if you add an experiment.

| Reviewer | Implemented workstream | Start here | Suggested review |
|---|---|---|---|
| FriedelCraft | Datasets, nested splits, preprocessing, protocol | [data.py](src/topic9/data.py), [data tests](tests/test_data.py), [dataset provenance](data/README.md) | Check the split sizes and how the scaler is fitted. |
| RaghavSinghal15 | MLP, logistic model primitives, manual backpropagation, finite differences | [models.py](src/topic9/models.py), [model tests](tests/test_models.py), [numerical checks](results/protocol.json) | Trace one gradient through the equations and verify an independent seed. |
| devam2006 | Linear comparison and width × training-size sweep | [experiment design](src/topic9/experiments.py), [capacity figure](results/figures/capacity.svg), [run results](results/runs.csv) | Check comparisons within each width and size with the data and initialization seeds matched. |
| yllow-lan | Unregularized model, weight decay, and early stopping | [training.py](src/topic9/training.py), [regularization figure](results/figures/regularization.svg), [tuning results](results/tuning.csv) | Verify validation-only selection and check where calibration or accuracy changes. |
| kumarrishimotwani-byte | Independently randomized labels and memorization | [data.py](src/topic9/data.py), [random-label figure](results/figures/random_labels.svg), [run results](results/runs.csv) | Trace the random-label streams, then compare train and test accuracy. |
| Prem-Mehta03 | Equal compute, convergence, diagnostics, and report | [experiments.py](src/topic9/experiments.py), [compute figure](results/figures/budget.svg), [diagnostic figure](results/figures/diagnostic.svg), [report](report/report.tex) | Check the compute accounting and whether each controlled change supports its diagnosis. |
