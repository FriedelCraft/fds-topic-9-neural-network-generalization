# MAC 312 Project 9 · Neural-network capacity and generalization

A NumPy implementation of logistic regression and a one-hidden-layer neural network, evaluated on noisy XOR and the Wisconsin Diagnostic Breast Cancer dataset. The [report](REPORT.md) contains the derivation, protocol, results, and limitations. The [team review map](TEAM.md) points all six members to their completed workstream.

## Reproduce the results

Use Python 3.10 or newer. From the repository root, in an activated virtual environment:

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
python -m topic9.experiments --output results
python -m topic9.figures results
```

For an end-to-end check that writes to a separate directory:

```sh
python -m topic9.experiments --quick --output smoke-results
python -m topic9.figures smoke-results
```

The full run covers 90 experiment conditions with 15 paired repetitions per condition. The same fit is reused where different comparisons call for identical settings. Figures and raw results in `results/` were generated from the full run. `runs.csv` contains selected-model scores, compute, and stop reasons. `curves.csv` contains the regularization training and validation trajectories. `tuning.csv` records every weight-decay candidate and its validation score. `protocol.json` records seeds, search settings, and numerical gradient checks. See [dataset provenance](data/README.md) for the real dataset citation and license.
