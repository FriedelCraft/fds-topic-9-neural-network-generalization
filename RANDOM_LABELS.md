# Random-label control: independence and chance

The random-label experiment asks whether a model that fits its training labels has learned the population relationship or has only memorized the assignments. It repeats the clean runs with the same features, training rows, and initial weights. Only the labels change.

## How the labels are drawn

For data seed $s$, the label seed is $10\,000+s$ (`random_label_seed` in [experiments.py](src/topic9/experiments.py)). `prepare_split` in [data.py](src/topic9/data.py) spawns three child streams from that seed with `SeedSequence.spawn(3)`. The first fills the whole training reservoir, the second the validation pool, and the third the test pool. Every label is an independent $\operatorname{Bernoulli}(0.5)$ draw.

1. **Independent of the inputs.** The draw never reads features or true labels. A test changes every feature and flips every true label, and the random labels stay identical.
2. **Independent across splits.** The three child streams are statistically independent. Tests confirm that the train, validation, and test vectors differ.
3. **Paired across models.** The label seed depends only on the data seed. All widths and both sample sizes therefore see the same labels. The $n=32$ labels are the first 32 of the $n=96$ labels.
4. **Pairing across datasets.** Label seeds 10301–10305 differ from the sampling, initialization and split seeds. Each label seed is reused on both datasets, so the 114 WDBC test labels match the first 114 XOR test labels for that seed. The validation streams also share a prefix.

**Consequence.** The fitted model is a function of the training features and training labels. The test labels are independent of both. So for any fitted model, the number of correct test predictions is exactly $\operatorname{Binomial}(n_{\text{test}},0.5)$. Chance is 50% on both datasets, even though the true WDBC labels are 37% malignant. A majority-class rule also scores 50% in expectation here.

## Empirical audit

[random_labels_audit.csv](results/random_labels_audit.csv) checks every draw. It covers 2 datasets × 5 data seeds × 3 splits = 30 label vectors, using the 96 training rows. For each draw it reports three things:

- the positive rate;
- agreement with the true labels;
- the largest absolute Pearson correlation between the labels and any standardized feature.

The correlations are compared with an approximate 95% threshold, $z_{1-0.025/d}/\sqrt n$, adjusted for the $d$ features in each draw. All 30 draws fall below their thresholds. The largest fraction of a threshold occurs for the synthetic training labels at seed 303 ($0.2075/0.2288$, about 91%). Agreement with the true labels ranges from 45.7% to 59.4%.

## Training fit versus chance

[random_labels_summary.csv](results/random_labels_summary.csv) reports every width at both sizes, for clean and random labels. Means and SDs are taken across the five data-seed means. The chance interval is the exact central 95% binomial interval:

- [47.9%, 52.1%] for the 2048 synthetic test points;
- [41.2%, 58.8%] for the 114 WDBC test points.

| Dataset | $n$ | Width | Random train acc. | Runs fitting every label | Random test acc. | Runs inside chance interval | Clean test acc. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Noisy XOR | 32 | 32 | 74.8% | 0/15 | 49.6% | 15/15 | 75.8% |
| Noisy XOR | 96 | 32 | 64.4% | 0/15 | 50.2% | 15/15 | 82.6% |
| WDBC | 32 | 0 | 92.3% | 3/15 | 48.5% | 13/15 | 94.6% |
| WDBC | 32 | 32 | 100.0% | 15/15 | 49.8% | 15/15 | 94.0% |
| WDBC | 96 | 0 | 72.6% | 0/15 | 49.1% | 15/15 | 95.4% |
| WDBC | 96 | 8 | 99.4% | 9/15 | 49.4% | 14/15 | 94.3% |
| WDBC | 96 | 32 | 100.0% | 15/15 | 49.2% | 15/15 | 94.6% |

Across all 16 random-label conditions, 229 of 240 runs (95.4%) fall inside the 95% chance interval. This matches the nominal rate. Runs within a condition share five test-label draws, so they are not independent, and the count is descriptive rather than a test.

**Interpretation.** On WDBC at width 32, all 15 runs fit every random training label at each size. Mean test accuracy stayed near 50%, leaving a generalization gap of about 50 percentage points. With the true labels, the same models reached 94–95% mean test accuracy and the gap was 5–6 points.

Capacity matters at $n=96$. Logistic regression fits only 72.6% of the random labels, width 8 fits 99.4%, and width 32 fits 100%. On the two-dimensional XOR inputs, width 32 fits only 64–75% of the random labels within the 800-update cap. Low-dimensional inputs and a finite training budget limit memorization there. This is a statement about the tested protocol, not a bound on what the architecture could fit.

Regenerate both tables from `runs.csv` and `protocol.json`:

```sh
python -m topic9.random_labels results
```
