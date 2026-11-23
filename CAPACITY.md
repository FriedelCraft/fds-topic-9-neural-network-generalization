# Paired capacity comparison

The capacity sweep compares logistic regression (width 0) with networks of widths 2, 8, and 32 at training sizes of 32, 96, and 256. Each comparison uses the same dataset, training size, data seed, and initialization seed. A data seed determines the training examples and thus the fitted feature scaler. The same initialization seed gives a reproducible draw for each architecture.

The [paired results](results/capacity_comparison.csv) summarize all 18 comparisons against logistic regression. For each split and initialization, subtract the baseline test score from the network test score. Average the three differences within each data seed, then report the mean and sample standard deviation across the five seed means. Accuracy differences are in percentage points. Negative cross-entropy differences favor the network. The last two columns count data seeds with a strict improvement after averaging over initializations.

| Dataset | Training examples | Hidden units | Mean accuracy difference ± SD | Mean cross-entropy difference ± SD | Seeds with higher accuracy / lower loss |
|---|---:|---:|---:|---:|---:|
| Noisy XOR | 32 | 32 | +30.16 ± 12.45 pp | −0.1634 ± 0.0895 | 5/5 · 5/5 |
| Noisy XOR | 256 | 32 | +33.56 ± 4.25 pp | −0.2607 ± 0.0109 | 5/5 · 5/5 |
| WDBC | 256 | 32 | −1.11 ± 1.06 pp | +0.0005 ± 0.0367 | 1/5 · 4/5 |

On noisy XOR, every tested width improved both metrics in every data-seed average. For width 32, the accuracy advantage varied less across data seeds at 256 examples than at 32. On WDBC, logistic regression had the highest mean accuracy in all nine comparisons. At 256 examples, width 32 achieved lower cross-entropy on four of five data seeds, with a near-zero mean difference. Accuracy depends on the 0.5 threshold, while cross-entropy also responds to the probabilities the models assign.

These comparisons use the terminal checkpoints of the original capacity sweep, including runs that reached the update cap. Both models share the same validation and test pools across data seeds within each dataset, so the standard deviations describe variation across training samples under this protocol.

Regenerate the table from `runs.csv` and `protocol.json` in the results directory:

```sh
python -m topic9.capacity results
```
