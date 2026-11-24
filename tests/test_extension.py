import tempfile
import unittest
from pathlib import Path

import numpy as np

from topic9.figures import _runtime_conditions
from topic9.benchmark import cost_models, measure, summarize
from topic9.sensitivity import _trajectory_summary, run as run_sensitivity


class BenchmarkTests(unittest.TestCase):
    def test_timings_are_matched_and_reproducible(self):
        rows = measure(data_seeds=(301,), init_seeds=(11,), repeats=2)
        self.assertEqual(len(rows), 16 * 2)
        outcomes = {}
        for row in rows:
            key = (row["dataset"], row["width"], row["budget"])
            outcomes.setdefault(key, set()).add((row["steps"], row["stop_reason"], row["compute"]))
            self.assertGreater(float(row["seconds"]), 0)
        self.assertTrue(all(len(values) == 1 for values in outcomes.values()))
        # The common-compute rule keeps every model at or below the width-8 reference ceiling.
        fixed = {(r["dataset"], r["width"]): r["compute"] for r in rows if r["budget"] == "fixed"}
        for dataset in ("synthetic", "wdbc"):
            self.assertLessEqual(fixed[(dataset, 32)], fixed[(dataset, 8)])
            self.assertGreater(fixed[(dataset, 32)], 0.99 * fixed[(dataset, 8)])

    def test_run_accepts_shorter_settings(self):
        import json
        from topic9.benchmark import run as run_benchmark
        with tempfile.TemporaryDirectory() as directory:
            run_benchmark(Path(directory), quick=True, init_seeds=(11, 29), repeats=1)
            metadata = json.loads((Path(directory) / "runtime.json").read_text(encoding="utf-8"))
            lines = (Path(directory) / "runtime.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual((metadata["repeats"], metadata["init_seeds"]), (1, [11, 29]))
        self.assertEqual(len(lines) - 1, 16 * 2)        # one data seed x two initialization seeds x one repeat

    def test_summary_and_cost_models(self):
        rows = []
        for width, steps in ((0, 10), (2, 40), (8, 90), (32, 400)):
            for repeat in (1, 2):
                rows.append(dict(
                    dataset="synthetic", width=width, train_size=96, budget="fixed", data_seed=1,
                    init_seed=1, repeat=repeat, seconds=f"{2e-4 * (steps + 1) * repeat ** 0:.6f}",
                    steps=steps, stop_reason="budget", compute=1000 * (steps + 1),
                ))
        summary = summarize(rows)
        self.assertEqual(len(summary), 4)
        models = cost_models(summary)
        self.assertLess(models["updates_only"]["median_relative_error"], 1e-3)
        self.assertGreater(models["work_only"]["max_relative_error"], 0)


class RuntimeFigureTests(unittest.TestCase):
    def test_work_is_averaged_over_matched_runs(self):
        rows = [
            dict(dataset="wdbc", width=32, budget="convergence", data_seed=seed, init_seed=1,
                 repeat=repeat, seconds=0.1 * seed, compute=compute)
            for seed, compute in ((1, 100), (2, 200), (3, 600)) for repeat in (1, 2)
        ]
        conditions, work = _runtime_conditions(rows)
        key = ("wdbc", 32, "convergence")
        self.assertEqual(work[key], 300)            # the mean, not the last run's 600
        self.assertEqual(len(conditions[key]), 3)


class SensitivityTests(unittest.TestCase):
    def test_trajectory_summary_counts_loss_increases(self):
        history = [{"train_loss": value} for value in (1.0, 0.8, 0.9, 0.7)]
        summary = _trajectory_summary(history)
        self.assertEqual(summary["loss_increases"], 1)
        self.assertAlmostEqual(summary["largest_loss_increase"], 0.1)
        self.assertEqual(summary["train_loss_drop_last_100"], "")

    def test_quick_run_writes_both_sweeps(self):
        with tempfile.TemporaryDirectory() as directory:
            run_sensitivity(Path(directory), quick=True)
            text = (Path(directory) / "convergence.csv").read_text(encoding="utf-8").splitlines()
        header = text[0].split(",")
        rows = [dict(zip(header, line.split(","))) for line in text[1:]]
        self.assertEqual(len(rows), 12 * 2)
        self.assertEqual({row["sweep"] for row in rows}, {"cap", "learning_rate"})
        self.assertTrue(all(row["stop_reason"] for row in rows))
        self.assertTrue(np.isfinite([float(row["test_loss"]) for row in rows if row["test_loss"]]).all())


if __name__ == "__main__":
    unittest.main()
