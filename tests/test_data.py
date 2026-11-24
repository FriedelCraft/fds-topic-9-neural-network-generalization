"""Guard the experimental controls that make comparisons interpretable."""

import unittest

import numpy as np

from topic9.data import make_dataset, prepare_split


class DataTests(unittest.TestCase):
    def test_scaler_uses_selected_training_rows(self):
        dataset = make_dataset("synthetic")
        data_seed = 17
        indices = np.random.default_rng(data_seed).permutation(len(dataset.y_train_pool))[:32]
        train = dataset.x_train_pool[indices]
        center = train.mean(axis=0)
        scale = train.std(axis=0)
        split = prepare_split(dataset, 32, data_seed)

        np.testing.assert_array_equal(split.y_train, dataset.y_train_pool[indices])
        np.testing.assert_allclose(split.x_train, (train - center) / scale)
        np.testing.assert_allclose(split.x_val, (dataset.x_val - center) / scale)
        np.testing.assert_allclose(split.x_test, (dataset.x_test - center) / scale)

    def test_random_controls_preserve_matching_rows_and_nested_labels(self):
        dataset = make_dataset("synthetic")
        observed = prepare_split(dataset, 32, data_seed=17)
        small = prepare_split(dataset, 32, data_seed=17, label_seed=21)
        large = prepare_split(dataset, 96, data_seed=17, label_seed=21)
        repeat = prepare_split(dataset, 32, data_seed=17, label_seed=22)

        np.testing.assert_array_equal(observed.x_train, small.x_train)
        np.testing.assert_array_equal(observed.x_val, small.x_val)
        np.testing.assert_array_equal(observed.x_test, small.x_test)
        np.testing.assert_array_equal(small.y_train, large.y_train[:32])
        np.testing.assert_array_equal(small.y_val, large.y_val)
        np.testing.assert_array_equal(small.y_test, large.y_test)
        self.assertFalse(np.array_equal(observed.y_train, small.y_train))
        self.assertFalse(np.array_equal(small.y_test, repeat.y_test))

    def test_real_data_disjoint_and_stratified(self):
        dataset = make_dataset("wdbc")
        train = set(map(tuple, dataset.x_train_pool))
        val = set(map(tuple, dataset.x_val))
        test = set(map(tuple, dataset.x_test))

        self.assertEqual((len(train), len(val), len(test)), (341, 114, 114))
        self.assertEqual(len(train | val | test), 569)
        np.testing.assert_array_equal(
            [dataset.y_train_pool.sum(), dataset.y_val.sum(), dataset.y_test.sum()],
            [127, 43, 42],
        )


class RandomLabelTests(unittest.TestCase):
    def test_random_labels_ignore_features_and_true_labels(self):
        from dataclasses import replace

        dataset = make_dataset("wdbc")
        scrambled = replace(
            dataset,
            x_train_pool=dataset.x_train_pool[::-1] * 3.0 + 1.0,
            y_train_pool=1.0 - dataset.y_train_pool,
            x_val=np.zeros_like(dataset.x_val),
            y_val=1.0 - dataset.y_val,
            x_test=np.ones_like(dataset.x_test),
            y_test=1.0 - dataset.y_test,
        )
        original = prepare_split(dataset, 96, data_seed=301, label_seed=10_301)
        changed = prepare_split(scrambled, 96, data_seed=301, label_seed=10_301)

        np.testing.assert_array_equal(original.y_train, changed.y_train)
        np.testing.assert_array_equal(original.y_val, changed.y_val)
        np.testing.assert_array_equal(original.y_test, changed.y_test)

    def test_split_streams_are_separate(self):
        dataset = make_dataset("synthetic")
        split = prepare_split(dataset, 256, data_seed=301, label_seed=10_301)
        n = min(len(split.y_train), len(split.y_val))
        self.assertFalse(np.array_equal(split.y_train[:n], split.y_val[:n]))
        self.assertFalse(np.array_equal(split.y_val[:n], split.y_test[:n]))
        self.assertTrue(set(np.unique(split.y_test)) <= {0.0, 1.0})

    def test_experiment_label_seeds_are_distinct_from_other_seeds(self):
        from topic9.data import DEFAULT_SPLIT_SEED
        from topic9.experiments import DATA_SEEDS, INIT_SEEDS, random_label_seed

        label_seeds = {random_label_seed(seed) for seed in DATA_SEEDS}
        self.assertEqual(len(label_seeds), len(DATA_SEEDS))
        self.assertFalse(label_seeds & (set(DATA_SEEDS) | set(INIT_SEEDS) | {DEFAULT_SPLIT_SEED}))

    def test_chance_interval_is_exact_binomial(self):
        from math import comb

        from topic9.random_labels import binomial_interval

        for n in (114, 2048):
            lo, hi = binomial_interval(n)
            k_lo, k_hi = round(lo * n), round(hi * n)
            coverage = sum(comb(n, k) for k in range(k_lo, k_hi + 1)) / 2**n
            self.assertGreaterEqual(coverage, 0.95)
            self.assertAlmostEqual(lo + hi, 1.0)
            wider = sum(comb(n, k) for k in range(k_lo + 1, k_hi)) / 2**n
            self.assertLess(wider, 0.95)
