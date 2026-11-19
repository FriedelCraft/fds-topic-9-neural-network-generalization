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
