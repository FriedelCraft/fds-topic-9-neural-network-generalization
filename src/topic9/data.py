"""Reproducible datasets, paired train subsets, and train-only preprocessing."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
DEFAULT_SPLIT_SEED = 2026


@dataclass(frozen=True)
class Dataset:
    name: str
    x_train_pool: FloatArray
    y_train_pool: FloatArray
    x_val: FloatArray
    y_val: FloatArray
    x_test: FloatArray
    y_test: FloatArray


@dataclass(frozen=True)
class Split:
    x_train: FloatArray
    y_train: FloatArray
    x_val: FloatArray
    y_val: FloatArray
    x_test: FloatArray
    y_test: FloatArray


def _noisy_xor(size: int, seed: np.random.SeedSequence) -> tuple[FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    features = rng.uniform(-1.0, 1.0, size=(size, 2))
    clean_labels = features[:, 0] * features[:, 1] < 0
    flips = rng.random(size) < 0.10
    return features, (clean_labels ^ flips).astype(np.float64)


def make_dataset(name: str, split_seed: int = DEFAULT_SPLIT_SEED) -> Dataset:
    """Construct a fixed train reservoir and held-out pools before any tuning."""
    if name == "synthetic":
        train_seed, val_seed, test_seed = np.random.SeedSequence(split_seed).spawn(3)
        x_train, y_train = _noisy_xor(4096, train_seed)
        x_val, y_val = _noisy_xor(256, val_seed)
        x_test, y_test = _noisy_xor(2048, test_seed)
    elif name == "wdbc":
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split

        x, target = load_breast_cancer(return_X_y=True)
        # scikit-learn encodes malignant as 0. Use malignant as the positive class.
        y = (target == 0).astype(np.float64)
        x_dev, x_test, y_dev, y_test = train_test_split(
            x, y, test_size=114, stratify=y, random_state=split_seed
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_dev, y_dev, test_size=114, stratify=y_dev, random_state=split_seed + 1
        )
    else:
        raise ValueError(f"Unknown dataset {name!r}; use 'synthetic' or 'wdbc'.")

    for array in (x_train, y_train, x_val, y_val, x_test, y_test):
        array.setflags(write=False)
    return Dataset(name, x_train, y_train, x_val, y_val, x_test, y_test)


def prepare_split(
    dataset: Dataset, train_size: int, data_seed: int, label_seed: int | None = None
) -> Split:
    """Select a seeded prefix and standardize with its mean and population SD.

    For a random-label control, independently draw every pool's labels before
    subsetting. Matching seeds preserve labels on nested prefixes and across
    every model and optimization setting.
    """
    pool_size = len(dataset.y_train_pool)
    if not 1 <= train_size <= pool_size:
        raise ValueError(f"train_size must be between 1 and {pool_size}.")
    indices = np.random.default_rng(data_seed).permutation(pool_size)[:train_size]
    x_train = dataset.x_train_pool[indices]
    center = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    # Rounding the mean of identical decimals can give a constant column a tiny positive SD.
    constant = np.all(x_train == x_train[0], axis=0)
    center[constant] = x_train[0, constant]
    scale = np.where((std > 0.0) & ~constant, std, 1.0)

    y_train_pool = dataset.y_train_pool
    y_val = dataset.y_val
    y_test = dataset.y_test
    if label_seed is not None:
        train_seed, val_seed, test_seed = np.random.SeedSequence(label_seed).spawn(3)
        y_train_pool = np.random.default_rng(train_seed).integers(2, size=pool_size)
        y_val = np.random.default_rng(val_seed).integers(2, size=len(y_val))
        y_test = np.random.default_rng(test_seed).integers(2, size=len(y_test))

    return Split(
        (x_train - center) / scale,
        y_train_pool[indices].astype(np.float64),
        (dataset.x_val - center) / scale,
        y_val.astype(np.float64),
        (dataset.x_test - center) / scale,
        y_test.astype(np.float64),
    )
