from __future__ import annotations

import numpy as np


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def spearman_rho(labels: list[float] | np.ndarray, preds: list[float] | np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    if len(labels) < 3:
        return float("nan")
    if np.std(labels) == 0 or np.std(preds) == 0:
        return float("nan")
    label_ranks = rankdata(labels)
    pred_ranks = rankdata(preds)
    return float(np.corrcoef(label_ranks, pred_ranks)[0, 1])


def kfold_indices(n_samples: int, n_splits: int, seed: int) -> list[tuple[list[int], list[int]]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_samples < n_splits:
        raise ValueError("n_samples must be >= n_splits")

    rng = np.random.RandomState(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    folds = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_idx.tolist(), val_idx.tolist()))
        current = stop
    return folds

