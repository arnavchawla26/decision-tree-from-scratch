"""Impurity criteria for CART splitting: Gini, entropy, and MSE.

Each impurity function takes a list of labels (classification) or
target values (regression) and returns a single non-negative float
scoring how "mixed" the set is (0 = pure). ``weighted_impurity_drop``
scores a candidate split the way CART does: the impurity reduction
from the parent to the size-weighted average of the two children.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Sequence


def gini_impurity(labels: Sequence) -> float:
    """Gini impurity: 1 - sum(p_i^2) over class proportions p_i."""
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    return 1.0 - sum((count / n) ** 2 for count in counts.values())


def entropy_impurity(labels: Sequence) -> float:
    """Shannon entropy in bits: -sum(p_i * log2(p_i))."""
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    total = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:
            total -= p * math.log2(p)
    return total


def mse_impurity(values: Sequence[float]) -> float:
    """Mean squared error around the mean — the regression impurity."""
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / n


CLASSIFICATION_CRITERIA: dict[str, Callable[[Sequence], float]] = {
    "gini": gini_impurity,
    "entropy": entropy_impurity,
}

REGRESSION_CRITERIA: dict[str, Callable[[Sequence[float]], float]] = {
    "mse": mse_impurity,
}


def weighted_impurity(
    left: Sequence, right: Sequence, impurity_fn: Callable[[Sequence], float]
) -> float:
    """Size-weighted average impurity of two child partitions."""
    n_left, n_right = len(left), len(right)
    n_total = n_left + n_right
    if n_total == 0:
        return 0.0
    return (
        n_left / n_total * impurity_fn(left) + n_right / n_total * impurity_fn(right)
    )


def impurity_decrease(
    parent: Sequence,
    left: Sequence,
    right: Sequence,
    impurity_fn: Callable[[Sequence], float],
) -> float:
    """Impurity reduction achieved by splitting ``parent`` into left/right.

    This is exactly what CART maximizes when choosing a split: a
    non-negative value (0 means the split didn't help at all).
    """
    return impurity_fn(parent) - weighted_impurity(left, right, impurity_fn)
