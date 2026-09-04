"""Evaluation metrics for classification and regression — no numpy."""

from __future__ import annotations

from collections import Counter
from typing import Sequence


def accuracy_score(y_true: Sequence, y_pred: Sequence) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def confusion_matrix(y_true: Sequence, y_pred: Sequence, labels: Sequence | None = None):
    """Return (labels, matrix) where matrix[i][j] counts true=labels[i], pred=labels[j]."""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred), key=lambda c: str(c))
    else:
        labels = list(labels)
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        matrix[index[t]][index[p]] += 1
    return labels, matrix


def precision_recall_f1(y_true: Sequence, y_pred: Sequence, labels: Sequence | None = None):
    """Per-class precision/recall/F1, plus macro averages.

    Returns a dict: {label: {"precision":.., "recall":.., "f1":..}, ...,
    "macro": {"precision":.., "recall":.., "f1":..}}.
    """
    label_list, matrix = confusion_matrix(y_true, y_pred, labels)
    n = len(label_list)
    result = {}
    precisions, recalls, f1s = [], [], []

    for i, label in enumerate(label_list):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(n)) - tp
        fn = sum(matrix[i][c] for c in range(n)) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        result[label] = {"precision": precision, "recall": recall, "f1": f1}
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    result["macro"] = {
        "precision": sum(precisions) / n if n else 0.0,
        "recall": sum(recalls) / n if n else 0.0,
        "f1": sum(f1s) / n if n else 0.0,
    }
    return result


def mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        return 0.0
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)


def root_mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return mean_squared_error(y_true, y_pred) ** 0.5


def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        return 0.0
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    mean_true = sum(y_true) / n
    ss_total = sum((t - mean_true) ** 2 for t in y_true)
    ss_residual = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    if ss_total == 0.0:
        return 1.0 if ss_residual == 0.0 else 0.0
    return 1.0 - ss_residual / ss_total


def train_test_split(X: Sequence, y: Sequence, test_size: float = 0.2, random_state=None):
    """Shuffle-and-split helper. Returns (X_train, X_test, y_train, y_test)."""
    import random as _random

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1 (exclusive)")
    n = len(X)
    if n != len(y):
        raise ValueError("X and y must have the same length")

    indices = list(range(n))
    _random.Random(random_state).shuffle(indices)
    n_test = max(1, int(round(n * test_size))) if n > 1 else 1
    test_idx = set(indices[:n_test])

    X_train = [X[i] for i in range(n) if i not in test_idx]
    X_test = [X[i] for i in range(n) if i in test_idx]
    y_train = [y[i] for i in range(n) if i not in test_idx]
    y_test = [y[i] for i in range(n) if i in test_idx]
    return X_train, X_test, y_train, y_test
