"""Synthetic dataset generators for classification and regression.

Deterministic given a ``random_state`` seed, and dependency-free
(plain ``random``, no numpy). Useful for demos, tests, and the CLI's
``--demo`` mode.
"""

from __future__ import annotations

import math
import random
from typing import Sequence


def make_classification(
    n_samples: int = 200,
    n_features: int = 2,
    n_classes: int = 2,
    class_sep: float = 2.0,
    noise: float = 1.0,
    random_state: int | None = None,
) -> tuple[list[list[float]], list[str]]:
    """Generate isotropic Gaussian blobs, one per class, in feature space.

    Class centers are placed on a circle (2D) / grid-ish pattern
    scaled by ``class_sep`` so higher separation makes an easier
    problem; ``noise`` is the per-feature Gaussian standard deviation.
    """
    if n_samples < n_classes:
        raise ValueError("n_samples must be >= n_classes")
    if n_features < 1:
        raise ValueError("n_features must be >= 1")

    rng = random.Random(random_state)
    labels = [f"class_{i}" for i in range(n_classes)]

    centers = []
    for i in range(n_classes):
        angle = 2 * math.pi * i / n_classes
        center = [class_sep * math.cos(angle), class_sep * math.sin(angle)]
        # extend to n_features with deterministic pseudo-random offsets
        while len(center) < n_features:
            center.append(class_sep * (0.5 - (i / max(n_classes, 1))))
        centers.append(center[:n_features])

    X, y = [], []
    for i in range(n_samples):
        class_idx = i % n_classes
        center = centers[class_idx]
        row = [center[f] + rng.gauss(0, noise) for f in range(n_features)]
        X.append(row)
        y.append(labels[class_idx])

    # shuffle so classes aren't grouped in order
    combined = list(zip(X, y))
    rng.shuffle(combined)
    X, y = [c[0] for c in combined], [c[1] for c in combined]
    return X, y


def make_regression(
    n_samples: int = 200,
    n_features: int = 2,
    noise: float = 1.0,
    random_state: int | None = None,
) -> tuple[list[list[float]], list[float]]:
    """Generate a linear-with-noise regression dataset.

    y = sum(coef_i * x_i) + noise, with coefficients drawn once from
    a fixed deterministic pattern so the "true" feature importances
    are known and predictable (feature 0 always matters most).
    """
    if n_features < 1:
        raise ValueError("n_features must be >= 1")

    rng = random.Random(random_state)
    # descending coefficients so feature importance has a clear ground truth
    coefficients = [n_features - i for i in range(n_features)]

    X, y = [], []
    for _ in range(n_samples):
        row = [rng.uniform(-5, 5) for _ in range(n_features)]
        target = sum(c * v for c, v in zip(coefficients, row)) + rng.gauss(0, noise)
        X.append(row)
        y.append(target)
    return X, y


def make_moons(
    n_samples: int = 200, noise: float = 0.15, random_state: int | None = None
) -> tuple[list[list[float]], list[str]]:
    """Generate the classic two-interleaving-half-moons 2D dataset.

    A nonlinear, non-axis-aligned classification problem — good for
    showing what a single tree's axis-aligned splits struggle with
    versus what a forest can approximate.
    """
    rng = random.Random(random_state)
    n_per_moon = n_samples // 2
    X, y = [], []

    for i in range(n_per_moon):
        angle = math.pi * i / n_per_moon
        x1 = math.cos(angle) + rng.gauss(0, noise)
        x2 = math.sin(angle) + rng.gauss(0, noise)
        X.append([x1, x2])
        y.append("moon_a")

    for i in range(n_samples - n_per_moon):
        angle = math.pi * i / n_per_moon
        x1 = 1 - math.cos(angle) + rng.gauss(0, noise)
        x2 = 1 - math.sin(angle) - 0.5 + rng.gauss(0, noise)
        X.append([x1, x2])
        y.append("moon_b")

    combined = list(zip(X, y))
    rng.shuffle(combined)
    return [c[0] for c in combined], [c[1] for c in combined]
