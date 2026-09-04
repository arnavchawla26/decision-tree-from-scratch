import math

import pytest

from dtree_scratch.criteria import (
    entropy_impurity,
    gini_impurity,
    impurity_decrease,
    mse_impurity,
    weighted_impurity,
)


def test_gini_pure_set_is_zero():
    assert gini_impurity(["a", "a", "a"]) == 0.0


def test_gini_empty_set_is_zero():
    assert gini_impurity([]) == 0.0


def test_gini_binary_50_50_is_half():
    assert gini_impurity(["a", "b"]) == pytest.approx(0.5)


def test_gini_three_class_balanced():
    # 1 - 3*(1/3)^2 = 1 - 1/3 = 2/3
    assert gini_impurity(["a", "b", "c"]) == pytest.approx(2 / 3)


def test_entropy_pure_set_is_zero():
    assert entropy_impurity([1, 1, 1, 1]) == 0.0


def test_entropy_empty_set_is_zero():
    assert entropy_impurity([]) == 0.0


def test_entropy_binary_50_50_is_one_bit():
    assert entropy_impurity([0, 1]) == pytest.approx(1.0)


def test_entropy_matches_manual_formula():
    labels = [0, 0, 0, 1]
    p0, p1 = 3 / 4, 1 / 4
    expected = -(p0 * math.log2(p0) + p1 * math.log2(p1))
    assert entropy_impurity(labels) == pytest.approx(expected)


def test_mse_constant_values_is_zero():
    assert mse_impurity([5.0, 5.0, 5.0]) == 0.0


def test_mse_empty_is_zero():
    assert mse_impurity([]) == 0.0


def test_mse_matches_manual_variance():
    values = [1.0, 2.0, 3.0, 4.0]
    mean = 2.5
    expected = sum((v - mean) ** 2 for v in values) / 4
    assert mse_impurity(values) == pytest.approx(expected)


def test_weighted_impurity_equal_split():
    left, right = ["a", "a"], ["b", "b"]
    # each side pure -> weighted impurity 0
    assert weighted_impurity(left, right, gini_impurity) == pytest.approx(0.0)


def test_weighted_impurity_empty_total_is_zero():
    assert weighted_impurity([], [], gini_impurity) == 0.0


def test_impurity_decrease_perfect_split_equals_parent_impurity():
    parent = ["a", "a", "b", "b"]
    left, right = ["a", "a"], ["b", "b"]
    decrease = impurity_decrease(parent, left, right, gini_impurity)
    assert decrease == pytest.approx(gini_impurity(parent))


def test_impurity_decrease_useless_split_is_zero():
    parent = ["a", "b", "a", "b"]
    left, right = ["a", "b"], ["a", "b"]
    decrease = impurity_decrease(parent, left, right, gini_impurity)
    assert decrease == pytest.approx(0.0)


def test_impurity_decrease_never_negative_random_splits():
    import random

    rng = random.Random(42)
    for _ in range(20):
        parent = [rng.choice(["a", "b", "c"]) for _ in range(10)]
        split_at = rng.randint(1, 9)
        left, right = parent[:split_at], parent[split_at:]
        decrease = impurity_decrease(parent, left, right, gini_impurity)
        assert decrease >= -1e-12
