import random

import pytest

from dtree_scratch.tree import DecisionTreeClassifier, DecisionTreeRegressor


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def test_classifier_perfectly_separable_1d():
    X = [[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]]
    y = ["low", "low", "low", "high", "high", "high"]
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X, y)
    assert clf.predict(X) == y


def test_classifier_predict_unseen_points_between_clusters():
    X = [[0.0], [1.0], [2.0], [20.0], [21.0], [22.0]]
    y = ["a", "a", "a", "b", "b", "b"]
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X, y)
    assert clf.predict([[1.5]]) == ["a"]
    assert clf.predict([[21.5]]) == ["b"]


def test_classifier_single_class_is_a_pure_leaf():
    X = [[1.0], [2.0], [3.0]]
    y = ["only", "only", "only"]
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X, y)
    assert clf.root_.is_leaf
    assert clf.predict(X) == y


def test_classifier_max_depth_one_makes_a_single_split():
    X = [[i] for i in range(20)]
    y = ["lo" if i < 10 else "hi" for i in range(20)]
    clf = DecisionTreeClassifier(max_depth=1, random_state=0)
    clf.fit(X, y)
    assert clf.depth() == 2  # root split + 2 leaves
    assert not clf.root_.is_leaf
    assert clf.root_.left.is_leaf and clf.root_.right.is_leaf


def test_classifier_min_samples_leaf_respected():
    X = [[i] for i in range(10)]
    y = ["a"] * 5 + ["b"] * 5
    clf = DecisionTreeClassifier(min_samples_leaf=3, random_state=0)
    clf.fit(X, y)

    def check(node):
        if node.is_leaf:
            assert node.n_samples >= 3
        else:
            check(node.left)
            check(node.right)

    check(clf.root_)


def test_classifier_entropy_criterion_also_separates_cleanly():
    X = [[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]]
    y = ["low", "low", "low", "high", "high", "high"]
    clf = DecisionTreeClassifier(criterion="entropy", random_state=0)
    clf.fit(X, y)
    assert clf.predict(X) == y


def test_classifier_predict_proba_sums_to_one():
    X = [[1.0], [2.0], [10.0], [11.0]]
    y = ["a", "a", "b", "b"]
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X, y)
    for probs in clf.predict_proba(X):
        assert sum(probs) == pytest.approx(1.0)


def test_classifier_feature_importances_sum_to_one_and_ignore_useless_feature():
    rng = random.Random(1)
    X, y = [], []
    for _ in range(60):
        useful = rng.uniform(0, 10)
        useless = rng.uniform(0, 10)  # unrelated to label
        X.append([useful, useless])
        y.append("hi" if useful > 5 else "lo")
    clf = DecisionTreeClassifier(random_state=0, min_samples_leaf=2)
    clf.fit(X, y)
    importances = clf.feature_importances_
    assert sum(importances) == pytest.approx(1.0)
    assert importances[0] > importances[1]


def test_classifier_invalid_criterion_raises():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(criterion="nonsense")


def test_classifier_mismatched_lengths_raise():
    clf = DecisionTreeClassifier()
    with pytest.raises(ValueError):
        clf.fit([[1.0], [2.0]], ["a"])


def test_classifier_empty_training_set_raises():
    clf = DecisionTreeClassifier()
    with pytest.raises(ValueError):
        clf.fit([], [])


def test_classifier_not_fitted_predict_raises():
    clf = DecisionTreeClassifier()
    with pytest.raises(RuntimeError):
        clf.predict([[1.0]])


def test_classifier_deterministic_across_runs_same_random_state():
    rng = random.Random(7)
    X = [[rng.uniform(0, 10), rng.uniform(0, 10)] for _ in range(40)]
    y = ["a" if x[0] + x[1] > 10 else "b" for x in X]
    preds = []
    for _ in range(2):
        clf = DecisionTreeClassifier(random_state=3, max_features="sqrt")
        clf.fit(X, y)
        preds.append(clf.predict(X))
    assert preds[0] == preds[1]


# ---------------------------------------------------------------------------
# Regressor
# ---------------------------------------------------------------------------


def test_regressor_fits_step_function_closely():
    X = [[i] for i in range(20)]
    y = [1.0 if i < 10 else 5.0 for i in range(20)]
    reg = DecisionTreeRegressor(random_state=0)
    reg.fit(X, y)
    preds = reg.predict(X)
    for p, actual in zip(preds, y):
        assert p == pytest.approx(actual)


def test_regressor_predicts_local_mean():
    X = [[1.0], [1.0], [1.0], [5.0]]
    y = [2.0, 4.0, 6.0, 100.0]
    reg = DecisionTreeRegressor(random_state=0, min_samples_leaf=1)
    reg.fit(X, y)
    # the three x=1.0 points can't be split further (identical feature value)
    preds = reg.predict([[1.0]])
    assert preds[0] == pytest.approx(4.0)


def test_regressor_max_depth_limits_tree_size():
    # max_depth counts edges from root to leaf (sklearn convention), so
    # depth() (which counts node levels, a lone leaf == 1) is at most
    # max_depth + 1.
    X = [[i] for i in range(50)]
    y = [float(i) for i in range(50)]
    reg = DecisionTreeRegressor(max_depth=2, random_state=0)
    reg.fit(X, y)
    assert reg.depth() <= 3


def test_regressor_single_value_target_is_pure_leaf():
    X = [[1.0], [2.0], [3.0]]
    y = [7.0, 7.0, 7.0]
    reg = DecisionTreeRegressor()
    reg.fit(X, y)
    assert reg.root_.is_leaf
    assert reg.predict(X) == [7.0, 7.0, 7.0]

    def check(node):
        if node.is_leaf:
            assert node.n_samples >= 1
        else:
            check(node.left)
            check(node.right)


def test_regressor_feature_importances_prefer_informative_feature():
    rng = random.Random(2)
    X, y = [], []
    for _ in range(60):
        useful = rng.uniform(0, 10)
        useless = rng.uniform(0, 10)
        X.append([useful, useless])
        y.append(useful * 2 + rng.uniform(-0.1, 0.1))
    reg = DecisionTreeRegressor(random_state=0, min_samples_leaf=2)
    reg.fit(X, y)
    importances = reg.feature_importances_
    assert sum(importances) == pytest.approx(1.0)
    assert importances[0] > importances[1]


def test_regressor_min_samples_split_stops_growth():
    X = [[i] for i in range(6)]
    y = [float(i % 2) for i in range(6)]
    reg = DecisionTreeRegressor(min_samples_split=6, random_state=0)
    reg.fit(X, y)
    # with min_samples_split == n_samples, at most one split is possible
    assert reg.count_nodes() <= 3


def test_node_count_matches_manual_traversal():
    X = [[i] for i in range(8)]
    y = ["a" if i < 4 else "b" for i in range(8)]
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X, y)

    def count(node):
        if node.is_leaf:
            return 1
        return 1 + count(node.left) + count(node.right)

    assert clf.count_nodes() == count(clf.root_)
