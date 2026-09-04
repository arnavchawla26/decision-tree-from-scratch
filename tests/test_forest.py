import random

import pytest

from dtree_scratch.forest import RandomForestClassifier, RandomForestRegressor
from dtree_scratch.metrics import accuracy_score, r2_score


def _make_classification_blobs(n=120, seed=0):
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        cx, cy = rng.uniform(0, 10), rng.uniform(0, 10)
        label = "a" if cx + cy < 10 else "b"
        X.append([cx, cy])
        y.append(label)
    return X, y


def _make_regression_data(n=120, seed=0):
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        x1 = rng.uniform(-5, 5)
        x2 = rng.uniform(-5, 5)
        noise = rng.uniform(-0.5, 0.5)
        X.append([x1, x2])
        y.append(3 * x1 - 2 * x2 + noise)
    return X, y


def test_forest_classifier_fits_and_beats_majority_baseline():
    X, y = _make_classification_blobs(seed=1)
    n_train = 90
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    clf = RandomForestClassifier(n_estimators=15, random_state=0)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    from collections import Counter

    majority_label = Counter(y_train).most_common(1)[0][0]
    baseline_acc = accuracy_score(y_test, [majority_label] * len(y_test))
    assert acc >= baseline_acc
    assert acc > 0.7


def test_forest_classifier_n_estimators_produces_that_many_trees():
    X, y = _make_classification_blobs(n=30, seed=2)
    clf = RandomForestClassifier(n_estimators=7, random_state=0)
    clf.fit(X, y)
    assert len(clf.trees_) == 7


def test_forest_classifier_predict_proba_sums_to_one():
    X, y = _make_classification_blobs(n=40, seed=3)
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(X, y)
    for probs in clf.predict_proba(X):
        assert sum(probs) == pytest.approx(1.0, abs=1e-9)


def test_forest_classifier_feature_importances_sum_to_one():
    X, y = _make_classification_blobs(n=60, seed=4)
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(X, y)
    assert sum(clf.feature_importances_) == pytest.approx(1.0)


def test_forest_classifier_not_fitted_predict_raises():
    clf = RandomForestClassifier()
    with pytest.raises(RuntimeError):
        clf.predict([[1.0, 2.0]])


def test_forest_classifier_invalid_n_estimators_raises():
    with pytest.raises(ValueError):
        RandomForestClassifier(n_estimators=0)


def test_forest_classifier_deterministic_with_seed():
    X, y = _make_classification_blobs(n=50, seed=5)
    preds = []
    for _ in range(2):
        clf = RandomForestClassifier(n_estimators=8, random_state=99)
        clf.fit(X, y)
        preds.append(clf.predict(X))
    assert preds[0] == preds[1]


def test_forest_classifier_bootstrap_false_uses_full_dataset_each_tree():
    X, y = _make_classification_blobs(n=20, seed=6)
    clf = RandomForestClassifier(n_estimators=5, bootstrap=False, random_state=0)
    clf.fit(X, y)
    for tree in clf.trees_:
        assert tree.root_.n_samples == 20


def test_forest_regressor_fits_linear_relationship_well():
    X, y = _make_regression_data(seed=1)
    n_train = 90
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    reg = RandomForestRegressor(n_estimators=20, random_state=0)
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    assert r2_score(y_test, preds) > 0.7


def test_forest_regressor_predict_returns_averaged_values():
    X, y = _make_regression_data(n=50, seed=2)
    reg = RandomForestRegressor(n_estimators=10, random_state=0)
    reg.fit(X, y)
    preds = reg.predict(X)
    assert len(preds) == len(X)
    assert all(isinstance(p, float) for p in preds)


def test_forest_regressor_feature_importances_prefer_bigger_coefficient():
    # y = 3*x1 - 2*x2 + noise -> x1 should matter more than x2
    X, y = _make_regression_data(n=150, seed=3)
    reg = RandomForestRegressor(n_estimators=15, random_state=0, max_features=None)
    reg.fit(X, y)
    importances = reg.feature_importances_
    assert sum(importances) == pytest.approx(1.0)
    assert importances[0] > importances[1]


def test_forest_regressor_not_fitted_predict_raises():
    reg = RandomForestRegressor()
    with pytest.raises(RuntimeError):
        reg.predict([[1.0, 2.0]])


def test_forest_mismatched_lengths_raise():
    clf = RandomForestClassifier()
    with pytest.raises(ValueError):
        clf.fit([[1.0], [2.0]], ["a"])
