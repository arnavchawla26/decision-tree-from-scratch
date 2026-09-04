import pytest

from dtree_scratch.datasets import make_classification, make_moons, make_regression
from dtree_scratch.forest import RandomForestClassifier, RandomForestRegressor
from dtree_scratch.metrics import accuracy_score, r2_score


def test_make_classification_shapes_and_labels():
    X, y = make_classification(n_samples=100, n_features=3, n_classes=3, random_state=0)
    assert len(X) == 100
    assert len(y) == 100
    assert all(len(row) == 3 for row in X)
    assert set(y) == {"class_0", "class_1", "class_2"}


def test_make_classification_deterministic_with_seed():
    X1, y1 = make_classification(n_samples=30, random_state=42)
    X2, y2 = make_classification(n_samples=30, random_state=42)
    assert X1 == X2
    assert y1 == y2


def test_make_classification_is_learnable():
    X, y = make_classification(n_samples=150, class_sep=4.0, noise=0.5, random_state=0)
    n_train = 120
    clf = RandomForestClassifier(n_estimators=15, random_state=0)
    clf.fit(X[:n_train], y[:n_train])
    preds = clf.predict(X[n_train:])
    assert accuracy_score(y[n_train:], preds) > 0.85


def test_make_classification_invalid_args_raise():
    with pytest.raises(ValueError):
        make_classification(n_samples=1, n_classes=5)
    with pytest.raises(ValueError):
        make_classification(n_features=0)


def test_make_regression_shapes():
    X, y = make_regression(n_samples=80, n_features=4, random_state=0)
    assert len(X) == 80
    assert len(y) == 80
    assert all(len(row) == 4 for row in X)


def test_make_regression_deterministic_with_seed():
    X1, y1 = make_regression(n_samples=30, random_state=7)
    X2, y2 = make_regression(n_samples=30, random_state=7)
    assert X1 == X2
    assert y1 == y2


def test_make_regression_is_learnable():
    X, y = make_regression(n_samples=150, n_features=2, noise=0.3, random_state=0)
    n_train = 120
    reg = RandomForestRegressor(n_estimators=15, random_state=0)
    reg.fit(X[:n_train], y[:n_train])
    preds = reg.predict(X[n_train:])
    assert r2_score(y[n_train:], preds) > 0.85


def test_make_regression_invalid_features_raises():
    with pytest.raises(ValueError):
        make_regression(n_features=0)


def test_make_moons_shapes_and_labels():
    X, y = make_moons(n_samples=100, random_state=0)
    assert len(X) == 100
    assert len(y) == 100
    assert set(y) == {"moon_a", "moon_b"}


def test_make_moons_deterministic_with_seed():
    X1, y1 = make_moons(n_samples=40, random_state=3)
    X2, y2 = make_moons(n_samples=40, random_state=3)
    assert X1 == X2
    assert y1 == y2
