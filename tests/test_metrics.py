import pytest

from dtree_scratch.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_f1,
    r2_score,
    root_mean_squared_error,
    train_test_split,
)


def test_accuracy_all_correct():
    assert accuracy_score(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_accuracy_all_wrong():
    assert accuracy_score(["a", "b"], ["b", "a"]) == 0.0


def test_accuracy_partial():
    assert accuracy_score(["a", "a", "b", "b"], ["a", "b", "b", "b"]) == pytest.approx(0.75)


def test_accuracy_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        accuracy_score(["a"], ["a", "b"])


def test_accuracy_empty_is_zero():
    assert accuracy_score([], []) == 0.0


def test_confusion_matrix_diagonal_for_perfect_predictions():
    labels, matrix = confusion_matrix(["a", "b", "a"], ["a", "b", "a"])
    assert labels == ["a", "b"]
    assert matrix == [[2, 0], [0, 1]]


def test_confusion_matrix_off_diagonal_counts_errors():
    labels, matrix = confusion_matrix(["a", "a", "b"], ["a", "b", "b"])
    assert labels == ["a", "b"]
    # true=a,pred=a -> 1 ; true=a,pred=b -> 1 ; true=b,pred=b -> 1
    assert matrix == [[1, 1], [0, 1]]


def test_precision_recall_f1_perfect_classifier():
    result = precision_recall_f1(["a", "b", "a", "b"], ["a", "b", "a", "b"])
    assert result["a"]["precision"] == pytest.approx(1.0)
    assert result["a"]["recall"] == pytest.approx(1.0)
    assert result["a"]["f1"] == pytest.approx(1.0)
    assert result["macro"]["f1"] == pytest.approx(1.0)


def test_precision_recall_f1_known_values():
    # true: a a a b b ; pred: a a b b b
    y_true = ["a", "a", "a", "b", "b"]
    y_pred = ["a", "a", "b", "b", "b"]
    result = precision_recall_f1(y_true, y_pred)
    # class a: tp=2, fp=0, fn=1 -> precision=1.0, recall=2/3
    assert result["a"]["precision"] == pytest.approx(1.0)
    assert result["a"]["recall"] == pytest.approx(2 / 3)
    # class b: tp=2, fp=1, fn=0 -> precision=2/3, recall=1.0
    assert result["b"]["precision"] == pytest.approx(2 / 3)
    assert result["b"]["recall"] == pytest.approx(1.0)


def test_precision_recall_f1_zero_division_handled():
    result = precision_recall_f1(["a", "a"], ["b", "b"], labels=["a", "b"])
    assert result["a"]["precision"] == 0.0
    assert result["a"]["recall"] == 0.0
    assert result["a"]["f1"] == 0.0


def test_mse_perfect_predictions_is_zero():
    assert mean_squared_error([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_mse_known_value():
    assert mean_squared_error([0.0, 0.0], [1.0, 3.0]) == pytest.approx((1 + 9) / 2)


def test_rmse_is_sqrt_of_mse():
    y_true, y_pred = [0.0, 0.0], [1.0, 3.0]
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(
        mean_squared_error(y_true, y_pred) ** 0.5
    )


def test_mae_known_value():
    assert mean_absolute_error([0.0, 0.0], [1.0, 3.0]) == pytest.approx(2.0)


def test_r2_perfect_predictions_is_one():
    assert r2_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_r2_mean_baseline_is_zero():
    y_true = [1.0, 2.0, 3.0, 4.0]
    mean = sum(y_true) / len(y_true)
    y_pred = [mean] * 4
    assert r2_score(y_true, y_pred) == pytest.approx(0.0)


def test_r2_constant_target_perfect_prediction_is_one():
    assert r2_score([5.0, 5.0, 5.0], [5.0, 5.0, 5.0]) == 1.0


def test_r2_constant_target_imperfect_prediction_is_zero():
    assert r2_score([5.0, 5.0, 5.0], [4.0, 5.0, 6.0]) == 0.0


def test_train_test_split_sizes():
    X = [[i] for i in range(20)]
    y = list(range(20))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    assert len(X_test) == 5
    assert len(X_train) == 15
    assert len(y_test) == 5
    assert len(y_train) == 15


def test_train_test_split_no_overlap_and_full_coverage():
    X = [[i] for i in range(30)]
    y = list(range(30))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
    train_vals = {row[0] for row in X_train}
    test_vals = {row[0] for row in X_test}
    assert train_vals.isdisjoint(test_vals)
    assert train_vals | test_vals == set(range(30))


def test_train_test_split_deterministic_with_seed():
    X = [[i] for i in range(20)]
    y = list(range(20))
    split1 = train_test_split(X, y, test_size=0.2, random_state=42)
    split2 = train_test_split(X, y, test_size=0.2, random_state=42)
    assert split1 == split2


def test_train_test_split_invalid_test_size_raises():
    with pytest.raises(ValueError):
        train_test_split([[1]], [1], test_size=1.5)
