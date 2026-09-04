import pytest

from dtree_scratch.csv_utils import (
    ColumnEncoder,
    build_encoders,
    encode_dataset,
    encode_rows_for_prediction,
    load_csv,
)


def _write_csv(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content)
    return str(path)


def test_load_csv_returns_header_and_rows(tmp_path):
    path = _write_csv(tmp_path, "data.csv", "a,b,label\n1,2,x\n3,4,y\n")
    header, rows = load_csv(path)
    assert header == ["a", "b", "label"]
    assert rows == [["1", "2", "x"], ["3", "4", "y"]]


def test_load_csv_empty_file_raises(tmp_path):
    path = _write_csv(tmp_path, "empty.csv", "")
    with pytest.raises(ValueError):
        load_csv(path)


def test_load_csv_header_only_raises(tmp_path):
    path = _write_csv(tmp_path, "header_only.csv", "a,b,label\n")
    with pytest.raises(ValueError):
        load_csv(path)


def test_column_encoder_numeric_passthrough():
    enc = ColumnEncoder(is_numeric=True)
    assert enc.encode("3.5") == 3.5
    assert enc.encode("-2") == -2.0


def test_column_encoder_categorical_assigns_stable_codes():
    enc = ColumnEncoder(is_numeric=False)
    assert enc.encode("red") == 0.0
    assert enc.encode("blue") == 1.0
    assert enc.encode("red") == 0.0  # stable on repeat


def test_column_encoder_encode_known_does_not_grow_mapping():
    enc = ColumnEncoder(is_numeric=False)
    enc.encode("red")
    enc.encode("blue")
    before = dict(enc.value_to_code)
    code = enc.encode_known("green")  # unseen
    assert enc.value_to_code == before  # unchanged
    assert code == 2.0  # falls past the known codes


def test_build_encoders_detects_numeric_and_categorical_columns():
    header = ["age", "color", "label"]
    rows = [["25", "red", "yes"], ["30", "blue", "no"]]
    feature_indices, target_index, encoders = build_encoders(header, rows, "label")
    assert feature_indices == [0, 1]
    assert target_index == 2
    assert encoders[0].is_numeric is True
    assert encoders[1].is_numeric is False


def test_build_encoders_missing_target_raises():
    header = ["age", "color"]
    rows = [["25", "red"]]
    with pytest.raises(ValueError):
        build_encoders(header, rows, "label")


def test_encode_dataset_produces_floats_and_preserves_target():
    header = ["age", "color", "label"]
    rows = [["25", "red", "yes"], ["30", "blue", "no"]]
    feature_indices, target_index, encoders = build_encoders(header, rows, "label")
    X, y = encode_dataset(rows, feature_indices, target_index, encoders, target_is_numeric=False)
    assert X == [[25.0, 0.0], [30.0, 1.0]]
    assert y == ["yes", "no"]


def test_encode_dataset_numeric_target():
    header = ["x", "target"]
    rows = [["1", "10.5"], ["2", "20.5"]]
    feature_indices, target_index, encoders = build_encoders(header, rows, "target")
    X, y = encode_dataset(rows, feature_indices, target_index, encoders, target_is_numeric=True)
    assert X == [[1.0], [2.0]]
    assert y == [10.5, 20.5]


def test_encode_rows_for_prediction_reuses_encoder_state():
    header = ["color", "label"]
    rows = [["red", "a"], ["blue", "b"]]
    feature_indices, target_index, encoders = build_encoders(header, rows, "label")
    encode_dataset(rows, feature_indices, target_index, encoders, target_is_numeric=False)

    new_rows = [["blue", "?"], ["red", "?"]]
    X_new = encode_rows_for_prediction(new_rows, feature_indices, encoders)
    assert X_new == [[1.0], [0.0]]
