"""Minimal CSV loading + categorical encoding, no pandas/numpy.

The CLI trains on arbitrary CSVs, so columns need to become plain
floats before they reach the trees. Numeric-looking columns pass
through as floats; anything else is label-encoded (each distinct
string value maps to a small integer, in first-seen order) and the
encoding map is returned so ``predict`` can reuse it consistently on
new rows.
"""

from __future__ import annotations

import csv
from typing import Sequence


class ColumnEncoder:
    """Encodes one column's string values to floats, consistently."""

    def __init__(self, is_numeric: bool) -> None:
        self.is_numeric = is_numeric
        self.value_to_code: dict[str, int] = {}

    def encode(self, raw_value: str) -> float:
        if self.is_numeric:
            return float(raw_value)
        if raw_value not in self.value_to_code:
            self.value_to_code[raw_value] = len(self.value_to_code)
        return float(self.value_to_code[raw_value])

    def encode_known(self, raw_value: str) -> float:
        """Like encode(), but never grows the mapping (for predict time)."""
        if self.is_numeric:
            return float(raw_value)
        if raw_value in self.value_to_code:
            return float(self.value_to_code[raw_value])
        # Unseen category at prediction time: map to a new trailing code
        # rather than raising, so a single unfamiliar row doesn't crash
        # a whole prediction batch.
        return float(len(self.value_to_code))


def _is_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def load_csv(path: str) -> tuple[list[str], list[list[str]]]:
    """Read a CSV file, returning (header, rows) as raw strings."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} is empty")
    header, data_rows = rows[0], rows[1:]
    if not data_rows:
        raise ValueError(f"{path} has a header but no data rows")
    return header, data_rows


def build_encoders(
    header: Sequence[str], rows: Sequence[Sequence[str]], target_column: str
) -> tuple[list[int], int, dict[int, ColumnEncoder]]:
    """Decide which columns are features, find the target index, and
    build one ColumnEncoder per feature column based on the training data.

    Returns (feature_indices, target_index, encoders_by_feature_index).
    """
    if target_column not in header:
        raise ValueError(f"Target column {target_column!r} not found in header {header!r}")
    target_index = header.index(target_column)
    feature_indices = [i for i in range(len(header)) if i != target_index]

    encoders: dict[int, ColumnEncoder] = {}
    for i in feature_indices:
        column_values = [row[i] for row in rows]
        numeric = all(_is_numeric(v) for v in column_values)
        encoders[i] = ColumnEncoder(is_numeric=numeric)

    return feature_indices, target_index, encoders


def encode_dataset(
    rows: Sequence[Sequence[str]],
    feature_indices: Sequence[int],
    target_index: int,
    encoders: dict[int, ColumnEncoder],
    target_is_numeric: bool,
) -> tuple[list[list[float]], list]:
    """Encode raw string rows into (X, y) ready for a tree/forest."""
    X: list[list[float]] = []
    y: list = []
    for row in rows:
        X.append([encoders[i].encode(row[i]) for i in feature_indices])
        target_raw = row[target_index]
        y.append(float(target_raw) if target_is_numeric else target_raw)
    return X, y


def encode_rows_for_prediction(
    rows: Sequence[Sequence[str]],
    feature_indices: Sequence[int],
    encoders: dict[int, ColumnEncoder],
) -> list[list[float]]:
    """Encode new rows (predict time) using an already-fit encoder map."""
    return [[encoders[i].encode_known(row[i]) for i in feature_indices] for row in rows]
