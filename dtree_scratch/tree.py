"""Dependency-free CART decision trees (classifier and regressor).

Pure Python, no numpy/scikit-learn. Implements the standard CART
algorithm: recursive binary splitting on numeric feature thresholds,
choosing at each node the (feature, threshold) pair that maximizes
impurity decrease, subject to max_depth / min_samples_split /
min_samples_leaf stopping rules. Feature importances are computed as
the mean decrease in impurity, exactly as scikit-learn defines it.

Rows (``X``) are plain lists of numbers, one row per sample. Encoding
categorical columns into numbers is the caller's job (see
``dtree_scratch.csv_utils``).
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Callable, Optional, Sequence

from dtree_scratch.criteria import (
    CLASSIFICATION_CRITERIA,
    REGRESSION_CRITERIA,
    impurity_decrease,
)


class _Node:
    """A single node in the tree: either a leaf or an internal split."""

    __slots__ = (
        "is_leaf",
        "prediction",
        "class_counts",
        "feature_index",
        "threshold",
        "left",
        "right",
        "n_samples",
        "impurity",
    )

    def __init__(
        self,
        *,
        is_leaf: bool,
        prediction=None,
        class_counts: Optional[dict] = None,
        feature_index: Optional[int] = None,
        threshold: Optional[float] = None,
        left: "_Node | None" = None,
        right: "_Node | None" = None,
        n_samples: int = 0,
        impurity: float = 0.0,
    ) -> None:
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.class_counts = class_counts
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.n_samples = n_samples
        self.impurity = impurity


def _node_to_dict(node: _Node) -> dict:
    return {
        "is_leaf": node.is_leaf,
        "prediction": node.prediction,
        "class_counts": node.class_counts,
        "feature_index": node.feature_index,
        "threshold": node.threshold,
        "left": _node_to_dict(node.left) if node.left is not None else None,
        "right": _node_to_dict(node.right) if node.right is not None else None,
        "n_samples": node.n_samples,
        "impurity": node.impurity,
    }


def _node_from_dict(d: dict) -> _Node:
    return _Node(
        is_leaf=d["is_leaf"],
        prediction=d["prediction"],
        class_counts=d["class_counts"],
        feature_index=d["feature_index"],
        threshold=d["threshold"],
        left=_node_from_dict(d["left"]) if d["left"] is not None else None,
        right=_node_from_dict(d["right"]) if d["right"] is not None else None,
        n_samples=d["n_samples"],
        impurity=d["impurity"],
    )


def _resolve_max_features(max_features, n_features: int) -> int:
    if max_features is None:
        return n_features
    if max_features == "sqrt":
        return max(1, int(n_features**0.5))
    if max_features == "log2":
        import math

        return max(1, int(math.log2(n_features)))
    if isinstance(max_features, float):
        return max(1, int(max_features * n_features))
    if isinstance(max_features, int):
        return max(1, min(max_features, n_features))
    raise ValueError(f"Unsupported max_features: {max_features!r}")


class BaseDecisionTree:
    """Shared CART machinery for classifier and regressor subclasses."""

    def __init__(
        self,
        criterion: str,
        criteria_table: dict[str, Callable[[Sequence], float]],
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features=None,
        random_state: Optional[int] = None,
    ) -> None:
        if criterion not in criteria_table:
            raise ValueError(
                f"Unknown criterion {criterion!r}; choose one of {sorted(criteria_table)}"
            )
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be >= 1 or None")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")

        self.criterion = criterion
        self._impurity_fn = criteria_table[criterion]
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self._rng = random.Random(random_state)

        self.root_: Optional[_Node] = None
        self.n_features_: int = 0
        self._importance_accum: list[float] = []
        self._n_train_samples: int = 0

    # -- template methods overridden by subclasses -------------------
    def _leaf_prediction(self, y: Sequence):
        raise NotImplementedError

    def _leaf_extra(self, y: Sequence) -> Optional[dict]:
        return None

    # -- public API ----------------------------------------------------
    def fit(self, X: Sequence[Sequence[float]], y: Sequence) -> "BaseDecisionTree":
        if len(X) == 0:
            raise ValueError("Cannot fit a tree on zero samples")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        self.n_features_ = len(X[0])
        self._importance_accum = [0.0] * self.n_features_
        self._n_train_samples = len(X)
        # depth=0 at the root, following the sklearn convention where
        # max_depth counts edges from root to leaf (a single-leaf tree
        # has depth 0; max_depth=1 allows exactly one split).
        self.root_ = self._build(list(X), list(y), depth=0)
        return self

    def predict(self, X: Sequence[Sequence[float]]) -> list:
        if self.root_ is None:
            raise RuntimeError("Tree is not fitted yet; call fit() first")
        return [self._predict_one(row, self.root_) for row in X]

    @property
    def feature_importances_(self) -> list[float]:
        total = sum(self._importance_accum)
        if total <= 0:
            return [0.0] * self.n_features_
        return [v / total for v in self._importance_accum]

    def depth(self) -> int:
        """Return the tree's depth (a single leaf has depth 1)."""
        return self._node_depth(self.root_) if self.root_ is not None else 0

    def _node_depth(self, node: _Node) -> int:
        if node.is_leaf:
            return 1
        return 1 + max(self._node_depth(node.left), self._node_depth(node.right))

    def count_nodes(self) -> int:
        return self._count_nodes(self.root_) if self.root_ is not None else 0

    def _count_nodes(self, node: _Node) -> int:
        if node.is_leaf:
            return 1
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)

    # -- serialization ---------------------------------------------------
    def to_dict(self) -> dict:
        """Serialize the fitted tree to a JSON-safe nested dict."""
        if self.root_ is None:
            raise RuntimeError("Tree is not fitted yet; call fit() first")
        return {
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "random_state": self.random_state,
            "n_features_": self.n_features_,
            "feature_importances_": self.feature_importances_,
            "root": _node_to_dict(self.root_),
        }

    def _load_common(self, d: dict) -> None:
        self.criterion = d["criterion"]
        self.max_depth = d["max_depth"]
        self.min_samples_split = d["min_samples_split"]
        self.min_samples_leaf = d["min_samples_leaf"]
        self.max_features = d["max_features"]
        self.random_state = d["random_state"]
        self.n_features_ = d["n_features_"]
        self._importance_accum = list(d["feature_importances_"])
        self.root_ = _node_from_dict(d["root"])

    # -- internals -------------------------------------------------------
    def _predict_one(self, row: Sequence[float], node: _Node):
        while not node.is_leaf:
            node = node.left if row[node.feature_index] <= node.threshold else node.right
        return node.prediction

    def _make_leaf(self, y: Sequence) -> _Node:
        return _Node(
            is_leaf=True,
            prediction=self._leaf_prediction(y),
            class_counts=self._leaf_extra(y),
            n_samples=len(y),
            impurity=self._impurity_fn(y),
        )

    def _build(self, X: list, y: list, depth: int) -> _Node:
        n_samples = len(y)
        node_impurity = self._impurity_fn(y)

        stop = (
            node_impurity == 0.0
            or n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        )
        # `depth` here is the number of edges from the root to this node;
        # reaching max_depth means this node itself must be a leaf.
        if stop:
            return self._make_leaf(y)

        split = self._best_split(X, y)
        if split is None:
            return self._make_leaf(y)

        feature_index, threshold, left_idx, right_idx, gain = split
        if gain <= 0.0:
            return self._make_leaf(y)

        self._importance_accum[feature_index] += (n_samples / self._n_train_samples) * gain

        left_X = [X[i] for i in left_idx]
        left_y = [y[i] for i in left_idx]
        right_X = [X[i] for i in right_idx]
        right_y = [y[i] for i in right_idx]

        left_node = self._build(left_X, left_y, depth + 1)
        right_node = self._build(right_X, right_y, depth + 1)

        return _Node(
            is_leaf=False,
            feature_index=feature_index,
            threshold=threshold,
            left=left_node,
            right=right_node,
            n_samples=n_samples,
            impurity=node_impurity,
        )

    def _candidate_features(self) -> list[int]:
        k = _resolve_max_features(self.max_features, self.n_features_)
        if k >= self.n_features_:
            return list(range(self.n_features_))
        return self._rng.sample(range(self.n_features_), k)

    def _best_split(self, X: list, y: list):
        n_samples = len(y)
        best_gain = 0.0
        best = None

        for feature_index in self._candidate_features():
            paired = sorted(zip((row[feature_index] for row in X), y))
            values = [p[0] for p in paired]
            sorted_labels = [p[1] for p in paired]

            # Candidate thresholds: midpoints between distinct consecutive values.
            for i in range(1, n_samples):
                if values[i] == values[i - 1]:
                    continue
                n_left = i
                n_right = n_samples - i
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                left_labels = sorted_labels[:i]
                right_labels = sorted_labels[i:]
                gain = impurity_decrease(sorted_labels, left_labels, right_labels, self._impurity_fn)

                if gain > best_gain:
                    threshold = (values[i - 1] + values[i]) / 2.0
                    left_idx = [
                        idx for idx, row in enumerate(X) if row[feature_index] <= threshold
                    ]
                    right_idx = [
                        idx for idx, row in enumerate(X) if row[feature_index] > threshold
                    ]
                    best_gain = gain
                    best = (feature_index, threshold, left_idx, right_idx, gain)

        return best


class DecisionTreeClassifier(BaseDecisionTree):
    """CART classification tree with Gini or entropy splitting."""

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features=None,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            criterion,
            CLASSIFICATION_CRITERIA,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            random_state,
        )
        self.classes_: list = []

    def fit(self, X, y) -> "DecisionTreeClassifier":
        self.classes_ = sorted(set(y), key=lambda c: str(c))
        return super().fit(X, y)

    def _leaf_prediction(self, y: Sequence):
        counts = Counter(y)
        max_count = max(counts.values())
        # Deterministic tie-break: smallest label (by string form) among the tied max.
        winners = sorted(
            (label for label, count in counts.items() if count == max_count),
            key=lambda c: str(c),
        )
        return winners[0]

    def _leaf_extra(self, y: Sequence) -> dict:
        return dict(Counter(y))

    def predict_proba(self, X) -> list[list[float]]:
        if self.root_ is None:
            raise RuntimeError("Tree is not fitted yet; call fit() first")
        rows = []
        for row in X:
            node = self.root_
            while not node.is_leaf:
                node = node.left if row[node.feature_index] <= node.threshold else node.right
            total = sum(node.class_counts.values())
            rows.append([node.class_counts.get(c, 0) / total for c in self.classes_])
        return rows

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["classes_"] = self.classes_
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionTreeClassifier":
        obj = cls(
            criterion=d["criterion"],
            max_depth=d["max_depth"],
            min_samples_split=d["min_samples_split"],
            min_samples_leaf=d["min_samples_leaf"],
            max_features=d["max_features"],
            random_state=d["random_state"],
        )
        obj._load_common(d)
        obj.classes_ = d["classes_"]
        return obj


class DecisionTreeRegressor(BaseDecisionTree):
    """CART regression tree with MSE splitting."""

    def __init__(
        self,
        criterion: str = "mse",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features=None,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            criterion,
            REGRESSION_CRITERIA,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            random_state,
        )

    def _leaf_prediction(self, y: Sequence[float]) -> float:
        return sum(y) / len(y)

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionTreeRegressor":
        obj = cls(
            criterion=d["criterion"],
            max_depth=d["max_depth"],
            min_samples_split=d["min_samples_split"],
            min_samples_leaf=d["min_samples_leaf"],
            max_features=d["max_features"],
            random_state=d["random_state"],
        )
        obj._load_common(d)
        return obj
