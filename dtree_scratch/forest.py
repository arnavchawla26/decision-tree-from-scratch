"""Dependency-free random forests built from the CART trees in tree.py.

A random forest is bagging (bootstrap-sample each tree's training set)
plus per-split feature subsampling (already supported by
``BaseDecisionTree`` via ``max_features``). Predictions aggregate by
majority vote (classifier) or mean (regressor); feature importances
average each tree's mean-decrease-in-impurity importances.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Optional, Sequence

from dtree_scratch.tree import DecisionTreeClassifier, DecisionTreeRegressor


class BaseRandomForest:
    _tree_cls = None  # set by subclass

    def __init__(
        self,
        n_estimators: int = 50,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features="sqrt",
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.trees_: list = []
        self.n_features_: int = 0

    def _bootstrap_sample(self, X, y, rng: random.Random):
        n = len(X)
        if not self.bootstrap:
            return list(X), list(y)
        indices = [rng.randrange(n) for _ in range(n)]
        return [X[i] for i in indices], [y[i] for i in indices]

    def fit(self, X: Sequence[Sequence[float]], y: Sequence) -> "BaseRandomForest":
        if len(X) == 0:
            raise ValueError("Cannot fit a forest on zero samples")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        master_rng = random.Random(self.random_state)
        self.n_features_ = len(X[0])
        self.trees_ = []

        for i in range(self.n_estimators):
            tree_seed = master_rng.randrange(2**31)
            tree_rng = random.Random(tree_seed)
            X_sample, y_sample = self._bootstrap_sample(X, y, tree_rng)

            tree = self._make_tree(tree_seed)
            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)

        return self

    def _make_tree(self, seed: int):
        raise NotImplementedError

    @property
    def feature_importances_(self) -> list[float]:
        if not self.trees_:
            return []
        n = self.n_features_
        totals = [0.0] * n
        for tree in self.trees_:
            for i, v in enumerate(tree.feature_importances_):
                totals[i] += v
        return [t / len(self.trees_) for t in totals]

    def predict(self, X: Sequence[Sequence[float]]) -> list:
        raise NotImplementedError

    def to_dict(self) -> dict:
        if not self.trees_:
            raise RuntimeError("Forest is not fitted yet; call fit() first")
        return {
            "n_estimators": self.n_estimators,
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
            "n_features_": self.n_features_,
            "trees": [tree.to_dict() for tree in self.trees_],
        }

    def _load_common(self, d: dict) -> None:
        self.n_estimators = d["n_estimators"]
        self.criterion = d["criterion"]
        self.max_depth = d["max_depth"]
        self.min_samples_split = d["min_samples_split"]
        self.min_samples_leaf = d["min_samples_leaf"]
        self.max_features = d["max_features"]
        self.bootstrap = d["bootstrap"]
        self.random_state = d["random_state"]
        self.n_features_ = d["n_features_"]


class RandomForestClassifier(BaseRandomForest):
    """Bagged ensemble of CART classification trees with majority voting."""

    def __init__(
        self,
        n_estimators: int = 50,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features="sqrt",
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            n_estimators,
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            bootstrap,
            random_state,
        )
        self.classes_: list = []

    def fit(self, X, y) -> "RandomForestClassifier":
        self.classes_ = sorted(set(y), key=lambda c: str(c))
        return super().fit(X, y)

    def _make_tree(self, seed: int) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=seed,
        )

    def predict(self, X) -> list:
        if not self.trees_:
            raise RuntimeError("Forest is not fitted yet; call fit() first")
        all_preds = [tree.predict(X) for tree in self.trees_]  # n_trees x n_samples
        results = []
        for sample_idx in range(len(X)):
            votes = Counter(all_preds[t][sample_idx] for t in range(len(self.trees_)))
            max_count = max(votes.values())
            winners = sorted(
                (label for label, count in votes.items() if count == max_count),
                key=lambda c: str(c),
            )
            results.append(winners[0])
        return results

    def predict_proba(self, X) -> list[list[float]]:
        if not self.trees_:
            raise RuntimeError("Forest is not fitted yet; call fit() first")
        n_trees = len(self.trees_)
        per_tree_proba = [tree.predict_proba(X) for tree in self.trees_]
        results = []
        for sample_idx in range(len(X)):
            n_classes = len(self.classes_)
            avg = [0.0] * n_classes
            for t in range(n_trees):
                tree_classes = self.trees_[t].classes_
                for local_i, cls in enumerate(tree_classes):
                    global_i = self.classes_.index(cls)
                    avg[global_i] += per_tree_proba[t][sample_idx][local_i]
            results.append([v / n_trees for v in avg])
        return results

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["classes_"] = self.classes_
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RandomForestClassifier":
        obj = cls(
            n_estimators=d["n_estimators"],
            criterion=d["criterion"],
            max_depth=d["max_depth"],
            min_samples_split=d["min_samples_split"],
            min_samples_leaf=d["min_samples_leaf"],
            max_features=d["max_features"],
            bootstrap=d["bootstrap"],
            random_state=d["random_state"],
        )
        obj._load_common(d)
        obj.classes_ = d["classes_"]
        obj.trees_ = [DecisionTreeClassifier.from_dict(t) for t in d["trees"]]
        return obj


class RandomForestRegressor(BaseRandomForest):
    """Bagged ensemble of CART regression trees, averaging predictions."""

    def __init__(
        self,
        n_estimators: int = 50,
        criterion: str = "mse",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features="sqrt",
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            n_estimators,
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            bootstrap,
            random_state,
        )

    def _make_tree(self, seed: int) -> DecisionTreeRegressor:
        return DecisionTreeRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=seed,
        )

    def predict(self, X) -> list[float]:
        if not self.trees_:
            raise RuntimeError("Forest is not fitted yet; call fit() first")
        all_preds = [tree.predict(X) for tree in self.trees_]
        n_trees = len(self.trees_)
        return [
            sum(all_preds[t][i] for t in range(n_trees)) / n_trees for i in range(len(X))
        ]

    @classmethod
    def from_dict(cls, d: dict) -> "RandomForestRegressor":
        obj = cls(
            n_estimators=d["n_estimators"],
            criterion=d["criterion"],
            max_depth=d["max_depth"],
            min_samples_split=d["min_samples_split"],
            min_samples_leaf=d["min_samples_leaf"],
            max_features=d["max_features"],
            bootstrap=d["bootstrap"],
            random_state=d["random_state"],
        )
        obj._load_common(d)
        obj.trees_ = [DecisionTreeRegressor.from_dict(t) for t in d["trees"]]
        return obj
