"""Library-level demo: compare a single tree vs. a random forest on the
two-moons dataset (a nonlinear problem where a forest should help)."""

from dtree_scratch.datasets import make_moons
from dtree_scratch.forest import RandomForestClassifier
from dtree_scratch.metrics import accuracy_score, precision_recall_f1, train_test_split
from dtree_scratch.tree import DecisionTreeClassifier


def main() -> None:
    X, y = make_moons(n_samples=400, noise=0.2, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    tree = DecisionTreeClassifier(max_depth=5, random_state=0)
    tree.fit(X_train, y_train)
    tree_preds = tree.predict(X_test)

    forest = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=0)
    forest.fit(X_train, y_train)
    forest_preds = forest.predict(X_test)

    print(f"train={len(X_train)} test={len(X_test)}\n")
    for name, preds in [("single tree", tree_preds), ("random forest", forest_preds)]:
        acc = accuracy_score(y_test, preds)
        f1 = precision_recall_f1(y_test, preds)["macro"]["f1"]
        print(f"{name:>13}: accuracy={acc:.4f}  macro_f1={f1:.4f}")


if __name__ == "__main__":
    main()
