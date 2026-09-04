"""Library-level demo: compare a single regression tree vs. a random
forest, and print feature importances against the known ground truth
(make_regression's coefficients are descending: feature 0 matters most)."""

from dtree_scratch.datasets import make_regression
from dtree_scratch.forest import RandomForestRegressor
from dtree_scratch.metrics import mean_absolute_error, r2_score, train_test_split
from dtree_scratch.tree import DecisionTreeRegressor


def main() -> None:
    X, y = make_regression(n_samples=400, n_features=4, noise=1.5, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    tree = DecisionTreeRegressor(max_depth=6, random_state=0)
    tree.fit(X_train, y_train)
    tree_preds = tree.predict(X_test)

    forest = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=0)
    forest.fit(X_train, y_train)
    forest_preds = forest.predict(X_test)

    print(f"train={len(X_train)} test={len(X_test)}\n")
    for name, model, preds in [
        ("single tree", tree, tree_preds),
        ("random forest", forest, forest_preds),
    ]:
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        importances = [round(v, 3) for v in model.feature_importances_]
        print(f"{name:>13}: r2={r2:.4f}  mae={mae:.4f}  feature_importances={importances}")


if __name__ == "__main__":
    main()
