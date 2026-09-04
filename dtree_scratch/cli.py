"""``dtree`` command-line interface: train, predict, and demo.

    dtree train  --csv data.csv --target label --task classification --model forest --out model.json
    dtree predict --model model.json --csv new_rows.csv --out predictions.csv
    dtree demo --task classification --model tree

All commands work on plain CSV files; no numpy/pandas anywhere in the
pipeline. ``train`` holds out a test split, reports metrics, and saves
a JSON model file that ``predict`` reloads later.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

from dtree_scratch.csv_utils import (
    build_encoders,
    encode_dataset,
    encode_rows_for_prediction,
    load_csv,
)
from dtree_scratch.datasets import make_classification, make_moons, make_regression
from dtree_scratch.forest import RandomForestClassifier, RandomForestRegressor
from dtree_scratch.metrics import (
    accuracy_score,
    mean_absolute_error,
    precision_recall_f1,
    r2_score,
    root_mean_squared_error,
    train_test_split,
)
from dtree_scratch.tree import DecisionTreeClassifier, DecisionTreeRegressor

MODEL_CLASSES = {
    ("classification", "tree"): DecisionTreeClassifier,
    ("classification", "forest"): RandomForestClassifier,
    ("regression", "tree"): DecisionTreeRegressor,
    ("regression", "forest"): RandomForestRegressor,
}


def _build_model(task: str, model_type: str, args: argparse.Namespace):
    cls = MODEL_CLASSES[(task, model_type)]
    kwargs = dict(
        criterion=args.criterion,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )
    if model_type == "forest":
        kwargs["n_estimators"] = args.n_estimators
        kwargs["max_features"] = args.max_features
    if args.criterion is None:
        kwargs.pop("criterion")
    return cls(**kwargs)


def _default_criterion(task: str, model_type: str) -> str:
    return "gini" if task == "classification" else "mse"


def cmd_train(args: argparse.Namespace) -> int:
    header, rows = load_csv(args.csv)
    if args.criterion is None:
        args.criterion = _default_criterion(args.task, args.model)

    feature_indices, target_index, encoders = build_encoders(header, rows, args.target)
    target_is_numeric = args.task == "regression"
    X, y = encode_dataset(rows, feature_indices, target_index, encoders, target_is_numeric)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    model = _build_model(args.task, args.model, args)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"Trained {args.model} ({args.task}) on {len(X_train)} rows, "
          f"evaluated on {len(X_test)} held-out rows.\n")

    if args.task == "classification":
        acc = accuracy_score(y_test, preds)
        print(f"accuracy: {acc:.4f}")
        pr = precision_recall_f1(y_test, preds)
        print(f"macro precision: {pr['macro']['precision']:.4f}  "
              f"macro recall: {pr['macro']['recall']:.4f}  "
              f"macro f1: {pr['macro']['f1']:.4f}")
    else:
        rmse = root_mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"rmse: {rmse:.4f}  mae: {mae:.4f}  r2: {r2:.4f}")

    feature_names = [header[i] for i in feature_indices]
    importances = model.feature_importances_
    print("\nfeature importances:")
    for name, importance in sorted(zip(feature_names, importances), key=lambda t: -t[1]):
        print(f"  {name}: {importance:.4f}")

    if args.out:
        payload = {
            "task": args.task,
            "model_type": args.model,
            "header": header,
            "target_column": args.target,
            "feature_indices": feature_indices,
            "target_is_numeric": target_is_numeric,
            "encoders": {
                str(i): {"is_numeric": enc.is_numeric, "value_to_code": enc.value_to_code}
                for i, enc in encoders.items()
            },
            "model": model.to_dict(),
        }
        with open(args.out, "w") as f:
            json.dump(payload, f)
        print(f"\nsaved model to {args.out}")

    return 0


def _load_model_payload(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _rebuild_model(payload: dict):
    cls = MODEL_CLASSES[(payload["task"], payload["model_type"])]
    return cls.from_dict(payload["model"])


class _LoadedEncoder:
    def __init__(self, is_numeric: bool, value_to_code: dict) -> None:
        self.is_numeric = is_numeric
        self.value_to_code = value_to_code

    def encode_known(self, raw_value: str) -> float:
        if self.is_numeric:
            return float(raw_value)
        if raw_value in self.value_to_code:
            return float(self.value_to_code[raw_value])
        return float(len(self.value_to_code))


def cmd_predict(args: argparse.Namespace) -> int:
    payload = _load_model_payload(args.model)
    model = _rebuild_model(payload)

    header, rows = load_csv(args.csv)
    feature_indices = payload["feature_indices"]
    encoders = {
        int(i): _LoadedEncoder(v["is_numeric"], v["value_to_code"])
        for i, v in payload["encoders"].items()
    }

    X_new = encode_rows_for_prediction(rows, feature_indices, encoders)
    preds = model.predict(X_new)

    out_header = header + [f"{payload['target_column']}_predicted"]
    out_rows = [list(row) + [pred] for row, pred in zip(rows, preds)]

    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(out_header)
            writer.writerows(out_rows)
        print(f"wrote {len(out_rows)} predictions to {args.out}")
    else:
        writer = csv.writer(sys.stdout)
        writer.writerow(out_header)
        writer.writerows(out_rows)

    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    if args.criterion is None:
        args.criterion = _default_criterion(args.task, args.model)

    if args.task == "classification":
        if args.dataset == "moons":
            X, y = make_moons(n_samples=args.n_samples, random_state=args.random_state)
        else:
            X, y = make_classification(
                n_samples=args.n_samples, n_classes=args.n_classes, random_state=args.random_state
            )
    else:
        X, y = make_regression(n_samples=args.n_samples, random_state=args.random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    model = _build_model(args.task, args.model, args)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"demo: {args.model} {args.task} on {len(X_train)} train / {len(X_test)} test rows\n")
    if args.task == "classification":
        acc = accuracy_score(y_test, preds)
        pr = precision_recall_f1(y_test, preds)
        print(f"accuracy: {acc:.4f}  macro f1: {pr['macro']['f1']:.4f}")
    else:
        rmse = root_mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"rmse: {rmse:.4f}  r2: {r2:.4f}")

    print("\nfeature importances:", [round(v, 4) for v in model.feature_importances_])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dtree", description="Dependency-free CART decision trees and random forests."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", choices=["tree", "forest"], default="tree")
    common.add_argument("--criterion", choices=["gini", "entropy", "mse"], default=None)
    common.add_argument("--max-depth", type=int, default=None, dest="max_depth")
    common.add_argument("--min-samples-split", type=int, default=2, dest="min_samples_split")
    common.add_argument("--min-samples-leaf", type=int, default=1, dest="min_samples_leaf")
    common.add_argument("--n-estimators", type=int, default=50, dest="n_estimators")
    common.add_argument("--max-features", default="sqrt", dest="max_features")
    common.add_argument("--random-state", type=int, default=0, dest="random_state")
    common.add_argument("--test-size", type=float, default=0.2, dest="test_size")

    train_p = subparsers.add_parser("train", parents=[common], help="Train on a CSV file.")
    train_p.add_argument("--csv", required=True)
    train_p.add_argument("--target", required=True, help="Target column name.")
    train_p.add_argument("--task", choices=["classification", "regression"], required=True)
    train_p.add_argument("--out", default=None, help="Path to save the trained model as JSON.")
    train_p.set_defaults(func=cmd_train)

    predict_p = subparsers.add_parser("predict", help="Predict with a saved model.")
    predict_p.add_argument("--model", required=True, dest="model", help="Path to a saved model JSON file.")
    predict_p.add_argument("--csv", required=True, help="CSV of new rows (target column optional/ignored).")
    predict_p.add_argument("--out", default=None, help="Where to write predictions CSV (stdout if omitted).")
    predict_p.set_defaults(func=cmd_predict)

    demo_p = subparsers.add_parser("demo", parents=[common], help="Run on a synthetic dataset.")
    demo_p.add_argument("--task", choices=["classification", "regression"], required=True)
    demo_p.add_argument("--dataset", choices=["blobs", "moons"], default="blobs")
    demo_p.add_argument("--n-samples", type=int, default=300, dest="n_samples")
    demo_p.add_argument("--n-classes", type=int, default=2, dest="n_classes")
    demo_p.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
