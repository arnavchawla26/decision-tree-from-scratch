import csv
import json
import subprocess
import sys

import pytest

from dtree_scratch.cli import main


def _write_classification_csv(path):
    rows = [
        ["age", "income", "city", "buys"],
        ["22", "20000", "sf", "no"],
        ["25", "22000", "sf", "no"],
        ["45", "80000", "nyc", "yes"],
        ["50", "95000", "nyc", "yes"],
        ["23", "21000", "sf", "no"],
        ["48", "88000", "nyc", "yes"],
        ["27", "23000", "sf", "no"],
        ["52", "99000", "nyc", "yes"],
        ["24", "19000", "sf", "no"],
        ["46", "85000", "nyc", "yes"],
    ]
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def _write_regression_csv(path):
    rows = [["x1", "x2", "y"]]
    for i in range(30):
        x1, x2 = float(i), float(i) * 0.5
        rows.append([str(x1), str(x2), str(3 * x1 - x2)])
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def test_demo_classification_tree_runs(capsys):
    rc = main(["demo", "--task", "classification", "--model", "tree", "--n-samples", "80"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "accuracy" in out


def test_demo_classification_forest_moons_runs(capsys):
    rc = main([
        "demo", "--task", "classification", "--model", "forest",
        "--dataset", "moons", "--n-samples", "80", "--n-estimators", "5",
    ])
    assert rc == 0
    assert "accuracy" in capsys.readouterr().out


def test_demo_regression_tree_runs(capsys):
    rc = main(["demo", "--task", "regression", "--model", "tree", "--n-samples", "80"])
    assert rc == 0
    assert "rmse" in capsys.readouterr().out


def test_demo_regression_forest_runs(capsys):
    rc = main([
        "demo", "--task", "regression", "--model", "forest",
        "--n-samples", "80", "--n-estimators", "5",
    ])
    assert rc == 0
    assert "r2" in capsys.readouterr().out


def test_train_classification_tree_saves_model_and_reports_metrics(tmp_path, capsys):
    csv_path = tmp_path / "buys.csv"
    _write_classification_csv(csv_path)
    model_path = tmp_path / "model.json"

    rc = main([
        "train", "--csv", str(csv_path), "--target", "buys", "--task", "classification",
        "--model", "tree", "--out", str(model_path), "--test-size", "0.3", "--random-state", "1",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "accuracy" in out
    assert "feature importances" in out
    assert model_path.exists()

    with open(model_path) as f:
        payload = json.load(f)
    assert payload["task"] == "classification"
    assert payload["model_type"] == "tree"


def test_train_regression_forest_saves_model(tmp_path, capsys):
    csv_path = tmp_path / "reg.csv"
    _write_regression_csv(csv_path)
    model_path = tmp_path / "model.json"

    rc = main([
        "train", "--csv", str(csv_path), "--target", "y", "--task", "regression",
        "--model", "forest", "--n-estimators", "5", "--out", str(model_path),
        "--test-size", "0.2", "--random-state", "0",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "rmse" in out
    with open(model_path) as f:
        payload = json.load(f)
    assert payload["model_type"] == "forest"


def test_predict_uses_saved_model_and_writes_csv(tmp_path, capsys):
    csv_path = tmp_path / "buys.csv"
    _write_classification_csv(csv_path)
    model_path = tmp_path / "model.json"
    main([
        "train", "--csv", str(csv_path), "--target", "buys", "--task", "classification",
        "--model", "tree", "--out", str(model_path), "--random-state", "0",
    ])
    capsys.readouterr()  # discard train output

    new_csv = tmp_path / "new_rows.csv"
    with open(new_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["age", "income", "city", "buys"])
        writer.writerow(["23", "21000", "sf", "?"])
        writer.writerow(["49", "90000", "nyc", "?"])

    out_csv = tmp_path / "predictions.csv"
    rc = main(["predict", "--model", str(model_path), "--csv", str(new_csv), "--out", str(out_csv)])
    assert rc == 0
    assert out_csv.exists()

    with open(out_csv) as f:
        rows = list(csv.reader(f))
    assert rows[0][-1] == "buys_predicted"
    assert rows[1][-1] in ("yes", "no")
    assert rows[2][-1] in ("yes", "no")
    # sanity: the low-age/low-income row should predict "no", the
    # high-age/high-income row should predict "yes" — this training
    # set is cleanly separable on income alone.
    assert rows[1][-1] == "no"
    assert rows[2][-1] == "yes"


def test_predict_without_out_writes_to_stdout(tmp_path, capsys):
    csv_path = tmp_path / "buys.csv"
    _write_classification_csv(csv_path)
    model_path = tmp_path / "model.json"
    main([
        "train", "--csv", str(csv_path), "--target", "buys", "--task", "classification",
        "--model", "tree", "--out", str(model_path), "--random-state", "0",
    ])
    capsys.readouterr()

    new_csv = tmp_path / "new_rows.csv"
    with open(new_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["age", "income", "city", "buys"])
        writer.writerow(["23", "21000", "sf", "?"])

    rc = main(["predict", "--model", str(model_path), "--csv", str(new_csv)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "buys_predicted" in out


def test_cli_end_to_end_subprocess_console_script(tmp_path):
    """Real subprocess invocation of the installed `dtree` console script."""
    csv_path = tmp_path / "buys.csv"
    _write_classification_csv(csv_path)
    model_path = tmp_path / "model.json"

    result = subprocess.run(
        [
            "dtree", "train", "--csv", str(csv_path), "--target", "buys",
            "--task", "classification", "--model", "tree", "--out", str(model_path),
            "--random-state", "0",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "accuracy" in result.stdout
    assert model_path.exists()


def test_load_csv_missing_target_column_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows([["a", "b"], ["1", "2"]])

    with pytest.raises(ValueError):
        main([
            "train", "--csv", str(csv_path), "--target", "nope",
            "--task", "classification", "--model", "tree",
        ])
