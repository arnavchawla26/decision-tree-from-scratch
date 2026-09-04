# decision-tree-from-scratch

CART decision trees and random forests, implemented from scratch in pure
Python — no numpy, no scikit-learn, no dependencies at all. Built to
understand (and demonstrate understanding of) how tree-based models
actually work under the hood: impurity criteria, recursive binary
splitting, pruning stopping rules, bagging, and feature importance.

## What it does

- **`DecisionTreeClassifier`** and **`DecisionTreeRegressor`** — CART
  trees that recursively pick the (feature, threshold) split maximizing
  impurity decrease, using Gini impurity or entropy for classification
  and MSE for regression.
- **`RandomForestClassifier`** and **`RandomForestRegressor`** —
  bagged ensembles of CART trees: each tree trains on a bootstrap
  resample of the data and considers a random subset of features
  (`sqrt`, `log2`, a fraction, or a fixed count) at every split.
  Predictions aggregate by majority vote (classifier) or mean
  (regressor).
- **Pruning controls**: `max_depth`, `min_samples_split`,
  `min_samples_leaf` — the standard CART stopping rules.
- **Feature importance** via mean decrease in impurity, computed and
  normalized exactly the way scikit-learn defines it, and averaged
  across trees for a forest.
- **Metrics**: accuracy, confusion matrix, per-class + macro
  precision/recall/F1 for classification; MSE, RMSE, MAE, R² for
  regression; a deterministic `train_test_split`.
- **Synthetic dataset generators**: Gaussian blobs (`make_classification`),
  linear-with-noise (`make_regression`), and the two-moons nonlinear
  benchmark (`make_moons`) — all seeded and dependency-free.
- **JSON model serialization**: a fitted tree or forest round-trips
  through `to_dict()` / `from_dict()` to plain JSON, so the CLI can
  train once and predict later without numpy pickling weirdness.
- **`dtree` CLI**: train on any CSV, evaluate on a held-out split, save
  the model, and predict on new rows — or run instantly against a
  synthetic dataset with `dtree demo`.

## Tech stack

Python 3.10+, standard library only (`csv`, `json`, `argparse`,
`random`, `math`, `collections`). Dev/test tooling: `pytest`.

## How to run

```bash
git clone https://github.com/arnavchawla26/decision-tree-from-scratch.git
cd decision-tree-from-scratch
pip install -e ".[dev]"
pytest              # 106 tests
```

### CLI: train on your own CSV

```bash
dtree train --csv examples/sample_purchase_data.csv --target buys \
    --task classification --model forest --n-estimators 25 \
    --test-size 0.25 --random-state 0
```

Real output from that exact command:

```
Trained forest (classification) on 15 rows, evaluated on 5 held-out rows.

accuracy: 1.0000
macro precision: 1.0000  macro recall: 1.0000  macro f1: 1.0000

feature importances:
  age: 0.4400
  city: 0.3600
  income: 0.2000
```

Add `--out model.json` to save the fitted model, then predict on new
rows later without retraining:

```bash
dtree train --csv examples/sample_purchase_data.csv --target buys \
    --task classification --model forest --out model.json
dtree predict --model model.json --csv new_customers.csv --out predictions.csv
```

`train` auto-detects numeric vs. categorical columns and label-encodes
categoricals (e.g. `city`) consistently between train and predict.
`--task` is `classification` or `regression`; `--model` is `tree` or
`forest`; `--criterion` defaults to `gini` for classification and `mse`
for regression (or pass `entropy` for classification).

### CLI: instant demo, no CSV needed

```bash
dtree demo --task classification --model forest --dataset moons --n-estimators 30
```

Real output:

```
demo: forest classification on 240 train / 60 test rows

accuracy: 0.9833  macro f1: 0.9833

feature importances: [0.4659, 0.5341]
```

### Library usage

```python
from dtree_scratch.tree import DecisionTreeClassifier
from dtree_scratch.datasets import make_moons
from dtree_scratch.metrics import accuracy_score, train_test_split

X, y = make_moons(n_samples=400, noise=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

clf = DecisionTreeClassifier(max_depth=5, random_state=0)
clf.fit(X_train, y_train)
print(accuracy_score(y_test, clf.predict(X_test)))
```

See `examples/classification_demo.py` and `examples/regression_demo.py`
for single-tree-vs-forest comparisons with real captured output:

```
$ python examples/classification_demo.py
train=300 test=100

  single tree: accuracy=0.9500  macro_f1=0.9498
random forest: accuracy=0.9700  macro_f1=0.9696

$ python examples/regression_demo.py
train=300 test=100

  single tree: r2=0.8180  mae=4.9284  feature_importances=[0.484, 0.366, 0.132, 0.019]
random forest: r2=0.9126  mae=3.5632  feature_importances=[0.478, 0.312, 0.162, 0.049]
```

(`make_regression`'s coefficients are deliberately descending, so
feature 0 is always the most predictive — both models correctly rank
it first.)

## Design notes

- **`max_depth` follows the sklearn convention**: it counts edges from
  the root to a leaf, so `max_depth=1` allows exactly one split
  (a root plus two leaves), and a lone-leaf tree has depth 0.
- **Split search** considers every midpoint between adjacent distinct
  sorted values of each candidate feature — the standard CART approach
  for continuous features. It's O(n log n) per feature per node (from
  the sort), not the fastest possible, but it's exact and easy to
  verify against a hand worked example.
- **Ties break deterministically** (smallest label by string form, in
  both leaf majority-vote and forest majority-vote), so the same
  `random_state` always reproduces the same tree/forest and the same
  predictions.
- **Unseen categorical values at predict time** get a fresh trailing
  code instead of raising, so one unfamiliar row doesn't crash a whole
  prediction batch — a deliberate CLI-usability tradeoff over strict
  correctness.

## Project layout

```
dtree_scratch/
  criteria.py     # Gini, entropy, MSE impurity + weighted split scoring
  tree.py         # CART DecisionTreeClassifier / DecisionTreeRegressor
  forest.py       # RandomForestClassifier / RandomForestRegressor (bagging)
  metrics.py      # accuracy, confusion matrix, precision/recall/F1, MSE/RMSE/MAE/R2, train_test_split
  datasets.py     # make_classification, make_regression, make_moons
  csv_utils.py    # CSV loading + categorical encoding for the CLI
  cli.py          # `dtree train|predict|demo`
tests/            # 106 pytest tests across every module
examples/         # library-usage demo scripts + a sample CSV
```

## Current status

**v1 — complete and functional.** Both models (classifier and
regressor), both tree types (single tree and forest), all three
splitting criteria, pruning controls, feature importances, full metric
suite, synthetic dataset generators, JSON model serialization, and the
full `train`/`predict`/`demo` CLI are implemented and covered by 106
passing tests (including a real subprocess run of the installed `dtree`
console script). No known gaps for the stated scope.

Possible future extensions (not started): out-of-bag scoring for
forests, cost-complexity pruning, and a small ASCII tree-printing
utility for inspecting a fitted tree's structure.

## License

MIT — see [LICENSE](LICENSE).
