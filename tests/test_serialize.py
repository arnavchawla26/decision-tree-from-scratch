import json
import random

from dtree_scratch.forest import RandomForestClassifier, RandomForestRegressor
from dtree_scratch.tree import DecisionTreeClassifier, DecisionTreeRegressor


def test_classifier_round_trip_predictions_match():
    X = [[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]]
    y = ["low", "low", "low", "high", "high", "high"]
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X, y)

    d = clf.to_dict()
    json_blob = json.dumps(d)  # must be JSON-serializable
    reloaded = DecisionTreeClassifier.from_dict(json.loads(json_blob))

    assert reloaded.predict(X) == clf.predict(X)
    assert reloaded.classes_ == clf.classes_
    assert reloaded.feature_importances_ == clf.feature_importances_


def test_regressor_round_trip_predictions_match():
    X = [[i] for i in range(20)]
    y = [float(i) * 2 for i in range(20)]
    reg = DecisionTreeRegressor(random_state=0, max_depth=4)
    reg.fit(X, y)

    d = reg.to_dict()
    json_blob = json.dumps(d)
    reloaded = DecisionTreeRegressor.from_dict(json.loads(json_blob))

    assert reloaded.predict(X) == reg.predict(X)


def test_forest_classifier_round_trip_predictions_match():
    rng = random.Random(0)
    X = [[rng.uniform(0, 10), rng.uniform(0, 10)] for _ in range(60)]
    y = ["a" if x[0] + x[1] > 10 else "b" for x in X]
    clf = RandomForestClassifier(n_estimators=6, random_state=0)
    clf.fit(X, y)

    d = clf.to_dict()
    json_blob = json.dumps(d)
    reloaded = RandomForestClassifier.from_dict(json.loads(json_blob))

    assert reloaded.predict(X) == clf.predict(X)
    assert len(reloaded.trees_) == len(clf.trees_)


def test_forest_regressor_round_trip_predictions_match():
    rng = random.Random(1)
    X = [[rng.uniform(-5, 5), rng.uniform(-5, 5)] for _ in range(60)]
    y = [2 * x[0] - x[1] for x in X]
    reg = RandomForestRegressor(n_estimators=6, random_state=0)
    reg.fit(X, y)

    d = reg.to_dict()
    json_blob = json.dumps(d)
    reloaded = RandomForestRegressor.from_dict(json.loads(json_blob))

    assert reloaded.predict(X) == reg.predict(X)
