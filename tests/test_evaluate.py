"""Tests for the in-domain metric computation (M5).

`compute_metrics` is pure (no torch, no files), so we verify the numbers on a
small hand-built case — especially the per-class false-negative rate, the metric
the project cares about most.
"""

import numpy as np

from src.maize_detection.evaluate import compute_metrics

CLASSES = ["healthy", "common_rust", "northern_leaf_blight", "gray_leaf_spot"]


def test_perfect_predictions():
    y = [0, 1, 2, 3, 0, 1, 2, 3]
    m = compute_metrics(y, y, CLASSES)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    for name in CLASSES:
        assert m["per_class"][name]["false_negative_rate"] == 0.0
        assert m["per_class"][name]["recall"] == 1.0


def test_fn_rate_is_one_minus_recall():
    # class 0 (healthy): 4 true; 1 predicted as class 2 -> recall 3/4, FN-rate 1/4
    y_true = [0, 0, 0, 0, 1, 1]
    y_pred = [0, 0, 0, 2, 1, 1]
    m = compute_metrics(y_true, y_pred, CLASSES)
    healthy = m["per_class"]["healthy"]
    assert healthy["support"] == 4
    assert abs(healthy["recall"] - 0.75) < 1e-9
    assert abs(healthy["false_negative_rate"] - 0.25) < 1e-9
    # FN-rate must always equal 1 - recall, per class
    for name in CLASSES:
        c = m["per_class"][name]
        assert abs(c["false_negative_rate"] - (1.0 - c["recall"])) < 1e-9


def test_confusion_matrix_shape_and_counts():
    y_true = [0, 0, 1, 2, 3]
    y_pred = [0, 1, 1, 2, 3]
    m = compute_metrics(y_true, y_pred, CLASSES)
    cm = np.array(m["confusion_matrix"])
    assert cm.shape == (4, 4)
    assert cm.sum() == len(y_true)
    assert cm[0, 0] == 1 and cm[0, 1] == 1  # one healthy correct, one -> common_rust


def test_metrics_are_json_serializable_plain_floats():
    import json
    y = [0, 1, 2, 3]
    m = compute_metrics(y, y, CLASSES)
    json.dumps(m)  # must not raise (no numpy scalar types leaking through)
    assert isinstance(m["accuracy"], float)
    assert isinstance(m["per_class"]["healthy"]["support"], int)
