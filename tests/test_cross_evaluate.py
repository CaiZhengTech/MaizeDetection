"""Tests for the cross-domain metric logic (M6).

`compute_cross_metrics` and `compute_domain_gap` are pure (numpy in, dict out),
so we verify the two views -- open 4-way (with leakage) and restricted 2-class --
on hand-built logits with known argmax behavior. No torch / no images needed.

Class index space (from labels.CANONICAL_LABELS):
    healthy=0, common_rust=1, northern_leaf_blight=2, gray_leaf_spot=3
CD&S contains only NLB(2) and GLS(3).
"""

import json

import numpy as np

from src.maize_detection.cross_evaluate import (
    OVERLAP_IDX,
    OVERLAP_LABELS,
    compute_cross_metrics,
    compute_domain_gap,
)

# Four field images: NLB, NLB, GLS, GLS. Logits are [healthy, rust, NLB, GLS].
Y_TRUE = [2, 2, 3, 3]
LOGITS = np.array([
    [0.0, 0.0, 5.0, 1.0],   # true NLB -> open: NLB (correct),     restricted: NLB (correct)
    [9.0, 0.0, 1.0, 0.0],   # true NLB -> open: healthy (LEAKAGE), restricted: NLB (recovered)
    [0.0, 0.0, 1.0, 5.0],   # true GLS -> open: GLS (correct),     restricted: GLS (correct)
    [0.0, 0.0, 5.0, 1.0],   # true GLS -> open: NLB (confusion),   restricted: NLB (wrong)
])


def test_overlap_indices_are_nlb_and_gls():
    assert OVERLAP_LABELS == ["northern_leaf_blight", "gray_leaf_spot"]
    assert OVERLAP_IDX == [2, 3]


def test_open_4way_accuracy_and_leakage():
    m = compute_cross_metrics(Y_TRUE, LOGITS)["open_4way"]
    # open argmax: [NLB, healthy, GLS, NLB] -> 2 of 4 land on the true class
    assert abs(m["accuracy_on_overlap"] - 0.5) < 1e-9
    # exactly one prediction leaked outside {NLB, GLS}, into healthy
    assert m["leakage"]["total"] == 1
    assert abs(m["leakage"]["rate"] - 0.25) < 1e-9
    assert m["leakage"]["by_class"]["healthy"] == 1
    assert m["leakage"]["by_class"]["common_rust"] == 0
    # per-class recall under the open view
    assert abs(m["per_class"]["northern_leaf_blight"]["recall"] - 0.5) < 1e-9
    assert abs(m["per_class"]["gray_leaf_spot"]["recall"] - 0.5) < 1e-9


def test_restricted_2class_recovers_leaked_prediction():
    m = compute_cross_metrics(Y_TRUE, LOGITS)["restricted_2class"]
    # forced NLB-vs-GLS argmax: [NLB, NLB, GLS, NLB] vs true [NLB, NLB, GLS, GLS]
    assert abs(m["accuracy"] - 0.75) < 1e-9
    # NLB recall 1.0 (both NLB found, incl. the one that leaked to healthy in the open view)
    assert abs(m["per_class"]["northern_leaf_blight"]["recall"] - 1.0) < 1e-9
    # GLS recall 0.5 (one GLS misread as NLB) -> FN-rate 0.5
    assert abs(m["per_class"]["gray_leaf_spot"]["recall"] - 0.5) < 1e-9
    assert abs(m["per_class"]["gray_leaf_spot"]["false_negative_rate"] - 0.5) < 1e-9


def test_n_images_counts_per_class():
    cross = compute_cross_metrics(Y_TRUE, LOGITS)
    assert cross["n_images"] == {"northern_leaf_blight": 2, "gray_leaf_spot": 2}


def test_cross_metrics_json_serializable():
    cross = compute_cross_metrics(Y_TRUE, LOGITS)
    json.dumps(cross)  # must not raise -- no numpy scalars leaking through


def test_domain_gap_arithmetic():
    cross = compute_cross_metrics(Y_TRUE, LOGITS)
    # Pretend in-domain recall was perfect for both overlap classes.
    in_domain = {
        "accuracy": 0.984,
        "per_class": {
            "northern_leaf_blight": {"recall": 1.0},
            "gray_leaf_spot": {"recall": 1.0},
        },
    }
    gap = compute_domain_gap(in_domain, cross)
    # GLS dropped from 1.0 in-domain to 0.5 field (restricted) -> drop 0.5
    assert abs(gap["gray_leaf_spot"]["recall_drop_restricted"] - 0.5) < 1e-9
    # NLB recovered fully under the restricted view -> zero drop
    assert abs(gap["northern_leaf_blight"]["recall_drop_restricted"] - 0.0) < 1e-9
    assert gap["_accuracy"]["in_domain_accuracy_4class"] == 0.984


def test_perfect_field_predictions_have_no_leakage():
    # Every image's max logit is its true class -> recall 1.0, zero leakage.
    perfect = np.array([
        [0, 0, 5, 0],  # NLB
        [0, 0, 5, 0],  # NLB
        [0, 0, 0, 5],  # GLS
        [0, 0, 0, 5],  # GLS
    ], dtype=float)
    cross = compute_cross_metrics(Y_TRUE, perfect)
    assert cross["open_4way"]["leakage"]["total"] == 0
    assert abs(cross["open_4way"]["accuracy_on_overlap"] - 1.0) < 1e-9
    assert abs(cross["restricted_2class"]["accuracy"] - 1.0) < 1e-9
