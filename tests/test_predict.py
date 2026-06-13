"""Tests for the CPU inference function (M7).

`predict` needs a real checkpoint and a real image, so these tests skip cleanly
when either is absent (e.g. on a fresh clone before the Colab `best.pt` is placed).
When they are present, we assert the output CONTRACT -- the shape every caller
(and the future FastAPI wrapper) depends on -- not a specific label.
"""

from pathlib import Path

import pytest

from src.maize_detection.labels import CANONICAL_LABELS
from src.maize_detection.predict import DEFAULT_CHECKPOINT, PROJECT_ROOT, predict


def _first_image() -> Path | None:
    """Any available leaf image: prefer a controlled one, fall back to field."""
    search = [
        PROJECT_ROOT / "data" / "raw" / "plantvillage",
        PROJECT_ROOT / "data" / "external" / "cds_overlap",
    ]
    for root in search:
        if root.exists():
            for img in root.rglob("*.jpg"):
                return img
    return None


needs_assets = pytest.mark.skipif(
    not DEFAULT_CHECKPOINT.exists() or _first_image() is None,
    reason="requires outputs/checkpoints/best.pt and at least one local image",
)


@needs_assets
def test_predict_output_contract():
    result = predict(_first_image())
    assert set(result) == {"label", "confidence", "probabilities"}
    assert result["label"] in CANONICAL_LABELS
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["probabilities"]) == set(CANONICAL_LABELS)


@needs_assets
def test_probabilities_sum_to_one_and_match_confidence():
    result = predict(_first_image())
    probs = result["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    # reported confidence is the probability of the reported label (the argmax)
    assert abs(result["confidence"] - probs[result["label"]]) < 1e-6
    assert result["confidence"] == max(probs.values())
