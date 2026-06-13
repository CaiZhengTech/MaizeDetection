"""Tests for the leaf-grouped split (M3).

The CARDINAL test is `test_zero_leaf_overlap_*`: no physical leaf may appear in
more than one split. It runs on a synthetic fixture (always) and on the real
split manifest (when present), so the invariant is proven on the logic itself
regardless of whether the dataset is downloaded.
"""

import csv
from pathlib import Path

import pytest

from src.maize_detection.data import (
    CANONICAL_LABELS,
    DEFAULT_RATIOS,
    SPLIT_NAMES,
    SPLITS_PATH,
    assign_splits,
    leaf_overlap,
    load_splits,
)


# ── Synthetic fixture ─────────────────────────────────────────────────────────

def _make_rows() -> list[dict]:
    """Build a manifest-shaped fixture.

    healthy / NLB / GLS have multi-image leaves (grouping matters); common_rust
    is all singletons (mirrors the real, documented residual-risk case).
    """
    rows: list[dict] = []
    multishot = {
        "healthy": 40,
        "northern_leaf_blight": 30,
        "gray_leaf_spot": 24,
    }
    for label, n_leaves in multishot.items():
        for leaf in range(n_leaves):
            leaf_id = f"{label}:::leaf{leaf}"
            for shot in range(3):  # 3 images per physical leaf
                rows.append({
                    "image_path": f"data/raw/plantvillage/{label}/{leaf}_{shot}.jpg",
                    "canonical_label": label,
                    "leaf_id": leaf_id,
                })
    for i in range(60):  # common_rust singletons
        rows.append({
            "image_path": f"data/raw/plantvillage/common_rust/{i}.jpg",
            "canonical_label": "common_rust",
            "leaf_id": f"common_rust:::rust{i}",
        })
    return rows


@pytest.fixture
def split_rows() -> list[dict]:
    return assign_splits(_make_rows(), DEFAULT_RATIOS, seed=42)


# ── Cardinal invariant ────────────────────────────────────────────────────────

def test_zero_leaf_overlap_synthetic(split_rows):
    for pair, shared in leaf_overlap(split_rows).items():
        assert shared == set(), f"leaf_id leak between {pair}: {sorted(shared)[:5]}"


def test_every_image_of_a_leaf_shares_one_split(split_rows):
    split_by_leaf: dict[str, set[str]] = {}
    for r in split_rows:
        split_by_leaf.setdefault(r["leaf_id"], set()).add(r["split"])
    leaky = {lid: s for lid, s in split_by_leaf.items() if len(s) > 1}
    assert not leaky, f"leaves spanning multiple splits: {list(leaky)[:5]}"


# ── Coverage + sanity ─────────────────────────────────────────────────────────

def test_all_four_classes_in_each_split(split_rows):
    for split in SPLIT_NAMES:
        present = {r["canonical_label"] for r in split_rows if r["split"] == split}
        assert present == set(CANONICAL_LABELS), f"{split} missing {set(CANONICAL_LABELS) - present}"


def test_every_row_assigned_a_valid_split(split_rows):
    assert all(r["split"] in SPLIT_NAMES for r in split_rows)
    assert len(split_rows) == len(_make_rows())


def test_split_is_deterministic():
    rows = _make_rows()
    a = assign_splits(rows, DEFAULT_RATIOS, seed=42)
    b = assign_splits(rows, DEFAULT_RATIOS, seed=42)
    assert [r["split"] for r in a] == [r["split"] for r in b]


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        assign_splits(_make_rows(), {"train": 0.8, "val": 0.15, "test": 0.15}, seed=42)


def test_leaf_split_proportions_are_reasonable(split_rows):
    """Test leaves should be ~15% of leaves in each class (group-level stratified)."""
    for label in CANONICAL_LABELS:
        leaves = {r["leaf_id"] for r in split_rows if r["canonical_label"] == label}
        test_leaves = {
            r["leaf_id"] for r in split_rows
            if r["canonical_label"] == label and r["split"] == "test"
        }
        frac = len(test_leaves) / len(leaves)
        assert 0.05 < frac < 0.30, f"{label} test leaf fraction {frac:.2f} off target 0.15"


# ── Real-data checks (skipped until the split manifest exists) ─────────────────

def test_zero_leaf_overlap_real_manifest():
    if not SPLITS_PATH.exists():
        pytest.skip("split manifest not built yet (run: python -m src.maize_detection.data)")
    rows = load_splits(SPLITS_PATH)
    for pair, shared in leaf_overlap(rows).items():
        assert shared == set(), f"REAL leaf_id leak between {pair}: {sorted(shared)[:5]}"


def test_real_manifest_all_classes_each_split():
    if not SPLITS_PATH.exists():
        pytest.skip("split manifest not built yet")
    rows = load_splits(SPLITS_PATH)
    for split in SPLIT_NAMES:
        present = {r["canonical_label"] for r in rows if r["split"] == split}
        assert present == set(CANONICAL_LABELS), f"{split} missing {set(CANONICAL_LABELS) - present}"
