import pytest

from src.maize_detection.labels import (
    CANONICAL_LABELS,
    CDS_TO_CANONICAL,
    PLANTVILLAGE_TO_CANONICAL,
    validate_mapping,
)


# ── Canonical label list ──────────────────────────────────────────────────────

def test_canonical_labels_exact_set():
    assert set(CANONICAL_LABELS) == {
        "healthy",
        "common_rust",
        "northern_leaf_blight",
        "gray_leaf_spot",
    }


def test_canonical_labels_no_duplicates():
    assert len(CANONICAL_LABELS) == len(set(CANONICAL_LABELS))


# ── PlantVillage mapping ──────────────────────────────────────────────────────

def test_plantvillage_covers_all_four_classes():
    assert set(PLANTVILLAGE_TO_CANONICAL.values()) == set(CANONICAL_LABELS)


def test_plantvillage_healthy():
    assert PLANTVILLAGE_TO_CANONICAL["Corn_(maize)___healthy"] == "healthy"


def test_plantvillage_common_rust():
    assert PLANTVILLAGE_TO_CANONICAL["Corn_(maize)___Common_rust_"] == "common_rust"


def test_plantvillage_nlb():
    assert PLANTVILLAGE_TO_CANONICAL["Corn_(maize)___Northern_Leaf_Blight"] == "northern_leaf_blight"


def test_plantvillage_gls():
    assert PLANTVILLAGE_TO_CANONICAL["Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"] == "gray_leaf_spot"


def test_plantvillage_all_targets_are_canonical():
    for source, target in PLANTVILLAGE_TO_CANONICAL.items():
        assert target in CANONICAL_LABELS, f"Non-canonical target '{target}' for source '{source}'"


# ── CD&S mapping ──────────────────────────────────────────────────────────────

def test_cds_nls_is_excluded():
    """NLS has no canonical match and must be None (excluded), never force-fit."""
    assert CDS_TO_CANONICAL["NLS"] is None


def test_cds_nlb_maps_correctly():
    assert CDS_TO_CANONICAL["NLB"] == "northern_leaf_blight"


def test_cds_gls_maps_correctly():
    assert CDS_TO_CANONICAL["GLS"] == "gray_leaf_spot"


def test_cds_non_none_targets_are_canonical():
    for source, target in CDS_TO_CANONICAL.items():
        if target is not None:
            assert target in CANONICAL_LABELS, f"Non-canonical target '{target}' for CD&S source '{source}'"


# ── validate_mapping() ────────────────────────────────────────────────────────

def test_validate_raises_on_nls_to_nlb():
    """The cardinal mis-mapping: NLS (Spot) ≠ NLB (Blight)."""
    with pytest.raises(ValueError, match="NLS"):
        validate_mapping({"NLS": "northern_leaf_blight"})


def test_validate_raises_on_southern_rust_to_common_rust():
    with pytest.raises(ValueError, match="Southern Rust"):
        validate_mapping({"Southern Rust": "common_rust"})


def test_validate_raises_on_unknown_target():
    with pytest.raises(ValueError):
        validate_mapping({"SomeDisease": "not_a_real_label"})


def test_validate_passes_on_valid_cds_mapping():
    validate_mapping({"NLB": "northern_leaf_blight", "GLS": "gray_leaf_spot", "NLS": None})


def test_validate_passes_on_valid_plantvillage_mapping():
    validate_mapping(PLANTVILLAGE_TO_CANONICAL)


def test_validate_passes_on_none_exclusion():
    validate_mapping({"UnknownClass": None})
